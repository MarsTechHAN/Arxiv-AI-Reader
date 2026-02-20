"""
Paper storage abstraction with optional SQLite (FTS5 + compression) backend.
Falls back to JSON files when SQLite/FTS5 unavailable or disabled.
"""

import json
import os
import zlib
from pathlib import Path
from typing import List, Dict, Optional, Protocol
from threading import Lock, local
import threading
import queue
import atexit

from models import Paper

# Single source of truth: project_root/data/ (absolute, cwd-independent)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = _PROJECT_ROOT / "data"
DEFAULT_DATA_DIR = str(DATA_ROOT / "papers")
DEFAULT_DB_PATH = str(DATA_ROOT / "papers.db")


def _merge_analysis_into(keep_paper: Paper, from_paper: Paper) -> bool:
    """Copy analysis fields from from_paper into keep_paper when keep lacks them."""
    changed = False
    if keep_paper.is_relevant is None and from_paper.is_relevant is not None:
        keep_paper.is_relevant = from_paper.is_relevant
        changed = True
    if (keep_paper.relevance_score or 0) == 0 and (from_paper.relevance_score or 0) > 0:
        keep_paper.relevance_score = from_paper.relevance_score or 0.0
        changed = True
    if not (keep_paper.one_line_summary or "").strip() and (from_paper.one_line_summary or "").strip():
        keep_paper.one_line_summary = from_paper.one_line_summary or ""
        changed = True
    if not (keep_paper.detailed_summary or "").strip() and (from_paper.detailed_summary or "").strip():
        keep_paper.detailed_summary = from_paper.detailed_summary or ""
        changed = True
    if not (keep_paper.extracted_keywords or []) and (from_paper.extracted_keywords or []):
        keep_paper.extracted_keywords = from_paper.extracted_keywords or []
        changed = True
    if not (keep_paper.qa_pairs or []) and (from_paper.qa_pairs or []):
        keep_paper.qa_pairs = from_paper.qa_pairs or []
        changed = True
    if not (getattr(keep_paper, "tags", []) or []) and (getattr(from_paper, "tags", []) or []):
        keep_paper.tags = getattr(from_paper, "tags", []) or []
        changed = True
    if not keep_paper.is_starred and from_paper.is_starred:
        keep_paper.is_starred = True
        keep_paper.star_category = getattr(from_paper, "star_category", "Other")
        changed = True
    if not keep_paper.is_hidden and from_paper.is_hidden:
        keep_paper.is_hidden = True
        changed = True
    return changed


def _compress(data: bytes) -> bytes:
    return zlib.compress(data, level=6)


def _decompress(data: bytes) -> bytes:
    return zlib.decompress(data)


class PaperStore(Protocol):
    """Abstract paper storage interface."""

    def save_paper(self, paper: Paper) -> None: ...
    def load_paper(self, arxiv_id: str) -> Paper: ...
    def paper_exists(self, arxiv_id: str) -> bool: ...
    def any_version_exists(self, arxiv_id: str) -> tuple[bool, Optional[str]]: ...
    def delete_paper(self, arxiv_id: str) -> None: ...
    def list_papers(self, skip: int, limit: Optional[int]) -> List[Paper]: ...
    def list_papers_metadata(self, max_files: int, check_stale: bool) -> List[dict]: ...
    def search(self, query: str, limit: int, search_full_text: bool) -> List[dict]: ...
    def refresh_metadata_cache(self) -> None: ...


class JSONPaperStore:
    """File-based storage using one JSON file per paper (current implementation)."""

    def __init__(self, data_dir: str = None):
        self.data_dir = Path(data_dir or DEFAULT_DATA_DIR)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._metadata_cache: Dict[str, dict] = {}
        self._cache_lock = Lock()
        self._cache_initialized = False
        self._merge_done = False

    def save_paper(self, paper: Paper) -> None:
        file_path = self.data_dir / f"{paper.id}.json"
        with open(file_path, "w") as f:
            json.dump(paper.to_dict(), f, indent=2, ensure_ascii=False)
        self._update_cache_from_paper(paper, file_path)

    def load_paper(self, arxiv_id: str, resolve_version: bool = True) -> Paper:
        """Load paper by id. If resolve_version=True and exact id not found, loads latest version of same base id."""
        if self.paper_exists(arxiv_id):
            file_path = self.data_dir / f"{arxiv_id}.json"
            with open(file_path) as f:
                return Paper.from_dict(json.load(f))
        if resolve_version:
            exists, latest_id = self.any_version_exists(arxiv_id)
            if exists and latest_id:
                file_path = self.data_dir / f"{latest_id}.json"
                with open(file_path) as f:
                    return Paper.from_dict(json.load(f))
        raise FileNotFoundError(f"Paper {arxiv_id} not found")

    def paper_exists(self, arxiv_id: str) -> bool:
        return (self.data_dir / f"{arxiv_id}.json").exists()

    def _get_base_id(self, arxiv_id: str) -> str:
        return arxiv_id.rsplit("v", 1)[0] if "v" in arxiv_id else arxiv_id

    def _version_number(self, arxiv_id: str) -> int:
        """Extract version for comparison. base_id=0, v1=1, v2=2. Higher=newer."""
        if "v" not in arxiv_id:
            return 0
        try:
            return int(arxiv_id.rsplit("v", 1)[1])
        except (ValueError, IndexError):
            return 0

    def any_version_exists(self, arxiv_id: str) -> tuple[bool, Optional[str]]:
        """Returns (exists, latest_version_id). latest_version_id is the one with highest version number."""
        base_id = self._get_base_id(arxiv_id)
        candidates: List[str] = []
        if self.paper_exists(arxiv_id):
            candidates.append(arxiv_id)
        for f in self.data_dir.glob(f"{base_id}v*.json"):
            candidates.append(f.stem)
        if self.paper_exists(base_id):
            candidates.append(base_id)
        if not candidates:
            return (False, None)
        latest = max(candidates, key=lambda x: self._version_number(x))
        return (True, latest)

    def delete_paper(self, arxiv_id: str) -> None:
        fp = self.data_dir / f"{arxiv_id}.json"
        if fp.exists():
            fp.unlink()
        with self._cache_lock:
            self._metadata_cache.pop(arxiv_id, None)

    def merge_duplicate_versions(self) -> int:
        """Merge duplicate versions: keep latest per base_id, delete others. Preserves analysis from deleted into kept."""
        deleted = 0
        by_base: Dict[str, List[str]] = {}
        for fp in self.data_dir.glob("*.json"):
            pid = fp.stem
            base = self._get_base_id(pid)
            by_base.setdefault(base, []).append(pid)
        for base, ids in by_base.items():
            if len(ids) <= 1:
                continue
            keep = max(ids, key=lambda x: self._version_number(x))
            keep_paper = None
            for pid in ids:
                if pid == keep:
                    try:
                        keep_paper = Paper.from_dict(json.loads((self.data_dir / f"{pid}.json").read_text()))
                    except Exception:
                        pass
                    break
            for pid in ids:
                if pid != keep and keep_paper is not None:
                    try:
                        other = Paper.from_dict(json.loads((self.data_dir / f"{pid}.json").read_text()))
                        if _merge_analysis_into(keep_paper, other):
                            self.save_paper(keep_paper)
                    except Exception:
                        pass
                    self.delete_paper(pid)
                    deleted += 1
        return deleted

    def list_papers(self, skip: int, limit: Optional[int]) -> List[Paper]:
        files = sorted(
            self.data_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        rng = files[skip:] if limit is None or limit <= 0 else files[skip : skip + limit]
        papers = []
        for fp in rng:
            try:
                with open(fp) as f:
                    papers.append(Paper.from_dict(json.load(f)))
            except Exception as e:
                print(f"Warning: Failed to load {fp.name}: {e}")
        return papers

    def _extract_metadata(self, data: dict, paper_id: str, file_path: Path, mtime: float) -> dict:
        return {
            "id": data.get("id", paper_id),
            "file_path": file_path,
            "mtime": mtime,
            "is_relevant": data.get("is_relevant"),
            "title": data.get("title", ""),
            "is_starred": data.get("is_starred", False),
            "is_hidden": data.get("is_hidden", False),
            "star_category": data.get("star_category", "Other"),
            "relevance_score": data.get("relevance_score", 0.0),
            "published_date": data.get("published_date", ""),
            "created_at": data.get("created_at", ""),
            "extracted_keywords": data.get("extracted_keywords", []),
            "detailed_summary": data.get("detailed_summary", ""),
            "one_line_summary": data.get("one_line_summary", ""),
            "abstract": data.get("abstract", ""),
            "authors": data.get("authors", []),
            "tags": data.get("tags", []),
            "preview_text": data.get("preview_text", ""),
        }

    def _update_cache_from_paper(self, paper: Paper, file_path: Path) -> None:
        try:
            mtime = file_path.stat().st_mtime
        except OSError:
            mtime = 0
        meta = {
            "id": paper.id,
            "file_path": file_path,
            "mtime": mtime,
            "title": paper.title,
            "is_relevant": paper.is_relevant,
            "is_starred": paper.is_starred,
            "is_hidden": paper.is_hidden,
            "star_category": getattr(paper, "star_category", "Other"),
            "relevance_score": paper.relevance_score,
            "published_date": paper.published_date,
            "created_at": paper.created_at,
            "extracted_keywords": paper.extracted_keywords,
            "detailed_summary": paper.detailed_summary,
            "one_line_summary": paper.one_line_summary,
            "abstract": paper.abstract,
            "authors": paper.authors,
            "tags": getattr(paper, "tags", []),
            "preview_text": getattr(paper, "preview_text", ""),
        }
        with self._cache_lock:
            self._metadata_cache[paper.id] = meta

    def _refresh_metadata_cache(self) -> None:
        print("🔄 Refreshing metadata cache...")
        if not self._merge_done:
            merged = self.merge_duplicate_versions()
            if merged:
                print(f"  Merged {merged} duplicate version(s), kept latest")
            self._merge_done = True
        files = list(self.data_dir.glob("*.json"))
        new_cache = {}
        for fp in files:
            try:
                mtime = fp.stat().st_mtime
                pid = fp.stem
                if pid in self._metadata_cache and self._metadata_cache[pid].get("mtime") == mtime:
                    new_cache[pid] = self._metadata_cache[pid]
                    continue
                with open(fp) as f:
                    data = json.load(f)
                    new_cache[pid] = self._extract_metadata(data, pid, fp, mtime)
            except Exception as e:
                print(f"Warning: Failed to read {fp.name}: {e}")
        with self._cache_lock:
            self._metadata_cache = new_cache
        print(f"✓ Metadata cache refreshed: {len(self._metadata_cache)} papers")
        self._cache_initialized = True

    def _refresh_stale_cache_entries(self) -> None:
        with self._cache_lock:
            for pid, meta in list(self._metadata_cache.items()):
                fp = meta.get("file_path")
                fp = Path(fp) if isinstance(fp, str) else fp
                if not fp or not fp.exists():
                    continue
                try:
                    mtime = fp.stat().st_mtime
                    if mtime > meta.get("mtime", 0):
                        with open(fp) as f:
                            data = json.load(f)
                            self._metadata_cache[pid] = self._extract_metadata(data, pid, fp, mtime)
                except (OSError, json.JSONDecodeError):
                    pass

    def list_papers_metadata(self, max_files: int, check_stale: bool) -> List[dict]:
        if not self._cache_initialized:
            self._refresh_metadata_cache()
        if check_stale:
            self._refresh_stale_cache_entries()
        with self._cache_lock:
            meta_list = list(self._metadata_cache.values())
        meta_list.sort(key=lambda m: m.get("mtime", 0), reverse=True)
        return meta_list[:max_files]

    def search(self, query: str, limit: int, search_full_text: bool) -> List[dict]:
        return []

    def refresh_metadata_cache(self) -> None:
        self._cache_initialized = False
        self._refresh_metadata_cache()


def _sqlite_connect(db_path: Path, wal: bool = True):
    """Create SQLite connection with WAL mode for read concurrency."""
    import sqlite3
    conn = sqlite3.connect(str(db_path), check_same_thread=False, timeout=30)
    conn.row_factory = sqlite3.Row
    if wal:
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=10000")
        except Exception:
            pass
    return conn


class SQLitePaperStore:
    """
    SQLite backend with FTS5 full-text search and zlib-compressed storage.
    Thread-safe: per-thread connections, WAL mode, write queue (reads prioritized).
    """

    def __init__(self, db_path: str = "data/papers.db", json_fallback_dir: str = "data/papers"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.json_dir = Path(json_fallback_dir)
        self._metadata_cache: Dict[str, dict] = {}
        self._cache_lock = Lock()
        self._cache_initialized = False
        self._trigram_available = False
        self._local = local()
        self._write_queue = queue.Queue()
        self._write_stop = threading.Event()
        self._write_thread = threading.Thread(target=self._write_worker, daemon=True)
        self._write_thread.start()
        atexit.register(self._stop_write_worker)
        self._init_db()

    def _get_conn(self):
        """Per-thread connection for reads (thread-safe)."""
        try:
            conn = self._local.conn
        except AttributeError:
            conn = None
        if conn is None:
            conn = _sqlite_connect(self.db_path)
            self._local.conn = conn
        return conn

    def _stop_write_worker(self):
        self._write_stop.set()
        self._write_queue.put(None)
        if self._write_thread.is_alive():
            self._write_thread.join(timeout=5)

    def _write_worker(self):
        """Dedicated thread for writes - reads never block on writes."""
        import sqlite3
        conn = _sqlite_connect(self.db_path)
        while not self._write_stop.is_set():
            try:
                item = self._write_queue.get(timeout=0.5)
                if item is None:
                    break
                op, args = item
                if op == "save":
                    self._save_internal(args[0], conn=conn, commit=True)
                elif op == "delete":
                    self._fts_delete(conn, args[0])
                    conn.execute("DELETE FROM papers WHERE id = ?", (args[0],))
                    conn.commit()
                    with self._cache_lock:
                        self._metadata_cache.pop(args[0], None)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Warning: Write worker failed: {e}")
        conn.close()

    def _init_db(self) -> None:
        import sqlite3
        conn = self._get_conn()
        try:
            conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS _fts_check USING fts5(content)")
            conn.execute("DROP TABLE _fts_check")
        except sqlite3.OperationalError as e:
            raise RuntimeError(f"FTS5 not available: {e}")

        conn.executescript("""
            CREATE TABLE IF NOT EXISTS papers (
                id TEXT PRIMARY KEY,
                data BLOB NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS papers_fts_map (
                rowid INTEGER PRIMARY KEY,
                paper_id TEXT UNIQUE NOT NULL
            );
        """)
        self._use_fts_contentless = self._create_fts_table(conn)
        self._ensure_fts_contentless(conn)
        conn.commit()
        self._trigram_available = self._ensure_fts_trigram(conn)
        if self._trigram_available:
            conn.commit()
        self._migrate_from_json()
        self._merge_duplicate_versions_sqlite(conn)
        conn.commit()

    def _merge_duplicate_versions_sqlite(self, conn) -> None:
        """Merge duplicate versions in SQLite: keep latest per base_id, delete others. Preserves analysis."""
        by_base: Dict[str, List[str]] = {}
        for (pid,) in conn.execute("SELECT id FROM papers").fetchall():
            base = self._get_base_id(pid)
            by_base.setdefault(base, []).append(pid)
        for base, ids in by_base.items():
            if len(ids) <= 1:
                continue
            keep = max(ids, key=lambda x: self._version_number(x))
            keep_row = conn.execute("SELECT data FROM papers WHERE id = ?", (keep,)).fetchone()
            if keep_row:
                try:
                    keep_paper = Paper.from_dict(json.loads(_decompress(keep_row[0]).decode("utf-8")))
                    for pid in ids:
                        if pid != keep:
                            row = conn.execute("SELECT data FROM papers WHERE id = ?", (pid,)).fetchone()
                            if row:
                                try:
                                    other = Paper.from_dict(json.loads(_decompress(row[0]).decode("utf-8")))
                                    if _merge_analysis_into(keep_paper, other):
                                        self._save_internal(keep_paper, conn=conn, commit=False)
                                except Exception:
                                    pass
                            self._fts_delete(conn, pid)
                            conn.execute("DELETE FROM papers WHERE id = ?", (pid,))
                            with self._cache_lock:
                                self._metadata_cache.pop(pid, None)
                except Exception:
                    for pid in ids:
                        if pid != keep:
                            self._fts_delete(conn, pid)
                            conn.execute("DELETE FROM papers WHERE id = ?", (pid,))
                            with self._cache_lock:
                                self._metadata_cache.pop(pid, None)
            else:
                for pid in ids:
                    if pid != keep:
                        self._fts_delete(conn, pid)
                        conn.execute("DELETE FROM papers WHERE id = ?", (pid,))
                        with self._cache_lock:
                            self._metadata_cache.pop(pid, None)

    def _create_fts_table(self, conn) -> bool:
        """Create FTS table. Returns True if contentless (content='') is used."""
        import sqlite3
        if conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='papers_fts'"
        ).fetchone():
            return True  # exists, _ensure_fts_contentless will handle migration
        try:
            conn.execute("""
                CREATE VIRTUAL TABLE papers_fts USING fts5(
                    paper_id UNINDEXED,
                    content,
                    content='',
                    contentless_delete=1,
                    tokenize='unicode61'
                )
            """)
            return True
        except sqlite3.OperationalError as e:
            if "contentless" in str(e).lower() or "no such" in str(e).lower():
                conn.execute("""
                    CREATE VIRTUAL TABLE papers_fts USING fts5(
                        paper_id UNINDEXED,
                        content,
                        tokenize='unicode61'
                    )
                """)
                print("  ⚠️ SQLite <3.43: using full FTS (larger DB). Upgrade for contentless.")
                return False
            raise

    def _ensure_fts_contentless(self, conn) -> None:
        """Rebuild papers_fts as contentless if it exists with old schema (stores duplicate content)."""
        try:
            # Check if papers_fts_content exists (old schema stores content)
            r = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='papers_fts_content'"
            ).fetchone()
            if r:
                print("  Rebuilding FTS as contentless (reduces DB size)...")
                conn.execute("DROP TABLE IF EXISTS papers_fts")
                import sqlite3
                try:
                    conn.execute("""
                        CREATE VIRTUAL TABLE papers_fts USING fts5(
                            paper_id UNINDEXED,
                            content,
                            content='',
                            contentless_delete=1,
                            tokenize='unicode61'
                        )
                    """)
                except sqlite3.OperationalError:
                    conn.execute("""
                        CREATE VIRTUAL TABLE papers_fts USING fts5(
                            paper_id UNINDEXED,
                            content,
                            tokenize='unicode61'
                        )
                    """)
                    print("  ⚠️ SQLite <3.43: keeping full FTS (larger DB)")
                # Rebuild index from papers
                rows = conn.execute("SELECT id, data FROM papers").fetchall()
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS papers_fts_map (rowid INTEGER PRIMARY KEY, paper_id TEXT UNIQUE NOT NULL)"
                )
                for i, (pid, blob) in enumerate(rows):
                    try:
                        data = json.loads(_decompress(blob).decode("utf-8"))
                        paper = Paper.from_dict(data)
                        content = self._fts_content(paper)
                        conn.execute(
                            "INSERT INTO papers_fts(paper_id, content) VALUES (?, ?)",
                            (pid, content),
                        )
                        rid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
                        conn.execute(
                            "INSERT OR REPLACE INTO papers_fts_map (rowid, paper_id) VALUES (?, ?)",
                            (rid, pid),
                        )
                    except Exception as e:
                        print(f"Warning: FTS rebuild failed for {pid}: {e}")
                    if (i + 1) % 2000 == 0:
                        print(f"  Rebuilt FTS: {i + 1}/{len(rows)}...")
                conn.commit()
                print(f"✓ FTS rebuild complete: {len(rows)} papers")
        except Exception as e:
            print(f"Warning: FTS contentless check failed: {e}")

    def _ensure_fts_trigram(self, conn) -> bool:
        """Create and populate papers_fts_trigram for ngram/fuzzy search. Returns True if available."""
        import sqlite3
        if conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='papers_fts_trigram'"
        ).fetchone():
            return True
        try:
            conn.execute("""
                CREATE VIRTUAL TABLE papers_fts_trigram USING fts5(
                    paper_id UNINDEXED,
                    content,
                    tokenize='trigram'
                )
            """)
            conn.execute(
                "CREATE TABLE IF NOT EXISTS papers_fts_trigram_map (rowid INTEGER PRIMARY KEY, paper_id TEXT UNIQUE NOT NULL)"
            )
            rows = conn.execute("SELECT id, data FROM papers").fetchall()
            for pid, blob in rows:
                try:
                    data = json.loads(_decompress(blob).decode("utf-8"))
                    paper = Paper.from_dict(data)
                    content = self._fts_content(paper)
                    conn.execute(
                        "INSERT INTO papers_fts_trigram(paper_id, content) VALUES (?, ?)",
                        (pid, content),
                    )
                    rid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
                    conn.execute(
                        "INSERT OR REPLACE INTO papers_fts_trigram_map (rowid, paper_id) VALUES (?, ?)",
                        (rid, pid),
                    )
                except Exception as e:
                    print(f"Warning: Trigram FTS insert failed for {pid}: {e}")
            return True
        except sqlite3.OperationalError as e:
            if "trigram" in str(e).lower() or "tokenize" in str(e).lower():
                return False
            raise

    def _migrate_from_json(self) -> None:
        """Import existing JSON papers into SQLite on first use (batch)."""
        if not self.json_dir.exists():
            return
        conn = self._get_conn()
        existing_ids = set(r[0] for r in conn.execute("SELECT id FROM papers").fetchall())
        # For each base_id, track best version we have (skip migrating older versions)
        by_base: Dict[str, int] = {}
        for pid in existing_ids:
            base = self._get_base_id(pid)
            v = self._version_number(pid)
            by_base[base] = max(by_base.get(base, 0), v)
        migrated = 0
        for fp in self.json_dir.glob("*.json"):
            try:
                pid = fp.stem
                if pid in existing_ids:
                    continue
                base = self._get_base_id(pid)
                json_ver = self._version_number(pid)
                if json_ver <= by_base.get(base, 0):
                    continue  # Skip: we already have newer/equal version
                with open(fp) as f:
                    data = json.load(f)
                paper = Paper.from_dict(data)
                self._save_internal(paper, conn, commit=False)
                existing_ids.add(pid)
                by_base[base] = max(by_base.get(base, 0), json_ver)
                migrated += 1
                if migrated <= 3 or migrated % 1000 == 0:
                    print(f"  Migrated {migrated} papers to SQLite...")
            except Exception as e:
                print(f"Warning: Migration failed for {fp.name}: {e}")
        conn.commit()
        if migrated:
            print(f"✓ Migration complete: {migrated} papers")

    def _fts_content(self, paper: Paper) -> str:
        """FTS index: title, authors, abstract, AI summaries, tags, keywords only (no full text)."""
        parts = [
            paper.title or "",
            paper.abstract or "",
            " ".join(paper.authors or []),
            paper.one_line_summary or "",
            paper.detailed_summary or "",
            " ".join(getattr(paper, "tags", []) or []),
            " ".join(paper.extracted_keywords or []),
        ]
        return " ".join(parts)

    def _save_internal(self, paper: Paper, conn=None, commit: bool = True) -> None:
        import time
        conn = conn or self._get_conn()
        data = _compress(json.dumps(paper.to_dict(), ensure_ascii=False).encode("utf-8"))
        updated = time.time()
        conn.execute(
            "INSERT OR REPLACE INTO papers (id, data, updated_at) VALUES (?, ?, ?)",
            (paper.id, data, updated),
        )
        content = self._fts_content(paper)
        try:
            self._fts_delete(conn, paper.id)
            conn.execute(
                "INSERT INTO papers_fts(paper_id, content) VALUES (?, ?)",
                (paper.id, content),
            )
            rid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            conn.execute(
                "INSERT OR REPLACE INTO papers_fts_map (rowid, paper_id) VALUES (?, ?)",
                (rid, paper.id),
            )
            if getattr(self, "_trigram_available", False):
                try:
                    conn.execute(
                        "INSERT INTO papers_fts_trigram(paper_id, content) VALUES (?, ?)",
                        (paper.id, content),
                    )
                    tri_rid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
                    conn.execute(
                        "INSERT OR REPLACE INTO papers_fts_trigram_map (rowid, paper_id) VALUES (?, ?)",
                        (tri_rid, paper.id),
                    )
                except Exception as te:
                    print(f"Warning: Trigram FTS insert failed for {paper.id}: {te}")
                except Exception as te:
                    print(f"Warning: Trigram FTS insert failed for {paper.id}: {te}")
        except Exception as e:
            print(f"Warning: FTS insert failed for {paper.id}: {e}")
        if commit:
            conn.commit()

    def save_paper(self, paper: Paper) -> None:
        """Queue write (non-blocking). Reads have priority; writes are deferred."""
        import time
        updated = time.time()
        meta = {
            "id": paper.id,
            "file_path": self.db_path,
            "mtime": updated,
            "title": paper.title,
            "is_relevant": paper.is_relevant,
            "is_starred": paper.is_starred,
            "is_hidden": paper.is_hidden,
            "star_category": getattr(paper, "star_category", "Other"),
            "relevance_score": paper.relevance_score,
            "published_date": paper.published_date,
            "created_at": paper.created_at,
            "extracted_keywords": paper.extracted_keywords,
            "detailed_summary": paper.detailed_summary,
            "one_line_summary": paper.one_line_summary,
            "abstract": paper.abstract,
            "authors": paper.authors,
            "tags": getattr(paper, "tags", []),
            "preview_text": getattr(paper, "preview_text", ""),
        }
        with self._cache_lock:
            self._metadata_cache[paper.id] = meta
        self._write_queue.put(("save", (paper,)))

    def load_paper(self, arxiv_id: str, resolve_version: bool = True) -> Paper:
        """Load paper by id. If resolve_version=True and exact id not found, loads latest version of same base id."""
        row = self._get_conn().execute("SELECT data FROM papers WHERE id = ?", (arxiv_id,)).fetchone()
        if row:
            data = json.loads(_decompress(row[0]).decode("utf-8"))
            return Paper.from_dict(data)
        if resolve_version:
            exists, latest_id = self.any_version_exists(arxiv_id)
            if exists and latest_id:
                row = self._get_conn().execute("SELECT data FROM papers WHERE id = ?", (latest_id,)).fetchone()
                if row:
                    data = json.loads(_decompress(row[0]).decode("utf-8"))
                    return Paper.from_dict(data)
        raise FileNotFoundError(f"Paper {arxiv_id} not found")

    def paper_exists(self, arxiv_id: str) -> bool:
        r = self._get_conn().execute("SELECT 1 FROM papers WHERE id = ?", (arxiv_id,)).fetchone()
        return r is not None

    def _get_base_id(self, arxiv_id: str) -> str:
        return arxiv_id.rsplit("v", 1)[0] if "v" in arxiv_id else arxiv_id

    def _version_number(self, arxiv_id: str) -> int:
        """Extract version for comparison. base_id=0, v1=1, v2=2. Higher=newer."""
        if "v" not in arxiv_id:
            return 0
        try:
            return int(arxiv_id.rsplit("v", 1)[1])
        except (ValueError, IndexError):
            return 0

    def any_version_exists(self, arxiv_id: str) -> tuple[bool, Optional[str]]:
        """Returns (exists, latest_version_id). latest_version_id is the one with highest version number."""
        base_id = self._get_base_id(arxiv_id)
        rows = self._get_conn().execute(
            "SELECT id FROM papers WHERE id LIKE ? OR id = ?",
            (f"{base_id}v%", base_id),
        ).fetchall()
        if not rows:
            return (False, None)
        candidates = [r[0] for r in rows]
        latest = max(candidates, key=lambda x: self._version_number(x))
        return (True, latest)

    def _fts_delete(self, conn, paper_id: str) -> None:
        """Remove paper from FTS. Contentless: use rowid map. Full FTS: WHERE paper_id."""
        try:
            rows = conn.execute(
                "SELECT rowid FROM papers_fts_map WHERE paper_id = ?", (paper_id,)
            ).fetchall()
            if rows:
                for (rid,) in rows:
                    conn.execute("DELETE FROM papers_fts WHERE rowid = ?", (rid,))
                conn.execute("DELETE FROM papers_fts_map WHERE paper_id = ?", (paper_id,))
            else:
                conn.execute("DELETE FROM papers_fts WHERE paper_id = ?", (paper_id,))
            if getattr(self, "_trigram_available", False):
                tri_rows = conn.execute(
                    "SELECT rowid FROM papers_fts_trigram_map WHERE paper_id = ?",
                    (paper_id,),
                ).fetchall()
                for (rid,) in tri_rows:
                    conn.execute("DELETE FROM papers_fts_trigram WHERE rowid = ?", (rid,))
                conn.execute(
                    "DELETE FROM papers_fts_trigram_map WHERE paper_id = ?", (paper_id,)
                )
        except Exception:
            conn.execute("DELETE FROM papers_fts WHERE paper_id = ?", (paper_id,))

    def delete_paper(self, arxiv_id: str) -> None:
        """Queue delete (non-blocking). Reads have priority."""
        with self._cache_lock:
            self._metadata_cache.pop(arxiv_id, None)
        self._write_queue.put(("delete", (arxiv_id,)))

    def list_papers(self, skip: int, limit: Optional[int]) -> List[Paper]:
        q = "SELECT id FROM papers ORDER BY updated_at DESC"
        if limit is not None and limit > 0:
            q += f" LIMIT {limit} OFFSET {skip}"
        else:
            q += f" LIMIT -1 OFFSET {skip}"
        rows = self._get_conn().execute(q).fetchall()
        return [self.load_paper(r[0]) for r in rows]

    def _refresh_metadata_cache(self) -> None:
        print("🔄 Refreshing metadata cache (SQLite)...")
        rows = self._get_conn().execute(
            "SELECT id, data, updated_at FROM papers"
        ).fetchall()
        new_cache = {}
        for r in rows:
            try:
                data = json.loads(_decompress(r[1]).decode("utf-8"))
                new_cache[r[0]] = {
                    "id": data.get("id", r[0]),
                    "file_path": self.db_path,
                    "mtime": r[2],
                    "is_relevant": data.get("is_relevant"),
                    "title": data.get("title", ""),
                    "is_starred": data.get("is_starred", False),
                    "is_hidden": data.get("is_hidden", False),
                    "star_category": data.get("star_category", "Other"),
                    "relevance_score": data.get("relevance_score", 0.0),
                    "published_date": data.get("published_date", ""),
                    "created_at": data.get("created_at", ""),
                    "extracted_keywords": data.get("extracted_keywords", []),
                    "detailed_summary": data.get("detailed_summary", ""),
                    "one_line_summary": data.get("one_line_summary", ""),
                    "abstract": data.get("abstract", ""),
                    "authors": data.get("authors", []),
                    "tags": data.get("tags", []),
                    "preview_text": data.get("preview_text", ""),
                }
            except Exception as e:
                print(f"Warning: Failed to cache {r[0]}: {e}")
        with self._cache_lock:
            self._metadata_cache = new_cache
        print(f"✓ Metadata cache refreshed: {len(self._metadata_cache)} papers")
        self._cache_initialized = True

    def list_papers_metadata(self, max_files: int, check_stale: bool) -> List[dict]:
        if not self._cache_initialized:
            self._refresh_metadata_cache()
        with self._cache_lock:
            meta_list = list(self._metadata_cache.values())
        meta_list.sort(key=lambda m: m.get("mtime", 0), reverse=True)
        return meta_list[:max_files]

    def search(self, query: str, limit: int, search_full_text: bool) -> List[dict]:
        """FTS5 search with phrase support and optional trigram merge."""
        import sqlite3
        from search_utils import normalize_fts_query
        conn = self._get_conn()
        q_clean = normalize_fts_query(query.strip())
        if not q_clean:
            return []
        score_by_pid: Dict[str, float] = {}
        try:
            use_contentless = not conn.execute(
                "SELECT 1 FROM sqlite_master WHERE name='papers_fts_content'"
            ).fetchone()
            if use_contentless:
                rows = conn.execute(
                    """SELECT rowid, bm25(papers_fts) as score
                       FROM papers_fts WHERE papers_fts MATCH ?
                       ORDER BY score DESC LIMIT ?""",
                    (q_clean, limit * 2),
                ).fetchall()
                pid_list = []
                for rid, _ in rows:
                    r = conn.execute(
                        "SELECT paper_id FROM papers_fts_map WHERE rowid = ?", (rid,)
                    ).fetchone()
                    pid_list.append(r[0] if r else None)
                rows = [(pid, s) for (_, s), pid in zip(rows, pid_list) if pid]
            else:
                rows = conn.execute(
                    """SELECT paper_id, bm25(papers_fts) as score
                       FROM papers_fts WHERE papers_fts MATCH ?
                       ORDER BY score DESC LIMIT ?""",
                    (q_clean, limit * 2),
                ).fetchall()
            for pid, s in rows:
                sc = -float(s) if s else 0.0
                score_by_pid[pid] = max(score_by_pid.get(pid, 0), sc)
            if getattr(self, "_trigram_available", False):
                from search_utils import tokenize
                q_trigram = " ".join(tokenize(query.strip(), stem=False)) or q_clean
                if q_trigram:
                    try:
                        tri_rows = conn.execute(
                            """SELECT rowid, bm25(papers_fts_trigram) as score
                               FROM papers_fts_trigram WHERE papers_fts_trigram MATCH ?
                               ORDER BY score DESC LIMIT ?""",
                            (q_trigram, limit * 2),
                        ).fetchall()
                        for rid, s in tri_rows:
                            r = conn.execute(
                                "SELECT paper_id FROM papers_fts_trigram_map WHERE rowid = ?",
                                (rid,),
                            ).fetchone()
                            if r:
                                pid = r[0]
                                sc = -float(s) if s else 0.0
                                score_by_pid[pid] = max(
                                    score_by_pid.get(pid, 0), sc * 0.7
                                )
                    except sqlite3.OperationalError:
                        pass
        except sqlite3.OperationalError as e:
            if "syntax error" in str(e).lower() or "fts5" in str(e).lower():
                return []
            raise
        rows = sorted(score_by_pid.items(), key=lambda x: -x[1])[:limit]
        results = []
        for pid, score in rows:
            try:
                paper = self.load_paper(pid)
                meta = self._metadata_cache.get(pid, {})
                if meta.get("is_hidden", False):
                    continue
                results.append({
                    "id": paper.id,
                    "search_score": max(0.01, score),
                    "title": paper.title,
                    "authors": paper.authors,
                    "abstract": paper.abstract[:300] + "..." if len(paper.abstract or "") > 300 else (paper.abstract or ""),
                    "url": paper.url,
                    "one_line_summary": paper.one_line_summary,
                    "detailed_summary": (paper.detailed_summary or "")[:500],
                    "published_date": getattr(paper, "published_date", "") or "",
                })
            except Exception:
                continue
        return results[:limit]

    def refresh_metadata_cache(self) -> None:
        self._cache_initialized = False
        self._refresh_metadata_cache()


def get_paper_store(
    data_dir: str = None,
    db_path: str = None,
    force_json: bool = None,
) -> PaperStore:
    """
    Get paper store. Prefers SQLite (FTS5 + compression) when available.
    Set env ARXIV_USE_JSON_STORAGE=1 to force JSON. Falls back to JSON on any SQLite/FTS5 error.
    """
    if force_json is None:
        force_json = os.environ.get("ARXIV_USE_JSON_STORAGE", "").lower() in ("1", "true", "yes")
    if force_json:
        print("📁 Using JSON file storage (forced by ARXIV_USE_JSON_STORAGE)")
        return JSONPaperStore(data_dir=data_dir or DEFAULT_DATA_DIR)

    _data_dir = data_dir or DEFAULT_DATA_DIR
    _db_path = db_path or DEFAULT_DB_PATH
    try:
        store = SQLitePaperStore(db_path=_db_path, json_fallback_dir=_data_dir)
        print("📦 Using SQLite storage (FTS5 + compression)")
        return store
    except Exception as e:
        print(f"⚠️ SQLite unavailable ({e}), falling back to JSON storage")
        return JSONPaperStore(data_dir=_data_dir)
