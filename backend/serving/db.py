"""
SQLite DB for serving mode: users, configs, TOTP, invite codes, paper_user_results.
Separate from papers.db (paper metadata + AI results).
"""

import json
import os
import sqlite3
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, List
from threading import Lock

from models import Config


def _config_hash(code: str) -> str:
    return hashlib.sha256(code.encode()).hexdigest()[:16]


class ServingDB:
    """SQLite backend for serving-mode business data."""

    def __init__(self, db_path: str = None):
        from storage import DATA_ROOT
        self.db_path = Path(db_path or str(DATA_ROOT / "serving.db"))
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=10000")
        except Exception:
            pass
        return conn

    def _init_db(self) -> None:
        with self._lock:
            conn = self._get_conn()
            try:
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT UNIQUE NOT NULL,
                        totp_secret TEXT NOT NULL,
                        invite_code_used TEXT,
                        created_at TEXT DEFAULT (datetime('now'))
                    );
                    CREATE TABLE IF NOT EXISTS invite_codes (
                        code TEXT PRIMARY KEY,
                        created_at TEXT DEFAULT (datetime('now')),
                        used_by INTEGER,
                        FOREIGN KEY (used_by) REFERENCES users(id)
                    );
                    CREATE TABLE IF NOT EXISTS user_configs (
                        user_id INTEGER PRIMARY KEY,
                        config_json TEXT NOT NULL,
                        updated_at TEXT DEFAULT (datetime('now')),
                        FOREIGN KEY (user_id) REFERENCES users(id)
                    );
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_token TEXT PRIMARY KEY,
                        user_id INTEGER NOT NULL,
                        created_at TEXT DEFAULT (datetime('now')),
                        FOREIGN KEY (user_id) REFERENCES users(id)
                    );
                    CREATE TABLE IF NOT EXISTS paper_user_results (
                        paper_id TEXT NOT NULL,
                        user_id INTEGER NOT NULL,
                        is_relevant INTEGER,
                        relevance_score REAL DEFAULT 0,
                        extracted_keywords TEXT,
                        one_line_summary TEXT,
                        detailed_summary TEXT,
                        tags TEXT,
                        qa_pairs_json TEXT,
                        is_starred INTEGER DEFAULT 0,
                        is_hidden INTEGER DEFAULT 0,
                        star_category TEXT DEFAULT 'Other',
                        updated_at TEXT DEFAULT (datetime('now')),
                        PRIMARY KEY (paper_id, user_id),
                        FOREIGN KEY (user_id) REFERENCES users(id)
                    );
                    CREATE INDEX IF NOT EXISTS idx_paper_user_results_user ON paper_user_results(user_id);
                """)
                conn.commit()
                init_code = os.environ.get("ARXIV_INIT_INVITE_CODE", "").strip()
                if init_code:
                    conn.execute("INSERT OR IGNORE INTO invite_codes (code) VALUES (?)", (init_code,))
                    conn.commit()
                    print(f"✓ Seeded invite code (from ARXIV_INIT_INVITE_CODE)")
            finally:
                conn.close()

    def create_user(self, username: str, totp_secret: str, invite_code: str) -> int:
        """Create user, consume invite code. Returns user_id."""
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    "INSERT INTO users (username, totp_secret, invite_code_used) VALUES (?, ?, ?)",
                    (username, totp_secret, invite_code)
                )
                user_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
                conn.execute(
                    "UPDATE invite_codes SET used_by = ? WHERE code = ?",
                    (user_id, invite_code)
                )
                conn.commit()
                return user_id
            finally:
                conn.close()

    def get_user_by_username(self, username: str) -> Optional[Dict]:
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT id, username, totp_secret FROM users WHERE username = ?",
                (username,)
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def get_user_by_id(self, user_id: int) -> Optional[Dict]:
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT id, username, totp_secret FROM users WHERE id = ?",
                (user_id,)
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def create_invite_code(self, code: str) -> None:
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO invite_codes (code) VALUES (?)",
                    (code,)
                )
                conn.commit()
            finally:
                conn.close()

    def create_session(self, user_id: int) -> str:
        """Create session, return token."""
        import secrets
        token = secrets.token_urlsafe(32)
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    "INSERT INTO sessions (session_token, user_id) VALUES (?, ?)",
                    (token, user_id)
                )
                conn.commit()
                return token
            finally:
                conn.close()

    def get_session_user(self, token: str) -> Optional[int]:
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT user_id FROM sessions WHERE session_token = ?",
                (token,)
            ).fetchone()
            return row[0] if row else None
        finally:
            conn.close()

    def delete_session(self, token: str) -> None:
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute("DELETE FROM sessions WHERE session_token = ?", (token,))
                conn.commit()
            finally:
                conn.close()

    def get_user_config(self, user_id: int) -> Optional[Config]:
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT config_json FROM user_configs WHERE user_id = ?",
                (user_id,)
            ).fetchone()
            if not row:
                return None
            return Config.from_dict(json.loads(row[0]))
        finally:
            conn.close()

    def save_user_config(self, user_id: int, config: Config) -> None:
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    """INSERT INTO user_configs (user_id, config_json, updated_at)
                       VALUES (?, ?, datetime('now'))
                       ON CONFLICT(user_id) DO UPDATE SET
                         config_json = excluded.config_json,
                         updated_at = datetime('now')""",
                    (user_id, json.dumps(config.to_dict()))
                )
                conn.commit()
            finally:
                conn.close()

    def get_paper_user_result(self, paper_id: str, user_id: int) -> Optional[Dict]:
        conn = self._get_conn()
        try:
            row = conn.execute(
                """SELECT is_relevant, relevance_score, extracted_keywords, one_line_summary,
                          detailed_summary, tags, qa_pairs_json, is_starred, is_hidden, star_category
                   FROM paper_user_results WHERE paper_id = ? AND user_id = ?""",
                (paper_id, user_id)
            ).fetchone()
            if not row:
                return None
            r = dict(row)
            if r.get("extracted_keywords"):
                r["extracted_keywords"] = json.loads(r["extracted_keywords"]) if isinstance(r["extracted_keywords"], str) else r["extracted_keywords"]
            if r.get("tags"):
                r["tags"] = json.loads(r["tags"]) if isinstance(r["tags"], str) else r["tags"]
            if r.get("qa_pairs_json"):
                r["qa_pairs"] = json.loads(r["qa_pairs_json"])
            return r
        finally:
            conn.close()

    def save_paper_user_result(
        self,
        paper_id: str,
        user_id: int,
        is_relevant: Optional[bool] = None,
        relevance_score: float = 0,
        extracted_keywords: List[str] = None,
        one_line_summary: str = "",
        detailed_summary: str = "",
        tags: List[str] = None,
        qa_pairs: List[dict] = None,
        is_starred: bool = False,
        is_hidden: bool = False,
        star_category: str = "Other",
    ) -> None:
        with self._lock:
            conn = self._get_conn()
            try:
                kw_json = json.dumps(extracted_keywords or [])
                tags_json = json.dumps(tags or [])
                qa_json = json.dumps(qa_pairs or [])
                conn.execute(
                    """INSERT INTO paper_user_results
                       (paper_id, user_id, is_relevant, relevance_score, extracted_keywords,
                        one_line_summary, detailed_summary, tags, qa_pairs_json,
                        is_starred, is_hidden, star_category, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                       ON CONFLICT(paper_id, user_id) DO UPDATE SET
                         is_relevant = COALESCE(excluded.is_relevant, is_relevant),
                         relevance_score = COALESCE(excluded.relevance_score, relevance_score),
                         extracted_keywords = COALESCE(excluded.extracted_keywords, extracted_keywords),
                         one_line_summary = COALESCE(NULLIF(excluded.one_line_summary,''), one_line_summary),
                         detailed_summary = COALESCE(NULLIF(excluded.detailed_summary,''), detailed_summary),
                         tags = COALESCE(excluded.tags, tags),
                         qa_pairs_json = COALESCE(NULLIF(excluded.qa_pairs_json,'[]'), qa_pairs_json),
                         is_starred = COALESCE(excluded.is_starred, is_starred),
                         is_hidden = COALESCE(excluded.is_hidden, is_hidden),
                         star_category = COALESCE(NULLIF(excluded.star_category,''), star_category),
                         updated_at = datetime('now')""",
                    (
                        paper_id, user_id,
                        1 if is_relevant else (0 if is_relevant is False else None),
                        relevance_score, kw_json, one_line_summary or "", detailed_summary or "",
                        tags_json, qa_json,
                        1 if is_starred else 0, 1 if is_hidden else 0, star_category or "Other"
                    )
                )
                conn.commit()
            finally:
                conn.close()

    def get_user_paper_overlays(self, user_id: int) -> Dict[str, dict]:
        """Get all paper overlays for user. Returns {paper_id: {is_starred, is_hidden, ...}}."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """SELECT paper_id, is_relevant, relevance_score, extracted_keywords, one_line_summary,
                          detailed_summary, tags, is_starred, is_hidden, star_category
                   FROM paper_user_results WHERE user_id = ?""",
                (user_id,)
            ).fetchall()
            result = {}
            for r in rows:
                pid = r[0]
                result[pid] = {
                    "is_relevant": bool(r[1]) if r[1] is not None else None,
                    "relevance_score": r[2],
                    "extracted_keywords": json.loads(r[3]) if r[3] else [],
                    "one_line_summary": r[4] or "",
                    "detailed_summary": r[5] or "",
                    "tags": json.loads(r[6]) if r[6] else [],
                    "is_starred": bool(r[7]) if r[7] is not None else False,
                    "is_hidden": bool(r[8]) if r[8] is not None else False,
                    "star_category": r[9] or "Other",
                }
            return result
        finally:
            conn.close()

    def get_config_hash(self, config: Config) -> str:
        """Hash for KV cache batching - same config => same hash."""
        code = json.dumps({
            "filter_keywords": sorted(config.filter_keywords or []),
            "negative_keywords": sorted(config.negative_keywords or []),
            "system_prompt": config.system_prompt or "",
        }, sort_keys=True)
        return _config_hash(code)


_serving_db: Optional[ServingDB] = None


def get_serving_db() -> ServingDB:
    global _serving_db
    if _serving_db is None:
        _serving_db = ServingDB()
    return _serving_db
