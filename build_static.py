#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Static assets build script - Simple cache busting with content hash

Linus philosophy: "Do one thing and do it well"
"""

import os
import hashlib
import shutil
import re
from pathlib import Path


def calculate_file_hash(filepath: str) -> str:
    """Calculate MD5 hash of file content - simple and reliable"""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        hasher.update(f.read())
    return hasher.hexdigest()[:8]  # First 8 chars is enough


def build_static_assets(source_dir: str = "frontend", dest_dir: str = "frontend_dist"):
    """
    Build static assets with cache busting
    
    Simple flow:
    1. Calculate hash for all JS/CSS files
    2. Copy entire frontend/ to frontend_dist/
    3. Replace all references in HTML files with versioned URLs
    
    No special cases, no complex logic.
    """
    print(f"[Build] Starting static assets build...")
    
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)
    
    if not source_path.exists():
        print(f"[Build] Error: Source directory {source_dir} not found")
        return False
    
    # Step 1: Calculate hashes for all JS/CSS files
    print(f"[Build] Calculating file hashes...")
    file_hashes = {}
    
    for pattern in ["**/*.js", "**/*.css"]:
        for filepath in source_path.glob(pattern):
            if filepath.is_file():
                relative_path = filepath.relative_to(source_path)
                file_hash = calculate_file_hash(filepath)
                file_hashes[str(relative_path)] = file_hash
                print(f"[Build]   {relative_path} -> {file_hash}")
    
    if not file_hashes:
        print(f"[Build] Warning: No JS/CSS files found in {source_dir}")
    
    # Step 2: Clean and copy frontend directory
    print(f"[Build] Copying {source_dir}/ to {dest_dir}/...")
    if dest_path.exists():
        shutil.rmtree(dest_path)
    shutil.copytree(source_path, dest_path)
    print(f"[Build] Copy completed")
    
    # Step 3: Update HTML files with versioned URLs
    print(f"[Build] Updating HTML references...")
    html_files = list(dest_path.glob("**/*.html"))
    
    for html_file in html_files:
        print(f"[Build]   Processing {html_file.name}...")
        
        with open(html_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Replace CSS references: href="/static/style.css" -> href="/static/style.css?v=hash"
        # Pattern: matches href="..." or src="..." with .js or .css files
        def replace_reference(match):
            attr = match.group(1)  # href or src
            quote = match.group(2)  # " or '
            path = match.group(3)   # the file path
            
            # Extract the static resource path
            # Pattern: /static/xxx/file.js or /static/file.css
            static_match = re.search(r'/static/(.+\.(?:js|css))', path)
            if static_match:
                resource_path = static_match.group(1)  # e.g., "app.js" or "style.css"
                
                if resource_path in file_hashes:
                    file_hash = file_hashes[resource_path]
                    # Remove existing version parameter if present
                    path_clean = re.sub(r'\?v=[a-f0-9]+', '', path)
                    path_versioned = f"{path_clean}?v={file_hash}"
                    return f'{attr}={quote}{path_versioned}{quote}'
            
            return match.group(0)  # No change
        
        # Match href="..." or src="..." for .js and .css files
        content = re.sub(
            r'(href|src)=(["\'"])([^"\']*\.(?:js|css)[^"\']*)\2',
            replace_reference,
            content
        )
        
        if content != original_content:
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"[Build]     Updated {html_file.name}")
        else:
            print(f"[Build]     No changes in {html_file.name}")
    
    print(f"[Build] Build completed successfully!")
    print(f"[Build] Processed {len(file_hashes)} assets, {len(html_files)} HTML files")
    return True


if __name__ == "__main__":
    success = build_static_assets()
    exit(0 if success else 1)

