"""
Resolve Config for request: from user DB in serving mode, else default.
"""

from typing import Optional
from models import Config
from default_config import DEFAULT_CONFIG
from pathlib import Path

from .db import get_serving_db


def get_config_for_user(user_id: Optional[int], config_path: Path) -> Config:
    """
    In serving mode with user_id: load from user_configs.
    Else: load from config_path (single-user mode).
    """
    if user_id is not None:
        db = get_serving_db()
        cfg = db.get_user_config(user_id)
        if cfg is not None:
            return cfg
    return Config.load(str(config_path))


def get_config_path() -> Path:
    return Path("data/config.json")
