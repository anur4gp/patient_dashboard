"""
config.py

Loads ECW credentials from a local .env file (see .env.example).
Never commit a real .env — it's already covered by .gitignore below.
"""

from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv()


def _require(key: str) -> str:
    value = os.environ.get(key)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable: {key}. "
            f"Copy .env.example to .env and fill it in."
        )
    return value


ECW_BASE_URL = _require("ECW_BASE_URL")
ECW_CLIENT_ID = _require("ECW_CLIENT_ID")
ECW_TOKEN_URL = _require("ECW_TOKEN_URL")
ECW_PRIVATE_KEY_PATH = _require("ECW_PRIVATE_KEY_PATH")
ECW_KID = _require("ECW_KID")