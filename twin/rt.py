"""Small RawTree client shared by every script: SQL in, rows or DataFrames out.

    from twin.rt import q, rows, insert, TEAM
"""
from __future__ import annotations

import io
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"

_env = ROOT / ".env"
for _line in _env.read_text().splitlines() if _env.exists() else []:
    _key, _, _value = _line.partition("=")
    if _key.strip() and not _key.lstrip().startswith("#"):
        os.environ.setdefault(_key.strip(), _value.strip())

API = os.environ.get("RAWTREE_API_URL", "https://api.rawtree.com")
TEAM = os.environ["TEAM_ID"]

http = requests.Session()
http.headers["Authorization"] = f"Bearer {os.environ['RAWTREE_API_KEY']}"


def _post(payload: dict, timeout: float, retries: int) -> requests.Response:
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            r = http.post(f"{API}/v1/query", json=payload, timeout=timeout)
            if r.status_code >= 400:
                raise RuntimeError(f"RawTree {r.status_code}: {r.text[:400]}")
            return r
        except Exception as e:  # network blips are common on venue Wi-Fi
            last = e
            if attempt < retries:
                time.sleep(0.5 * (attempt + 1))
    raise last  # type: ignore[misc]


def rows(sql: str, timeout: float = 60, retries: int = 1) -> list[dict[str, Any]]:
    """Run SQL and return a list of JSON rows (keeps arrays and nested objects)."""
    return _post({"sql": sql}, timeout, retries).json()["data"]


def q(sql: str, timeout: float = 60, retries: int = 1) -> pd.DataFrame:
    """Run SQL and return a DataFrame (JSON transport: fine up to ~100k rows)."""
    return pd.DataFrame(rows(sql, timeout, retries))


def q_csv(sql: str, timeout: float = 120, retries: int = 1, **read_csv: Any) -> pd.DataFrame:
    """Run SQL and return a DataFrame through CSV transport: use for big flat results."""
    r = _post({"sql": sql, "format": "CSVWithNames"}, timeout, retries)
    return pd.read_csv(io.BytesIO(r.content), **read_csv)


def insert(table: str, records: list[dict[str, Any]], timeout: float = 10) -> None:
    """Batch insert JSON records into a table of our team database."""
    r = http.post(f"{API}/v1/tables/{table}", params={"database": TEAM}, json=records, timeout=timeout)
    r.raise_for_status()
