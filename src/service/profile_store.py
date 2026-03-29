from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_profile_db(db_path: str | Path) -> None:
    path = Path(db_path)
    with _connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS profiles (
                id TEXT PRIMARY KEY,
                display_name TEXT NOT NULL,
                budget TEXT DEFAULT '',
                preference TEXT DEFAULT '',
                need TEXT DEFAULT '',
                extra_json TEXT DEFAULT '{}',
                updated_at REAL NOT NULL
            )
            """
        )
        conn.commit()


def upsert_profile(
    db_path: str | Path,
    *,
    display_name: str,
    budget: str = "",
    preference: str = "",
    need: str = "",
    extra: Optional[Dict[str, Any]] = None,
    profile_id: Optional[str] = None,
) -> str:
    init_profile_db(db_path)
    pid = profile_id or str(uuid.uuid4())
    now = time.time()
    extra_json = json.dumps(extra or {}, ensure_ascii=False)
    with _connect(Path(db_path)) as conn:
        conn.execute(
            """
            INSERT INTO profiles (id, display_name, budget, preference, need, extra_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                display_name=excluded.display_name,
                budget=excluded.budget,
                preference=excluded.preference,
                need=excluded.need,
                extra_json=excluded.extra_json,
                updated_at=excluded.updated_at
            """,
            (pid, display_name, budget, preference, need, extra_json, now),
        )
        conn.commit()
    return pid


def get_profile(db_path: str | Path, profile_id: str) -> Optional[Dict[str, Any]]:
    if not Path(db_path).exists():
        return None
    with _connect(Path(db_path)) as conn:
        row = conn.execute(
            "SELECT id, display_name, budget, preference, need, extra_json FROM profiles WHERE id = ?",
            (profile_id,),
        ).fetchone()
    if not row:
        return None
    extra = {}
    try:
        extra = json.loads(row["extra_json"] or "{}")
    except json.JSONDecodeError:
        pass
    return {
        "id": row["id"],
        "display_name": row["display_name"],
        "budget": row["budget"] or "",
        "preference": row["preference"] or "",
        "need": row["need"] or "",
        "extra": extra,
    }


def list_profiles(db_path: str | Path) -> List[Dict[str, Any]]:
    if not Path(db_path).exists():
        return []
    with _connect(Path(db_path)) as conn:
        rows = conn.execute(
            "SELECT id, display_name, budget, preference, need, updated_at FROM profiles ORDER BY updated_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def delete_profile(db_path: str | Path, profile_id: str) -> bool:
    if not Path(db_path).exists():
        return False
    with _connect(Path(db_path)) as conn:
        cur = conn.execute("DELETE FROM profiles WHERE id = ?", (profile_id,))
        conn.commit()
        return cur.rowcount > 0


def merge_user_profile(
    stored: Optional[Dict[str, Any]],
    override: Optional[Dict[str, str]],
) -> Dict[str, str]:
    base: Dict[str, str] = {
        "budget": "",
        "preference": "",
        "need": "",
    }
    if stored:
        base["budget"] = str(stored.get("budget") or "")
        base["preference"] = str(stored.get("preference") or "")
        base["need"] = str(stored.get("need") or "")
    if not override:
        return base
    for k in ("budget", "preference", "need"):
        v = override.get(k)
        if v is not None and str(v).strip():
            base[k] = str(v).strip()
    return base
