from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def load_settings(path: str | Path = "configs/settings.yaml") -> Dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
