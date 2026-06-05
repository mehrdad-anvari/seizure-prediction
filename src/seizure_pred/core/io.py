from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Type, TypeVar, get_args, get_origin, get_type_hints

T = TypeVar("T")

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


def _to_dict(obj: Any) -> Dict[str, Any]:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return obj
    raise TypeError(f"Cannot serialize type: {type(obj)!r}")


def save_config(cfg: Any, path: str | Path) -> None:
    path = Path(path)
    data = _to_dict(cfg)
    if path.suffix.lower() in {".yml", ".yaml"}:
        if yaml is None:
            raise RuntimeError("PyYAML not installed. Install with: pip install seizure-pred[cli]")
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        return
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return
    raise ValueError(f"Unsupported config extension: {path.suffix}")


def load_dict(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".yml", ".yaml"}:
        if yaml is None:
            raise RuntimeError("PyYAML not installed. Install with: pip install seizure-pred[cli]")
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    raise ValueError(f"Unsupported config extension: {path.suffix}")


def merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge override into base (dicts only)."""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge_dict(out[k], v)
        else:
            out[k] = v
    return out


def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
    """Instantiate (possibly nested) dataclasses from a dict.

    Uses `typing.get_type_hints` so it works with `from __future__ import annotations`.
    """
    if not hasattr(cls, "__dataclass_fields__"):
        raise TypeError("from_dict expects a dataclass class")

    type_hints = get_type_hints(cls)
    kwargs: Dict[str, Any] = {}
    for name, field in cls.__dataclass_fields__.items():  # type: ignore[attr-defined]
        if name not in data:
            continue
        val = data[name]

        # Prefer resolved type hints (handles postponed annotations)
        ftype = type_hints.get(name, field.type)

        # Handle Optional[T] (Union[T, None])
        origin = get_origin(ftype)
        if origin is not None:
            args = get_args(ftype)
            # Optional[T]
            if origin is list or origin is dict:
                kwargs[name] = val
                continue
            if origin is tuple:
                kwargs[name] = val
                continue
            if origin is type(None):
                kwargs[name] = val
                continue
            if origin is getattr(__import__("typing"), "Union"):
                non_none = [a for a in args if a is not type(None)]
                if len(non_none) == 1:
                    ftype = non_none[0]

        # Nested dataclass
        if hasattr(ftype, "__dataclass_fields__") and isinstance(val, dict):
            kwargs[name] = from_dict(ftype, val)  # type: ignore[arg-type]
        else:
            kwargs[name] = val
    return cls(**kwargs)  # type: ignore[misc]
