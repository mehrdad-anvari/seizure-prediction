from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, TypeVar, Union, get_args, get_origin

from seizure_pred.core.config import TrainConfig

T = TypeVar("T")


class ConfigValidationError(ValueError):
    pass


def _is_optional(tp) -> bool:
    return get_origin(tp) is Union and type(None) in get_args(tp)


def _optional_inner(tp):
    return [a for a in get_args(tp) if a is not type(None)][0]


def _type_name(tp) -> str:
    try:
        return tp.__name__
    except Exception:
        return str(tp)


def _check_value(path: str, value: Any, expected_type) -> List[str]:
    errs: List[str] = []
    origin = get_origin(expected_type)

    if _is_optional(expected_type):
        inner = _optional_inner(expected_type)
        if value is None:
            return errs
        return _check_value(path, value, inner)

    if origin in (list, List):
        if not isinstance(value, list):
            return [f"{path}: expected list, got {type(value).__name__}"]
        (inner,) = get_args(expected_type) or (Any,)
        for i, v in enumerate(value):
            errs += _check_value(f"{path}[{i}]", v, inner)
        return errs

    if origin in (dict, Dict):
        if not isinstance(value, dict):
            return [f"{path}: expected dict, got {type(value).__name__}"]
        k_t, v_t = get_args(expected_type) or (Any, Any)
        for k, v in value.items():
            errs += _check_value(f"{path}.<key>", k, k_t)
            errs += _check_value(f"{path}[{repr(k)}]", v, v_t)
        return errs

    # dataclass
    if isinstance(expected_type, type) and is_dataclass(expected_type):
        if not isinstance(value, dict):
            return [f"{path}: expected mapping for {_type_name(expected_type)}, got {type(value).__name__}"]
        return validate_dict(value, expected_type, path=path)

    # basic types
    if expected_type is Any:
        return errs

    if isinstance(expected_type, type):
        if not isinstance(value, expected_type):
            # allow ints for floats
            if expected_type is float and isinstance(value, int):
                return errs
            return [f"{path}: expected {_type_name(expected_type)}, got {type(value).__name__}"]
        return errs

    return errs


def validate_dict(d: Dict[str, Any], dataclass_type: Type[T], *, path: str = "cfg") -> List[str]:
    """Validate a nested dict against a dataclass type.

    Returns a list of error strings (empty means valid).
    """
    if not is_dataclass(dataclass_type):
        raise TypeError("dataclass_type must be a dataclass type")

    errs: List[str] = []
    fields = dataclass_type.__dataclass_fields__  # type: ignore[attr-defined]

    # unknown keys
    for k in d.keys():
        if k not in fields:
            errs.append(f"{path}: unknown key '{k}'")

    # required fields + type checks
    for name, f in fields.items():
        if name not in d:
            # allow missing if default exists
            if f.default is not f.default_factory:  # type: ignore
                # default exists or default_factory exists
                continue
            # if both default and default_factory missing, treat as required
            # dataclasses always have something, but keep robust:
            continue

        errs += _check_value(f"{path}.{name}", d[name], f.type)

    return errs


def validate_train_config_dict(d: Dict[str, Any]) -> None:
    errs = validate_dict(d, TrainConfig, path="train")
    if errs:
        msg = "Invalid training config:\n" + "\n".join("- " + e for e in errs)
        raise ConfigValidationError(msg)


def validate_config_dict(d: Dict[str, Any], dataclass_type: Type[T] = TrainConfig) -> None:
    """Generic validator used by CLI.

    Defaults to TrainConfig for backwards compatibility.
    """
    errs = validate_dict(d, dataclass_type, path="cfg")
    if errs:
        msg = f"Invalid config for {dataclass_type.__name__}:\n" + "\n".join("- " + e for e in errs)
        raise ConfigValidationError(msg)
