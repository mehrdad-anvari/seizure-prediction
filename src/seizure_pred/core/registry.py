from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Generic, Iterable, Iterator, Optional, TypeVar

T = TypeVar("T")


class RegistryError(RuntimeError):
    """Raised for registry lookup/registration errors."""


@dataclass(frozen=True)
class Entry(Generic[T]):
    """A single registry entry."""

    name: str
    factory: Callable[..., T]
    help: str = ""


class Registry(Generic[T]):
    """Simple name -> factory registry.

    Intended usage:
        MODELS = Registry[nn.Module]("models")

        @MODELS.register("my_model")
        def build_my_model(cfg: ModelConfig):
            return MyModel(...)

        model = MODELS.create("my_model", cfg)
    """

    def __init__(self, kind: str):
        self.kind = kind
        self._entries: Dict[str, Entry[T]] = {}

    def register(self, name: str, *, help: str = "") -> Callable[[Callable[..., T]], Callable[..., T]]:
        if not name or not isinstance(name, str):
            raise RegistryError(f"Invalid {self.kind} name: {name!r}")

        def decorator(factory: Callable[..., T]) -> Callable[..., T]:
            if name in self._entries:
                raise RegistryError(f"Duplicate {self.kind} registration for '{name}'")
            self._entries[name] = Entry(name=name, factory=factory, help=help)
            return factory

        return decorator

    def get(self, name: str) -> Entry[T]:
        try:
            return self._entries[name]
        except KeyError as e:
            available = ", ".join(sorted(self._entries.keys())) or "<none>"
            raise RegistryError(
                f"Unknown {self.kind} '{name}'. Available: {available}"
            ) from e

    def create(self, name: str, *args, **kwargs) -> T:
        return self.get(name).factory(*args, **kwargs)

    def maybe_create(self, name: Optional[str], *args, **kwargs) -> Optional[T]:
        if name is None:
            return None
        return self.create(name, *args, **kwargs)

    def names(self) -> Iterable[str]:
        return self._entries.keys()

    def items(self) -> Iterable[Entry[T]]:
        return self._entries.values()

    def __contains__(self, name: str) -> bool:
        return name in self._entries

    def __iter__(self) -> Iterator[str]:
        return iter(self._entries)
