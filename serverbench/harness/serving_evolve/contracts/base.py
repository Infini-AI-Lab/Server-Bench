"""The shape every contract shares: frozen, versioned, canonically hashable.

A *contract* is a typed value that crosses a boundary — between the controller
and an adapter, between one run and its resume, between an evaluation and the
evidence that justifies it. Three properties make that safe:

* **Frozen.** Contracts are immutable dataclasses. A caller that wants a variant
  builds a new one with :meth:`Contract.evolve`, which is visible in a diff; it
  cannot quietly mutate a value another component is still holding.
* **Versioned.** Every serialized form carries ``schema_version``. Reading a
  document written by a newer schema raises rather than guessing.
* **Canonically hashable.** ``digest()`` binds normalized semantic content, so
  the same plan hashes the same on any host on any day.

Core code passes contracts, never ``dict[str, Any]``. Dictionaries appear only at
the edges — CLI arguments, config files, JSON on disk — and are converted
immediately by ``from_dict``.
"""

from __future__ import annotations

import dataclasses
import enum
import functools
import types
from collections.abc import Mapping
from typing import (
    Any,
    ClassVar,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from ..errors import ContractError, SchemaVersionError
from ..hashing import digest

T = TypeVar("T", bound="Contract")

#: Every origin that means "a union". See _coerce.
_UNION_ORIGINS: tuple[object, ...] = (Union, types.UnionType)


def _plain(value: Any) -> Any:
    """Recursively reduce contracts, enums and containers to JSON-able values."""
    if isinstance(value, Contract):
        return value.to_dict()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _plain(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        seq = sorted(value, key=repr) if isinstance(value, set | frozenset) else list(value)
        return [_plain(v) for v in seq]
    return value


@functools.cache
def _hints(cls: type) -> dict[str, Any]:
    """Resolved field annotations for ``cls``.

    ``from __future__ import annotations`` turns every annotation into a string,
    so deserialization has to resolve them before it can tell a nested contract
    from a plain dict. Cached because resolution is not cheap and the answer
    never changes for a given class.
    """
    return dict(get_type_hints(cls))


def _coerce(hint: Any, value: Any) -> Any:
    """Rebuild nested contracts, enums and typed lists from plain JSON values.

    Without this, a round-trip through JSON would hand back dictionaries where
    the type says ``LatencyProfile``, and the mistake would only surface much
    later at an attribute access. Unrecognised shapes are passed through: this
    coerces what it understands and never guesses.
    """
    if hint is None or value is None:
        return value
    origin = get_origin(hint)
    # Both spellings of a union. `X | None` resolves to types.UnionType, while
    # `Optional[X]` resolves to typing.Union; checking only one would skip
    # coercion for every field written the other way.
    if origin in _UNION_ORIGINS:
        for arg in get_args(hint):
            if arg is not type(None):        # noqa: E721 - identity check is the point
                return _coerce(arg, value)
        return value
    if origin in (list, tuple) and isinstance(value, list):
        args = get_args(hint)
        return [_coerce(args[0], v) for v in value] if args else value
    if origin is dict:
        return value
    if isinstance(hint, type):
        if issubclass(hint, Contract) and isinstance(value, Mapping):
            return hint.from_dict(value)
        if issubclass(hint, enum.Enum):
            return hint(value)
    return value


@dataclasses.dataclass(frozen=True)
class Contract:
    """Base for every versioned contract.

    Subclasses set :attr:`SCHEMA_VERSION` and :attr:`KIND`, declare fields, and
    optionally override :meth:`validate` to enforce invariants that types cannot.
    """

    #: Bumped whenever the serialized shape changes incompatibly. A class
    #: attribute, deliberately not a field: it describes the type, not a value,
    #: and must never appear in ``__init__`` or in a subclass's field order.
    SCHEMA_VERSION: ClassVar[int] = 1
    #: Stable discriminator written into every serialized form.
    KIND: ClassVar[str] = "contract"

    def __post_init__(self) -> None:
        self.validate()

    # -- invariants ----------------------------------------------------------

    def validate(self) -> None:
        """Raise :class:`ContractError` if the value is internally inconsistent.

        The default accepts anything the type system already accepted.
        """

    def require(self, condition: bool, message: str, **detail: Any) -> None:
        if not condition:
            raise ContractError(f"{type(self).__name__}: {message}", **detail)

    # -- serialization -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:  # noqa: D102
        out: dict[str, Any] = {"schema_version": type(self).SCHEMA_VERSION,
                               "kind": type(self).KIND}
        for f in dataclasses.fields(self):
            if not f.init:
                continue
            out[f.name] = _plain(getattr(self, f.name))
        return out

    @classmethod
    def from_dict(cls: type[T], data: Mapping[str, Any]) -> T:
        found = int(data.get("schema_version", 1))
        if found != cls.SCHEMA_VERSION:
            raise SchemaVersionError(cls.KIND, found, cls.SCHEMA_VERSION)
        names = {f.name for f in dataclasses.fields(cls) if f.init}
        unknown = set(data) - names - {"schema_version", "kind"}
        if unknown:
            raise ContractError(f"{cls.__name__}: unknown field(s) {sorted(unknown)}; "
                                f"refusing to silently ignore them", unknown=sorted(unknown))
        hints = _hints(cls)
        kwargs = {k: _coerce(hints.get(k), v) for k, v in data.items() if k in names}
        return cls(**kwargs)  # type: ignore[arg-type]

    def evolve(self: T, **changes: Any) -> T:
        """Return a copy with fields replaced. The original is untouched."""
        return dataclasses.replace(self, **changes)

    # -- identity ------------------------------------------------------------

    def semantic_dict(self) -> dict[str, Any]:  # noqa: D401
        """``to_dict`` with every field that is at its default value omitted.

        This is what the digest hashes, and the omission is the whole point: a
        task's identity must depend on the task, not on the schema version of the
        harness that reads it. Without it, adding one optional field with a
        default would change the digest of every frozen task, and evidence
        recorded against the old digest could never be matched again.

        A field sitting at its default carries no information: omitting it says the
        same thing as recording it, and says it in a way that survives the schema
        gaining a neighbour.
        """
        out: dict[str, Any] = {"schema_version": type(self).SCHEMA_VERSION,
                               "kind": type(self).KIND}
        for f in dataclasses.fields(self):
            if not f.init:
                continue
            value = getattr(self, f.name)
            if f.default is not dataclasses.MISSING and value == f.default:
                continue
            if f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
                # Building the default can itself fail: a contract whose
                # validate() rejects its own all-default form (a ModelIdentity
                # with no model id, for instance). A default that cannot be
                # constructed is not a value this field could be
                # sitting at, so the field is meaningful and gets hashed.
                try:
                    if value == f.default_factory():  # type: ignore[misc]
                        continue
                except Exception:  # noqa: BLE001 - any failure means "not default"
                    pass
            out[f.name] = (value.semantic_dict() if isinstance(value, Contract)
                           else _plain_semantic(value))
        return out

    def digest(self) -> str:
        """SHA-256 over normalized semantic content (see :mod:`..hashing`)."""
        return digest(self.semantic_dict(), semantic=True)

    @property
    def short_digest(self) -> str:
        return self.digest()[:12]


def _plain_semantic(value: Any) -> Any:
    """Like ``_plain``, but nested contracts keep their default-omitting form."""
    if isinstance(value, Contract):
        return value.semantic_dict()
    if isinstance(value, Mapping):
        return {k: _plain_semantic(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_plain_semantic(v) for v in value]
    return _plain(value)
