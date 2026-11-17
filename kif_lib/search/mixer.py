# Copyright (C) 2026 PUC-Rio/PUC-Behring Institute for AI
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from .. import itertools
from ..context import Context
from ..model import Item, KIF_Object, Lexeme, Property
from ..typing import (
    Any,
    AsyncIterator,
    Callable,
    ClassVar,
    Final,
    Hashable,
    Iterable,
    Iterator,
    Literal,
    override,
    Sequence,
    TypeAlias,
    TypeVar,
    Union,
)
from .abc import Search, SearchOptions

T = TypeVar('T')
TOptions = TypeVar('TOptions', bound=SearchOptions, default=SearchOptions)

#: Type alias for mixing methods.
TMixerSearchMethod: TypeAlias = Literal['chain', 'roundrobin']

#: Type alias for things coercible to a mixer source.
TMixerSearchSource: TypeAlias = Union[Search, str]


class MixerSearch(
        Search[TOptions],
        search_name='mixer',
        search_description='Mixer search'
):
    """Mixer search.

    Runs the same search over several underlying searches (sources) and
    returns the union of their results.

    Example::

       Search('mixer', 'wikidata', 'wikidata-vdb')
       Search('mixer', [Search('wikidata'), Search('wikidata-vdb')])

    Parameters:
       search_name: Name of the search plugin to instantiate.
       sources: Sources to mix.
       distinct: Whether to skip duplicated entities.
       method: Mixing method (``'roundrobin'`` or ``'chain'``).
    """

    #: The search options propagated to sources (when set explicitly).
    _synced_options: ClassVar[Sequence[str]] = (
        'debug', 'language', 'limit', 'lookahead', 'page_size', 'timeout')

    #: Default value for :attr:`MixerSearch.distinct`.
    DEFAULT_DISTINCT: Final[bool] = True

    #: Default value for :attr:`MixerSearch.method`.
    DEFAULT_METHOD: Final[TMixerSearchMethod] = 'roundrobin'

    __slots__ = (
        '_sources',
        '_distinct',
        '_method',
    )

    _sources: Sequence[Search]
    _distinct: bool
    _method: TMixerSearchMethod

    def __init__(
            self,
            search_name: str,
            *sources: TMixerSearchSource | Iterable[TMixerSearchSource],
            distinct: bool | None = None,
            method: TMixerSearchMethod | None = None,
            context: Context | None = None,
            **kwargs: Any
    ) -> None:
        self._init_sources(sources, context)
        self._distinct = self.DEFAULT_DISTINCT if distinct is None else bool(
            distinct)
        self._method = KIF_Object._check_arg(
            method or self.DEFAULT_METHOD,
            (method or self.DEFAULT_METHOD) in ('chain', 'roundrobin'),
            f"bad mixing method '{method}'",
            type(self), 'method', None, ValueError)
        super().__init__(search_name, context=context, **kwargs)

    def _init_sources(
            self,
            sources: Iterable[TMixerSearchSource | Iterable[
                TMixerSearchSource]],
            context: Context | None
    ) -> None:
        def coerce(src: Any) -> Search:
            if isinstance(src, Search):
                return src
            if isinstance(src, str):
                if src == self.search_name:
                    raise KIF_Object._arg_error(
                        'cannot mix a mixer by name',
                        type(self), 'sources', 2, ValueError)
                if context is None:
                    return Search(src)
                return Search(src, context=context)
            raise KIF_Object._arg_error(
                'expected Search or str', type(self), 'sources', 2,
                TypeError)
        result: list[Search] = []
        for src in sources:
            if isinstance(src, (Search, str)):
                result.append(coerce(src))
            elif isinstance(src, Iterable):
                result.extend(map(coerce, src))
            else:
                coerce(src)     # raises TypeError
        self._sources = result

    @property
    def sources(self) -> Sequence[Search]:
        """The mixed sources."""
        return self.get_sources()

    def get_sources(self) -> Sequence[Search]:
        """Gets the mixed sources.

        Returns:
           Mixed sources.
        """
        return self._sources

    @property
    def distinct(self) -> bool:
        """Whether to skip duplicated entities."""
        return self._distinct

    @property
    def method(self) -> TMixerSearchMethod:
        """The mixing method."""
        return self._method

# -- Options propagation ---------------------------------------------------

    def _get_source_kwargs(self, options: TOptions) -> dict[str, Any]:
        """Gets the options explicitly set in mixer (instance or call).

        Options left unset are not propagated so that each source keeps
        using its own defaults.
        """
        default = self.get_default_options()
        kwargs: dict[str, Any] = {}
        for name in self._synced_options:
            opts: Any = options
            seen: set[int] = set()
            while opts is not None and opts is not default \
                    and id(opts) not in seen:
                seen.add(id(opts))
                value = getattr(opts, '_' + name, None)
                if value is not None:
                    kwargs[name] = value
                    break
                try:
                    opts = opts.parent
                except Exception:
                    break
        return kwargs

# -- Mixing ----------------------------------------------------------------

    def _mix(
            self,
            get_x_fn: Callable[[Search], Callable[..., Iterator[T]]],
            search: str,
            options: TOptions,
            key: Callable[[T], Hashable] | None
    ) -> Iterator[T]:
        kwargs = self._get_source_kwargs(options)
        return itertools.mix(
            *(get_x_fn(src)(search, **kwargs) for src in self._sources),
            distinct=self._distinct and key is not None,
            distinct_key=key,  # type: ignore
            method=self._method)

    def _amix(
            self,
            get_ax_fn: Callable[[Search], Callable[..., AsyncIterator[T]]],
            search: str,
            options: TOptions,
            key: Callable[[T], Hashable] | None
    ) -> AsyncIterator[T]:
        kwargs = self._get_source_kwargs(options)
        return itertools.amix(
            *(get_ax_fn(src)(search, **kwargs) for src in self._sources),
            distinct=self._distinct and key is not None,
            distinct_key=key,  # type: ignore
            method=self._method)

    @staticmethod
    def _key_id(x: Any) -> Hashable:
        return x

    @staticmethod
    def _key_fst(x: tuple[Any, Any]) -> Hashable:
        return x[0]

# -- Item search -----------------------------------------------------------

    @override
    def _item(self, search: str, options: TOptions) -> Iterator[Item]:
        return self._mix(
            lambda s: s.item, search, options, self._key_id)

    @override
    def _item_descriptor(
            self,
            search: str,
            options: TOptions
    ) -> Iterator[tuple[Item, Item.Descriptor]]:
        return self._mix(
            lambda s: s.item_descriptor, search, options, self._key_fst)

    @override
    def _item_data(
            self,
            search: str,
            options: TOptions
    ) -> Iterator[Search.TData]:
        return self._mix(lambda s: s.item_data, search, options, None)

    @override
    def _aitem(self, search: str, options: TOptions) -> AsyncIterator[Item]:
        return self._amix(
            lambda s: s.aitem, search, options, self._key_id)

    @override
    def _aitem_descriptor(
            self,
            search: str,
            options: TOptions
    ) -> AsyncIterator[tuple[Item, Item.Descriptor]]:
        return self._amix(
            lambda s: s.aitem_descriptor, search, options, self._key_fst)

    @override
    def _aitem_data(
            self,
            search: str,
            options: TOptions
    ) -> AsyncIterator[Search.TData]:
        return self._amix(lambda s: s.aitem_data, search, options, None)

# -- Lexeme search ---------------------------------------------------------

    @override
    def _lexeme(self, search: str, options: TOptions) -> Iterator[Lexeme]:
        return self._mix(
            lambda s: s.lexeme, search, options, self._key_id)

    @override
    def _lexeme_descriptor(
            self,
            search: str,
            options: TOptions
    ) -> Iterator[tuple[Lexeme, Lexeme.Descriptor]]:
        return self._mix(
            lambda s: s.lexeme_descriptor, search, options, self._key_fst)

    @override
    def _lexeme_data(
            self,
            search: str,
            options: TOptions
    ) -> Iterator[Search.TData]:
        return self._mix(lambda s: s.lexeme_data, search, options, None)

    @override
    def _alexeme(
            self,
            search: str,
            options: TOptions
    ) -> AsyncIterator[Lexeme]:
        return self._amix(
            lambda s: s.alexeme, search, options, self._key_id)

    @override
    def _alexeme_descriptor(
            self,
            search: str,
            options: TOptions
    ) -> AsyncIterator[tuple[Lexeme, Lexeme.Descriptor]]:
        return self._amix(
            lambda s: s.alexeme_descriptor, search, options, self._key_fst)

    @override
    def _alexeme_data(
            self,
            search: str,
            options: TOptions
    ) -> AsyncIterator[Search.TData]:
        return self._amix(lambda s: s.alexeme_data, search, options, None)

# -- Property search -------------------------------------------------------

    @override
    def _property(
            self,
            search: str,
            options: TOptions
    ) -> Iterator[Property]:
        return self._mix(
            lambda s: s.property, search, options, self._key_id)

    @override
    def _property_descriptor(
            self,
            search: str,
            options: TOptions
    ) -> Iterator[tuple[Property, Property.Descriptor]]:
        return self._mix(
            lambda s: s.property_descriptor, search, options, self._key_fst)

    @override
    def _property_data(
            self,
            search: str,
            options: TOptions
    ) -> Iterator[Search.TData]:
        return self._mix(lambda s: s.property_data, search, options, None)

    @override
    def _aproperty(
            self,
            search: str,
            options: TOptions
    ) -> AsyncIterator[Property]:
        return self._amix(
            lambda s: s.aproperty, search, options, self._key_id)

    @override
    def _aproperty_descriptor(
            self,
            search: str,
            options: TOptions
    ) -> AsyncIterator[tuple[Property, Property.Descriptor]]:
        return self._amix(
            lambda s: s.aproperty_descriptor, search, options,
            self._key_fst)

    @override
    def _aproperty_data(
            self,
            search: str,
            options: TOptions
    ) -> AsyncIterator[Search.TData]:
        return self._amix(lambda s: s.aproperty_data, search, options, None)
