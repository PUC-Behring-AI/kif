# Copyright (C) 2026 PUC-Rio/PUC-Behring Institute for AI
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from kif_lib import Item, Lexeme, Property, Search
from kif_lib.search import MixerSearch
from kif_lib.typing import Any, AsyncIterator, Iterable, Iterator, override
from kif_lib.vocabulary import wd

from ...tests import TestCase


class FakeSearch(
        Search,
        search_name='test-mixer-fake',
        search_description='Fake search used by mixer tests'
):
    """Search that yields fixed ids and records the options it sees."""

    def __init__(
            self,
            search_name: str,
            items: Iterable[int] = (),
            lexemes: Iterable[int] = (),
            properties: Iterable[int] = (),
            **kwargs: Any
    ) -> None:
        self._ids = {
            'item': list(items),
            'lexeme': list(lexemes),
            'property': list(properties),
        }
        self.calls: list[dict[str, Any]] = []
        super().__init__(search_name, **kwargs)

    def _data(self, type: str, search: str, options: Any) -> list[dict]:
        self.calls.append({
            'type': type,
            'search': search,
            'language': options.language,
            'limit': options.limit,
            'timeout': options.timeout,
        })
        ids = self._ids[type]
        if options.limit is not None:
            ids = ids[:options.limit]
        return [{'id': i, 'source': id(self)} for i in ids]

    @override
    def _to_item(self, data: Search.TData) -> Item:
        return wd.Q(data['id'])

    @override
    def _to_lexeme(self, data: Search.TData) -> Lexeme:
        return wd.L(data['id'])

    @override
    def _to_property(self, data: Search.TData) -> Property:
        return wd.P(data['id'])

    @override
    def _item_data(self, search: str, options: Any) -> Iterator[dict]:
        return iter(self._data('item', search, options))

    @override
    def _lexeme_data(self, search: str, options: Any) -> Iterator[dict]:
        return iter(self._data('lexeme', search, options))

    @override
    def _property_data(self, search: str, options: Any) -> Iterator[dict]:
        return iter(self._data('property', search, options))

    async def _agen(self, data: list[dict]) -> AsyncIterator[dict]:
        for x in data:
            yield x

    @override
    def _aitem_data(self, search: str, options: Any) -> AsyncIterator[dict]:
        return self._agen(self._data('item', search, options))

    @override
    def _alexeme_data(
            self,
            search: str,
            options: Any
    ) -> AsyncIterator[dict]:
        return self._agen(self._data('lexeme', search, options))

    @override
    def _aproperty_data(
            self,
            search: str,
            options: Any
    ) -> AsyncIterator[dict]:
        return self._agen(self._data('property', search, options))


def fake(**kwargs: Any) -> FakeSearch:
    s = Search('test-mixer-fake', **kwargs)
    assert isinstance(s, FakeSearch)
    return s


class Test(TestCase):

    def test_registry(self) -> None:
        self.assertIs(Search.registry['mixer'], MixerSearch)
        self.assertIsInstance(Search('mixer'), MixerSearch)

    def test_init_sources_by_name(self) -> None:
        mx = Search('mixer', 'wikidata', 'wikidata-vdb')
        assert isinstance(mx, MixerSearch)
        self.assertEqual(
            [s.search_name for s in mx.sources],
            ['wikidata-wapi', 'wikidata-vdb'])

    def test_init_sources_mixed(self) -> None:
        a, b, c = fake(), fake(), fake()
        mx = Search('mixer', a, [b, 'empty'], (c,))
        assert isinstance(mx, MixerSearch)
        self.assertEqual(len(mx.sources), 4)
        self.assertIs(mx.sources[0], a)
        self.assertIs(mx.sources[1], b)
        self.assertEqual(mx.sources[2].search_name, 'empty')
        self.assertIs(mx.sources[3], c)
        self.assertTrue(mx.distinct)
        self.assertEqual(mx.method, 'roundrobin')

    def test_init_errors(self) -> None:
        self.assertRaises(TypeError, Search, 'mixer', 0)
        self.assertRaises(TypeError, Search, 'mixer', [fake(), 0])
        self.assertRaises(ValueError, Search, 'mixer', 'no-such-search')
        self.assertRaises(ValueError, Search, 'mixer', 'mixer')
        self.assertRaises(ValueError, Search, 'mixer', method='bad')

    def test_no_sources(self) -> None:
        mx = Search('mixer')
        self.assertEqual(list(mx.item('x')), [])
        self.assertEqual(list(mx.property('x')), [])
        self.assertEqual(list(mx.lexeme('x')), [])

    def test_item_roundrobin_distinct(self) -> None:
        mx = Search('mixer', fake(items=[1, 2, 3]), fake(items=[2, 4]))
        self.assertEqual(
            list(mx.item('x')), [wd.Q(1), wd.Q(2), wd.Q(4), wd.Q(3)])

    def test_item_not_distinct(self) -> None:
        mx = Search(
            'mixer', fake(items=[1, 2, 3]), fake(items=[2, 4]),
            distinct=False)
        self.assertEqual(
            list(mx.item('x')),
            [wd.Q(1), wd.Q(2), wd.Q(2), wd.Q(4), wd.Q(3)])

    def test_item_chain(self) -> None:
        mx = Search(
            'mixer', fake(items=[1, 2, 3]), fake(items=[2, 4]),
            method='chain')
        self.assertEqual(
            list(mx.item('x')), [wd.Q(1), wd.Q(2), wd.Q(3), wd.Q(4)])

    def test_item_limit(self) -> None:
        mx = Search('mixer', fake(items=[1, 2, 3]), fake(items=[4, 5, 6]))
        self.assertEqual(list(mx.item('x', limit=3)), [
            wd.Q(1), wd.Q(4), wd.Q(2)])
        self.assertEqual(list(mx.item('x', limit=0)), [])
        mx = Search(
            'mixer', fake(items=[1, 2, 3]), fake(items=[4, 5, 6]), limit=2)
        self.assertEqual(list(mx.item('x')), [wd.Q(1), wd.Q(4)])

    def test_item_descriptor(self) -> None:
        mx = Search('mixer', fake(items=[1, 2]), fake(items=[2, 3]))
        self.assertEqual(
            list(mx.item_descriptor('x')),
            [(wd.Q(1), {}), (wd.Q(2), {}), (wd.Q(3), {})])

    def test_item_data(self) -> None:
        a, b = fake(items=[1, 2]), fake(items=[2])
        mx = Search('mixer', a, b)
        self.assertEqual(list(mx.item_data('x')), [
            {'id': 1, 'source': id(a)},
            {'id': 2, 'source': id(b)},
            {'id': 2, 'source': id(a)}])

    def test_lexeme(self) -> None:
        mx = Search('mixer', fake(lexemes=[1]), fake(lexemes=[1, 2]))
        self.assertEqual(list(mx.lexeme('x')), [wd.L(1), wd.L(2)])
        self.assertEqual(
            list(mx.lexeme_descriptor('x')), [(wd.L(1), {}), (wd.L(2), {})])
        self.assertEqual(len(list(mx.lexeme_data('x'))), 3)

    def test_property(self) -> None:
        mx = Search('mixer', fake(properties=[1, 2]), fake(properties=[3]))
        self.assertEqual(
            list(mx.property('x')), [wd.P(1), wd.P(3), wd.P(2)])
        self.assertEqual(
            list(mx.property_descriptor('x')),
            [(wd.P(1), {}), (wd.P(3), {}), (wd.P(2), {})])
        self.assertEqual(len(list(mx.property_data('x'))), 3)

    def test_options_propagation(self) -> None:
        a = fake(items=[1], limit=7, timeout=3.)
        b = fake(items=[2])
        mx = Search('mixer', a, b)
        list(mx.item('abc', language='pt', limit=5))
        self.assertEqual(a.calls[-1]['search'], 'abc')
        self.assertEqual(a.calls[-1]['language'], 'pt')
        self.assertEqual(b.calls[-1]['language'], 'pt')
        self.assertEqual(a.calls[-1]['limit'], 5)
        self.assertEqual(b.calls[-1]['limit'], 5)
        # Options not set in mixer: sources keep their own.
        self.assertEqual(a.calls[-1]['timeout'], 3.)
        list(mx.item('abc'))
        self.assertEqual(a.calls[-1]['limit'], 7)
        # Options set in mixer instance are propagated.
        mx = Search('mixer', a, b, timeout=9.)
        list(mx.item('abc'))
        self.assertEqual(a.calls[-1]['timeout'], 9.)
        self.assertEqual(b.calls[-1]['timeout'], 9.)
        # Sources options are left untouched.
        self.assertEqual(a.limit, 7)
        self.assertEqual(a.timeout, 3.)

    async def test_aitem(self) -> None:
        mx = Search('mixer', fake(items=[1, 2, 3]), fake(items=[2, 4]))
        self.assertEqual(
            [x async for x in mx.aitem('x')],
            [wd.Q(1), wd.Q(2), wd.Q(4), wd.Q(3)])
        self.assertEqual(
            [x async for x in mx.aitem('x', limit=2)], [wd.Q(1), wd.Q(2)])
        self.assertEqual(
            [x async for x in mx.aitem_descriptor('x')],
            [(wd.Q(1), {}), (wd.Q(2), {}), (wd.Q(4), {}), (wd.Q(3), {})])
        self.assertEqual(len([x async for x in mx.aitem_data('x')]), 5)

    async def test_alexeme(self) -> None:
        mx = Search('mixer', fake(lexemes=[1]), fake(lexemes=[1, 2]))
        self.assertEqual(
            [x async for x in mx.alexeme('x')], [wd.L(1), wd.L(2)])
        self.assertEqual(
            [x async for x in mx.alexeme_descriptor('x')],
            [(wd.L(1), {}), (wd.L(2), {})])
        self.assertEqual(len([x async for x in mx.alexeme_data('x')]), 3)

    async def test_aproperty(self) -> None:
        mx = Search('mixer', fake(properties=[1, 2]), fake(properties=[3]))
        self.assertEqual(
            [x async for x in mx.aproperty('x')],
            [wd.P(1), wd.P(3), wd.P(2)])
        self.assertEqual(
            [x async for x in mx.aproperty_descriptor('x')],
            [(wd.P(1), {}), (wd.P(3), {}), (wd.P(2), {})])
        self.assertEqual(
            len([x async for x in mx.aproperty_data('x')]), 3)


if __name__ == '__main__':
    Test.main()
