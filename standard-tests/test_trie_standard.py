"""Standard tests for Trie data structure."""
from __future__ import unicode_literals

import pytest
import random
from itertools import chain
from collections import namedtuple
from importlib import import_module
from inspect import isgenerator

from cases import STR_EDGE_CASES

MODULENAME = 'trie'
CLASSNAME = 'Trie'
ROOT_ATTR = 'root'
END_CHAR = '$'
REMOVE_ERROR = ValueError

module = import_module(MODULENAME)
ClassDef = getattr(module, CLASSNAME)


REQ_METHODS = [
    'insert',
    'contains',
    'traversal',
    'remove',
    'size',
]


TrieFixture = namedtuple(
    'TrieFixture', (
        'instance',
        'sequence',
        'contains',
        'to_insert',
        'to_remove',
        'contain_false_shorter',
        'contain_false_longer',
        'start',
        'traverse',
    )
)


def _make_words():
    """Create lists of similar words from dictionary."""
    sample_size = 29
    words_between_samples = 2000

    sample_idx = random.randrange(words_between_samples)
    similar_words = []
    different_words = []

    with open('/usr/share/dict/words', 'r') as words:
        for idx, word in enumerate(words):
            word = word.strip()
            try:
                word = word.decode('utf-8')
            except AttributeError:
                pass
            if idx == sample_idx:
                different_words.append(word)
            if sample_idx <= idx <= sample_idx + sample_size:
                similar_words.append(word)
            elif idx > sample_idx + sample_size:
                yield similar_words
                sample_idx = idx + random.randrange(words_between_samples)
                similar_words = []
        yield similar_words
        yield different_words


def _start_stubs(sequence):
    """Generate many start points for each item in a sequence."""
    for word in sequence:
        num_starts = min(3, len(word))
        start_range = range(min(1, len(word)), len(word) + 1)
        for size in random.sample(start_range, num_starts):
            yield word[:size]


TEST_CASES = chain(
    (''.join(case) for case in STR_EDGE_CASES if END_CHAR not in case),
    _make_words(),
)
TEST_CASES = ((sequence, start) for sequence in TEST_CASES
              for start in _start_stubs(sequence))


@pytest.fixture(scope='function', params=TEST_CASES)
def new_trie(request):
    """Return a new empty instance of MyQueue."""
    sequence, start = request.param
    contains = set(sequence)
    instance = ClassDef()

    for item in sequence:
        instance.insert(item)

    to_insert = 'superuniquestring'
    to_remove = random.choice(sequence)

    longest = max(sequence, key=len) if sequence else ''
    contain_false_longer = longest + 'more'
    contain_false_shorter = longest

    while contain_false_shorter and contain_false_shorter in contains:
        contain_false_shorter = contain_false_shorter[:-1]
    if not contain_false_shorter:
        contain_false_shorter = 'superduperuniquestring'

    traverse = set(word for word in sequence if word.startswith(start))

    return TrieFixture(
        instance,
        sequence,
        contains,
        to_insert,
        to_remove,
        contain_false_shorter,
        contain_false_longer,
        start,
        traverse,
    )


@pytest.mark.parametrize('method_name', REQ_METHODS)
def test_has_method(method_name):
    """Test that graph has all the correct methods."""
    assert hasattr(ClassDef(), method_name)


def test_contains_all(new_trie):
    """Check that every item in the sequence is contained within the Trie."""
    assert all((new_trie.instance.contains(val) for val in new_trie.sequence))


def test_contains_false_shorter(new_trie):
    """Check that an item similar to one in Trie but shorter returns False."""
    assert not new_trie.instance.contains(new_trie.contain_false_shorter)


def test_contains_false_longer(new_trie):
    """Check that an item similar to one in Trie but longer returns False."""
    assert not new_trie.instance.contains(new_trie.contain_false_longer)


def test_insert(new_trie):
    """Check that a new item can be inserted and then contains is true."""
    new_trie.instance.insert(new_trie.to_insert)
    assert new_trie.instance.contains(new_trie.to_insert)


def test_remove(new_trie):
    """Check remove removes item."""
    new_trie.instance.remove(new_trie.to_remove)
    assert not new_trie.instance.contains(new_trie.to_remove)
    for x in new_trie.contains - set([new_trie.to_remove]):
        assert new_trie.instance.contains(x)


def test_remove_error(new_trie):
    """Check remove throws error when word to delete not in trie."""
    with pytest.raises(REMOVE_ERROR):
        new_trie.instance.remove('supercalifuniquestring')


def test_size(new_trie):
    """Check size..."""
    assert new_trie.instance.size() == len(new_trie.contains)


def test_size_after_remove(new_trie):
    """Check size decreases when delete."""
    num_remove = random.randint(1, len(new_trie.contains))
    to_remove = random.sample(new_trie.contains, num_remove)
    for word in to_remove:
        new_trie.instance.remove(word)
    assert new_trie.instance.size() == len(new_trie.contains) - len(to_remove)


def test_traversal_generator(new_trie):
    """Test that traversal method returns a generator."""
    assert isgenerator(new_trie.instance.traversal())


def test_traversal(new_trie):
    """Check that traversal returns all items contained in the Trie."""
    result = new_trie.instance.traversal(new_trie.start)
    assert set(result) == new_trie.traverse


def test_traversal_false_shorter(new_trie):
    """Check traversal doesn't return item similar but shorter."""
    result = new_trie.instance.traversal(new_trie.start)
    assert new_trie.contain_false_shorter not in set(result)


def test_traversal_false_longer(new_trie):
    """Check traversal doesn't return item similar but longer."""
    result = new_trie.instance.traversal(new_trie.start)
    assert new_trie.contain_false_longer not in set(result)
