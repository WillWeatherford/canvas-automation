"""Standard tests for Trie data structure."""
from __future__ import unicode_literals

import pytest
import random
from itertools import chain
from collections import namedtuple
from importlib import import_module
# from inspect import isgenerator


from cases import STR_EDGE_CASES
MODULENAME = 'trie'
CLASSNAME = 'Trie'
ROOT_ATTR = 'root'
END_CHAR = '$'

module = import_module(MODULENAME)
ClassDef = getattr(module, CLASSNAME)


REQ_METHODS = [
    'insert',
    'contains',
    'size',
    'remove',
    # 'traversal',
]


TrieFixture = namedtuple(
    'TrieFixture', (
        'instance',
        'sequence',
        'contains',
        'to_insert',
        'to_remove',
        # 'contain_false_shorter',
        # 'contain_false_longer',
        # 'start',
        # 'traverse',
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


@pytest.fixture
def empty_trie_tree():
    """Empty trie tree fixture."""
    from trie import TrieTree
    return TrieTree()


@pytest.fixture(scope='function', params=TEST_CASES)
def new_trie(request):
    """Return a new empty instance of MyQueue."""
    sequence, start = request.param
    contains = set(sequence)
    instance = ClassDef()

    for item in sequence:
        try:
            instance.insert(item)
        # don't insert duplicate values, pass instead
        except(ValueError):
            pass

    to_insert = 'superuniquestring'
    to_remove = 'removethisstring'

    # longest = max(sequence, key=len) if sequence else ''
    # contain_false_longer = longest + 'more'
    # contain_false_shorter = longest

    # while contain_false_shorter and contain_false_shorter in contains:
    #     contain_false_shorter = contain_false_shorter[:-1]
    # if not contain_false_shorter:
    #     contain_false_shorter = 'superduperuniquestring'

    # traverse = set(word for word in sequence if word.startswith(start))

    return TrieFixture(
        instance,
        sequence,
        contains,
        to_insert,
        to_remove,
        # contain_false_shorter,
        # contain_false_longer,
        # start,
        # traverse,
    )


@pytest.mark.parametrize('method_name', REQ_METHODS)
def test_has_method(method_name):
    """Test that graph has all the correct methods."""
    assert hasattr(ClassDef(), method_name)


def test_insert(new_trie):
    """Check that a new item can be inserted and then contains is true."""
    new_trie.instance.insert(new_trie.to_insert)
    assert new_trie.instance.contains(new_trie.to_insert)


def test_insert_raises_type_error_if_not_string(empty_trie_tree):
    """Test insert raises TypeError if input is not a string."""
    with pytest.raises(TypeError):
        empty_trie_tree.insert(100)


def test_insert_duplicate_string_raises_error(empty_trie_tree):
    """Test inserting duplicate string raises ValueError."""
    empty_trie_tree.insert('duplicatestring')
    with pytest.raises(ValueError):
        empty_trie_tree.insert('duplicatestring')


def test_contains_all(new_trie):
    """Check that every item in the sequence is contained within the Trie."""
    assert all((new_trie.instance.contains(val) for val in new_trie.sequence))


def test_size_method_on_empty_tree_returns_0(empty_trie_tree):
    """Test size method returns 0 on empty trie tree."""
    assert empty_trie_tree.size() == 0


def test_size(new_trie):
    """Check that size method returns the same length as the test sequence."""
    assert new_trie.instance.size() == len(set(new_trie.sequence))


def test_remove(new_trie):
    """Check that after an item is removed it is no longer in the Trie."""
    new_trie.instance.insert(new_trie.to_remove)
    new_trie.instance.remove(new_trie.to_remove)
    assert not new_trie.instance.contains(new_trie.to_remove)


def test_remove_method_raise_error_if_bad_word(empty_trie_tree):
    """Test remove method raises error if word not in tree."""
    with pytest.raises(ValueError):
        assert not empty_trie_tree.remove('stringdefinintelynotinsequence')


#######################################
# Uncomment below for Trie Traversals #
#######################################

# def test_contains_false_shorter(new_trie):
#     """Check that item similar to one in Trie but shorter returns False."""
#     assert not new_trie.instance.contains(new_trie.contain_false_shorter)


# def test_contains_false_longer(new_trie):
#     """Check that an item similar to one in Trie but longer returns False."""
#     assert not new_trie.instance.contains(new_trie.contain_false_longer)


# def test_traversal_generator(new_trie):
#     """Test that traversal method returns a generator."""
#     assert isgenerator(new_trie.instance.traversal())


# def test_traversal(new_trie):
#     """Check that traversal returns all items contained in the Trie."""
#     result = new_trie.instance.traversal(new_trie.start)
#     assert set(result) == new_trie.traverse


# def test_traversal_false_shorter(new_trie):
#     """Check traversal doesn't return item similar but shorter."""
#     result = new_trie.instance.traversal(new_trie.start)
#     assert new_trie.contain_false_shorter not in set(result)


# def test_traversal_false_longer(new_trie):
#     """Check traversal doesn't return item similar but longer."""
#     result = new_trie.instance.traversal(new_trie.start)
#     assert new_trie.contain_false_longer not in set(result)
