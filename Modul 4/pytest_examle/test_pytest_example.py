import pytest
from pytest_example import add, multiply, find_longest_word

def test_add():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0
    assert add(0, 0) == 0

def test_multiply():
    assert multiply(2, 3) == 6
    assert multiply(-1, 1) == -1
    assert multiply(0, 0) == 0

def test_find_longest_word():
    assert find_longest_word('hello world') == 'hello'