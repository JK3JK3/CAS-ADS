import re

def add(x, y):
    """This function adds two numbers."""
    return x + y

def multiply(x, y):
    """This function multiplies two numbers."""
    return x * y


def find_longest_word(sentence):
    """
    Finds the longest word in a given sentence.

    This function ignores punctuation and considers words to be sequences of
    alphabetic characters. If the input is not a string, empty, or contains no
    alphabetic words, it returns None. In case of a tie (multiple words
    of the same maximum length), it returns the first one encountered.

    Args:
        sentence (str): The input string to search within.

    Returns:
        str or None: The longest word found, or None if no valid word is found.
    """
    if not isinstance(sentence, str) or not sentence:
        return None

    # Remove punctuation and split the sentence into words
    words = re.findall(r'[a-zA-Z]+', sentence)

    if not words:
        return None

    # Find the longest word
    longest_word = ""
    for word in words:
        if len(word) > len(longest_word):
            longest_word = word
    
    return longest_word
            