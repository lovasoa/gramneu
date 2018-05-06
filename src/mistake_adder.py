# coding=utf-8
import random

import nltk

MAX_MISTAKE_PROBABILITY = 0.5  # Generate sentences with at most that proportion of error


def remove_special_chars(s: str):
    """Keep only latin characters in s. Helps reduce the vocabulary size"""
    latin_bytes = s.encode("latin-1", errors="xmlcharrefreplace")
    return latin_bytes.decode('latin-1')


def randomly_miswrite(word: str, homonyms_dict: dict, mistake_probability: float):
    if random.random() > mistake_probability:
        return word
    homonyms_list = homonyms_dict.get(word.lower(), (word,))
    return random.choice(homonyms_list)


def add_mistakes(paragraph: str, homonyms_dict: dict) -> Iterator[Tuple[str, str]]:
    """Add spelling mistakes to a text.
    Yields (correct_phrase, mistake_phrase)
    """
    for phrase in nltk.tokenize.sent_tokenize(paragraph, language="french"):
        words = nltk.tokenize.word_tokenize(phrase, language="french")
        words = list(map(remove_special_chars, words))  # Remove special characters for each word
        correct_phrase = ' '.join(words)
        yield correct_phrase, correct_phrase  # Train not to add mistakes to a correct sentence
        mistake_probability = random.random() * MAX_MISTAKE_PROBABILITY
        mistake_words = [randomly_miswrite(word, homonyms_dict, mistake_probability) for word in words]
        mistake_phrase = ' '.join(mistake_words)
        yield correct_phrase, mistake_phrase


def is_valid_paragraph(line: str):
    return line.strip() and not line.startswith("<doc") and not line.startswith("</doc")
