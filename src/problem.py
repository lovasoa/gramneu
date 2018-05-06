#!/usr/bin/env python3
# coding=utf-8
# Copyright 2018 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Data generators for grammar correction data-sets."""

# Dependency imports
import os
import random
from typing import Iterator, Tuple

from . import homonyms

import glob

from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry

import tensorflow as tf

import nltk.tokenize

WIKI_FILES_PATTERN = "/home/ophir/Bureau/wikiextractor/extracted/*/wiki_[012]*"
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


def generate_from_wikifile(wiki_filename: str, homonyms_dict: dict) -> Iterator[Tuple[str, str]]:
    tf.logging.info("Reading wikifile '%s'", wiki_filename)
    with open(wiki_filename) as wiki_file:
        for line in wiki_file:
            if is_valid_paragraph(line):
                yield from add_mistakes(line, homonyms_dict)


@registry.register_problem
class FixGrammarMistakes(text_problems.Text2TextProblem):
    """Problem spec for WMT En-Fr translation."""

    @property
    def multiprocess_generate(self):
        """Whether to generate the data in multiple parallel processes."""
        return True

    @property
    def num_generate_tasks(self):
        return os.cpu_count()

    def prepare_to_generate(self, data_dir, tmp_dir):
        """Prepare to generate data in parallel on different processes.

        This function is called if multiprocess_generate is True.

        Some things that might need to be done once are downloading the data
        if it is not yet downloaded, and building the vocabulary.

        Args:
          data_dir: a string
          tmp_dir: a string
        """
        self.get_or_create_vocab(data_dir, tmp_dir)

    @property
    def max_subtoken_length(self):
        """Maximum subtoken length when generating vocab."""
        return 100

    @property
    def max_samples_for_vocab(self):
        return 2 ** 14  # ~ 16k

    @property
    def is_generate_per_split(self):
        return False

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        homonyms_dict = homonyms.load(data_dir)
        for wiki_file in glob.iglob(WIKI_FILES_PATTERN):
            for correct_phrase, mistake_phrase in generate_from_wikifile(wiki_file, homonyms_dict):
                # tf.logging.info("%s -> %s", mistake_phrase, correct_phrase)
                yield {
                    "inputs": mistake_phrase,
                    "targets": correct_phrase
                }
