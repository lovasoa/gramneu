#!/usr/bin/env python3
# coding: utf-8
import json
import re
import sys
from pathlib import Path
from typing import List

import unidecode
import multiprocessing

FRENCH_DICT_FILENAME = "/usr/share/dict/french"

_replacements = {
    r'eau|au': 'o',
    r'qu|c(?=[aou]|\b)': 'k',
    r'sc|c|ç': 's',
    r'gu': 'g',
    r'ph': 'f',
    r'^h': '',
    r'([cflmnprst])\1': r'\1',
    r'oua': 'oi',
    r'(ant|ent|ees|ee|es|e|s|t|d|x)$': '',
    r'(ais|ait|ai|é|er)$': 'er',
    r'[aeui]+n': 'in',
}

_compiled_replacements = {
    re.compile(pat, re.I): value
    for pat, value in _replacements.items()
}


def normalize(word: str) -> str:
    for pat, rep in _compiled_replacements.items():
        word = re.sub(pat, rep, word)
    return unidecode.unidecode(word)


def get_homonyms_dict(words: List[str]) -> dict:
    by_sound = {}
    homonyms = {}
    with multiprocessing.Pool() as pool:
        normalized_words = pool.map(normalize, words, chunksize=100)
    for word, normalized in zip(words, normalized_words):
        same_sound = by_sound.setdefault(normalized, [])
        same_sound.append(word)
        homonyms[word] = same_sound
    return homonyms


def write_homonyms(outfile, dictionary_filename=FRENCH_DICT_FILENAME) -> dict:
    dictionary_words = [word.strip() for word in open(dictionary_filename)]
    homonyms_dict = get_homonyms_dict(dictionary_words)
    json.dump(homonyms_dict, outfile, indent=1)
    return homonyms_dict


def load(data_dir: str):
    homonyms_file = Path(data_dir) / "homonyms.json"
    try:
        return json.load(homonyms_file.open('r'))
    except (FileNotFoundError, json.JSONDecodeError):
        return write_homonyms(homonyms_file.open('w'))


if __name__ == "__main__":
    write_homonyms(sys.stdout)
