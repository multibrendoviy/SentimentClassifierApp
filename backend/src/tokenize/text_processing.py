"""
Очистка текста после парсинга
Версия 1.0
"""

import re
import pandas as pd
from collections import Counter
from itertools import chain
from typing import Tuple, List


def clean_text(s: str) -> str:
    """
    Очистка текста после парсинга
    """

    # удаление пробелов в начале и в конце строки
    s = s.strip()
    # приведение букв в нижный регистр
    s = s.lower()
    # отделение пробелами символов ".", ",", "!", "?"
    s = re.sub(r"([.,!?])", r" \1 ", s)
    # заменить на пробелы все символы, кроме а-я, А-Я, ".", ",", "!", "?"
    s = re.sub(r"[^а-яА-Я.,!?]+", " ", s)
    # убрать дублирующие пробелы
    s = re.sub(r"\s{2,}", " ", s)
    # убрать пробелы в начале и в конце строки
    s = s.strip()
    return s


def get_most_freq_words(data: pd.DataFrame) -> Tuple[List, List[int]]:
    """
    Получение списка самых используемых слов в корпусе отзывов
    :param data: dataframe
    :return words: список слов топ-30 слов
    :return count: количество употреблений слов в корпусе
    """
    punctuation_marks = [
        "!",
        ",",
        "(",
        ")",
        ":",
        "-",
        "?",
        ".",
        "..",
        "...",
        "«",
        "»",
        ";",
        "–",
        "--",
    ]

    # Корпус всех отзывов (без учета стоп-слов)
    corpus = [
        [
            word
            for word in rev.split()
            if word not in punctuation_marks and len(word) > 3
        ]
        for rev in data.reviewText.tolist()
    ]

    # Словарь корпуса
    counter = Counter(chain(*corpus))

    # Отсортированные по убыванию пары словаря (слово - количество употреблений в корпусе)
    most = counter.most_common()

    words = []
    count = []

    # 30 самых часто встречаемых слов в словаре
    for term in most[:30]:
        words.append(term[0])
        count.append(term[1])
    return words, count
