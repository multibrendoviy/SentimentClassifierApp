"""
Программа: Получение данных по пути и чтение
Версия: 1.0
"""
import streamlit as st

from io import BytesIO
import io

import pandas as pd
import numpy as np
from rusenttokenize import ru_sent_tokenize

from collections import Counter
from itertools import chain
from functools import lru_cache
from typing import Dict, Tuple

import re


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


@lru_cache(512)
def get_data_for_eda(dataset_path: str) -> pd.DataFrame:
    """
    Загрузка датасета из файла и вычисление статистик для EDA-анализа
    :param dataset_path: путь к файлу
    :return: датафрейм
    """
    data = pd.read_csv(dataset_path)
    data['reviewText'] = data.reviewText.transform(lambda x: clean_text(x))

    # Количество слов
    data["Words_count"] = data.reviewText.apply(lambda x: len(x.split()))

    # Количество отзывов
    data["Sentences_count"] = data.reviewText.apply(lambda x: len(ru_sent_tokenize(x)))
    # Длина отзывов, выраженная количеством символов
    data["Review_length"] = data.reviewText.str.len()

    # Средняя длина слов
    data["Mean_word_length"] = data.reviewText.apply(
        lambda x: np.mean([len(t) for t in x.split()])
    )
    # Средняя длина предложений
    data["Mean_sentence_length"] = data.reviewText.apply(
        lambda rev: np.mean([len(sent) for sent in ru_sent_tokenize(rev)])
    )
    return data


def get_most_freq_words(data: pd.DataFrame) -> tuple:
    """
    Получение списка самых используемых слов в корпусе отзывов
    :param data: датафрейм
    :return: words: список слов
    :return: count: сколько раз слово встречается в корпусе
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
        ] for rev in data.reviewText.tolist()]

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


def load_data(
    data: str, type_data: str
) -> Tuple[pd.DataFrame, Dict[str, Tuple[str, BytesIO, str]]]:
    """
    Получение данных и преобразование в тип BytesIO для обработки в streamlit
    :param data: данные
    :param type_data: тип датасет (train/test)
    :return: датасет, датасет в формате BytesIO
    """
    dataset = pd.read_csv(data)

    st.write("Dataset load")
    st.write(dataset.head())

    # Преобразовать dataframe в объект BytesIO (для последующего анализа в виде файла в FastAPI)
    dataset_bytes_obj = io.BytesIO()
    # запись в BytesIO буфер
    dataset.to_csv(dataset_bytes_obj, index=False)
    # Сбросить указатель, чтобы избежать ошибки с пустыми данными
    dataset_bytes_obj.seek(0)

    files = {
        "file": (f"{type_data}_dataset.csv", dataset_bytes_obj, "multipart/form-data")
    }
    return dataset, files
