"""
Программа получения данных из файла формата csv
Версия: 1.0
"""

from typing import Text
from functools import lru_cache
import pandas as pd
import numpy as np
from rusenttokenize import ru_sent_tokenize

from ..tokenize.text_processing import clean_text


def get_dataset(path: Text) -> pd.DataFrame:
    """
    Получение датасета из файла по указанному пути
    :param path: путь к файлу
    :return data: dataframe
    """
    data = pd.read_csv(path)

    return data.drop_duplicates().reset_index(drop=True)


@lru_cache(maxsize=None)
def get_data_for_eda(dataset_path: str) -> pd.DataFrame:
    """
    Загрузка датасета из файла и вычисление статистик для EDA-анализа
    :param dataset_path: путь к файлу
    :return data: dataframe
    """

    data = pd.read_csv(dataset_path)
    data["reviewText"] = data.reviewText.transform(lambda x: clean_text(x))

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
