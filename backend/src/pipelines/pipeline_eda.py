"""
Программа выполнения пайплайна получение характеристик для EDA
Версия 1.0
"""

import pandas as pd

from ..data.get_dataset import get_data_for_eda
from ..tokenize.text_processing import get_most_freq_words

from typing import Tuple
import yaml


def pipeline_eda(config_path: str) -> Tuple[pd.DataFrame, list, list]:
    """
    Пайплайн вычисления характеристик для EDA
    :param config_path: пусть к конфигурационному файлу
    :return data: датафрейм для EDA
    :return words: список топ-30 слов
    :return count: количество употребления слов в корпусе
    """

    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    preprocessing_config = config["preprocessing"]

    eda_data = get_data_for_eda(preprocessing_config["train_path"])
    words, count = get_most_freq_words(eda_data)

    return eda_data, words, count
