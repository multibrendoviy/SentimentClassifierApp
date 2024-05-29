"""
Программа: Разделение данных на train/test
Версия: 1.0
"""

import pandas as pd
from sklearn.model_selection import train_test_split


def split_train_test(data: pd.DataFrame, preprocessing_config: dict) -> None:
    """
    Разделение датасета на обучающий и тестовый с сохранением в csv файл
    :param data: dataframe
    :param preprocessing_config: словарь с конфигурацией
    """

    train_data, test_data = train_test_split(
        data,
        stratify=data.target,
        test_size=preprocessing_config["test_size"],
        random_state=preprocessing_config["random_state"],
    )

    train_data = train_data.drop_duplicates().reset_index(drop=True)
    test_data = test_data.drop_duplicates().reset_index(drop=True)
    train_data.to_csv(preprocessing_config["train_path"], index=False)
    test_data.to_csv(preprocessing_config["test_path"], index=False)


def get_train_test_data(data: pd.DataFrame, preprocessing_config: dict) -> tuple:
    """
    Разделение датафрейма на тестовый и обучающий
    :param data: dataframe
    :param preprocessing_config: словарь с конфигурацией
    :return train_data
    :return test_data
    """
    train_data, test_data = train_test_split(
        data,
        stratify=data.target,
        test_size=preprocessing_config["test_size"],
        random_state=preprocessing_config["random_state"],
    )

    return train_data, test_data
