"""
Программа получения данных из файла формата csv
Версия: 1.0
"""

import pandas as pd
from typing import Text


def get_dataset(path: Text) -> pd.DataFrame:
    """
    Получение датасета из файла по указанному пути
    param: path: путь к файлу
    return: датасет
    """
    data = pd.read_csv(path)

    return data.drop_duplicates().reset_index(drop=True)
