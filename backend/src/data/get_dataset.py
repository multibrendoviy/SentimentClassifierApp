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
    cols = list(data.columns)
    # Если датасет уже обработан
    if cols == ["reviewText", "target"] or cols == ["reviewText", "label"]:
        return data

    else:
        # Если датасет содержит столбцы с другими именами
        data = data.rename(columns={list(data)[0]: "reviewText"})

        try:
            # Проверка датасета на количество признаков
            data = data.rename(columns={list(data)[1]: "label"})
        # Если датасет содержащего только тексты отзывов
        except IndexError:

            print("This data only for predicts and have no labels")

        return data.drop_duplicates().reset_index(drop=True)
