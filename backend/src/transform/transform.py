"""
Программа преобразования датасета
Версия 1.0
"""
import pandas as pd
from ..tokenize.clean_text import clean_text
from typing import Union


def transform_data(
    data: pd.DataFrame, flg2eval=False
) -> Union[pd.DataFrame, pd.Series]:
    """
    Преобразование датасета: очистка текста и преобразование меток Label 1-5
    в целевую переменную 0/1
    :param data: датафрейм
    :param flg2eval: флаг, означающий что датасет предназначение для предсказания
    :return: датасет
    """

    # Если датасет raw_data - очистить текст и преобразовать метки
    if not flg2eval:
        data["target"] = data.label.transform(lambda x: 1 if x > 3 else 0)
        data = data.drop("label", axis=1)
        data["reviewText"] = data.reviewText.transform(lambda x: clean_text(x))

    else:
        # Возвращает датасет, состоящий только из текстов, для предсказания
        data = data["reviewText"]

    data.drop_duplicates().reset_index(drop=True)

    return data
