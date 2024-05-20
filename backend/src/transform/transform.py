"""
Программа преобразования датасета
Версия 1.0
"""

import pandas as pd
from ..tokenize.clean_text import clean_text
from typing import Union


def transform_labels(data: pd.DataFrame) -> pd.DataFrame:
    """
    Преобразование меток Label 1-5 в целевую переменную 0/1
    :param data: датафрейм
    :return: датасет
    """
    data["target"] = data.label.transform(lambda x: 1 if x > 3 else 0)
    data = data.drop("label", axis=1)
    return data


def check_data(data: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
    """
    Проверка датасета на названия, количество столбцов и типы данных в них содержащихся
    :param data: датасет
    :return: датасет
    """
    # Если Series - проверить тип данных
    if isinstance(data, pd.Series):
        if data.dtype in [str, object]:
            data = pd.DataFrame(data, columns=['reviewText'])
            return data
        else:
            raise TypeError("Неверный тип данных в серии.")

    if isinstance(data, pd.DataFrame):
        cols = list(data.columns)

        if cols in ["reviewText", "label", "target"]:
            return data
        # Если датасет содержит другие названия столбцов - проверить тип данных
        elif len(cols) == 2:
            for col in cols:
                if data[col].dtype in [str, object]:
                    data["reviewText"] = data[col]
                elif data[col].dtype == int:
                    data["label"] = data[col]
                else:
                    raise TypeError("Неверный тип данных в столбцах.")

            data = data[["reviewText", "label"]]
            return data

        elif len(cols) == 1:
            data.columns = ["reviewText"]
            if data["reviewText"].dtype in [str, object]:
                return data
            else:
                raise TypeError("Неверный формат данных.")

        else:
            raise TypeError("Неверный формат датасета.")

    else:
        raise TypeError(
            "Неподдерживаемый тип данных. Поддерживаются только pd.DataFrame и pd.Series."
        )


def pipeline_preprocess(
    data: Union[pd.DataFrame, pd.Series], flg2eval=False
) -> Union[pd.DataFrame, pd.Series]:
    """
    Пайплайн предобработки датасета
    :param: data - датасет
    :param: flg2eval - признак тестового датасета
    :return: data - предобработанный датасет
    """

    data = check_data(data)
    data["reviewText"] = data.reviewText.transform(lambda x: clean_text(x))

    if flg2eval:
        data = data["reviewText"]

    return data
