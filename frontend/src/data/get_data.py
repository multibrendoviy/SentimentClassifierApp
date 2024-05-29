"""
Программа: Получение данных по пути и чтение
Версия: 1.0
"""

import streamlit as st

from io import BytesIO, StringIO
import io
import requests
import json
import pandas as pd
from typing import Dict, Tuple


@st.cache_data
def get_eda_stats(endpoint: object) -> Tuple[pd.DataFrame, list, list]:
    """
    Кэширование результатов запроса датафрейма и списка топ-30 слов для визуализации данных
    :param endpoint: endpount
    :return data: датафрейм для EDA
    :return words: список топ-30 слов
    :return count: количество употребления слов в корпусе
    """
    response = requests.post(endpoint, timeout=8000)
    data_csv = StringIO(response.json()["data_csv"])

    words = json.loads(response.json()["words"])
    count = json.loads(response.json()["count"])
    data = pd.read_csv(data_csv)

    return data, words, count


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
