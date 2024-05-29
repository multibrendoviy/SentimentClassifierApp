"""
Программа получения эмбеддингов Bert
Версия 1.0
"""

import pandas as pd
from transformers import BertTokenizer
from ..tokenize.bert_embeddings import PrepareData, CustomDataset


def get_bert_embeddings(data: pd.DataFrame, train_config: dict) -> CustomDataset:
    """
    Полный цикл представления датасета в эмбеддинги модели Bert для использования
    с объектом Trainer
    :param data: dataframe
    :param train_config: словарь с конфигурацией
    :return dataset: датасет
    """

    tokenizer = BertTokenizer.from_pretrained(train_config["tokenizer_path"])

    try:
        # Если датасет содержит реальные метки классов
        obj_1 = PrepareData(data.reviewText.tolist(), tokenizer)
        encodings = obj_1.transform()

        dataset = CustomDataset(encodings, data.target.tolist())

    except AttributeError:
        # Обработка для тестового датасета, не содержащего меток класса
        obj_1 = PrepareData(data.to_list(), tokenizer)
        encodings = obj_1.transform()
        dataset = CustomDataset(encodings)

    return dataset
