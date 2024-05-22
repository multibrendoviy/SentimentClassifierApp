"""
Программа выполнения предсказаний из файла
Версия: 1.0
"""

import yaml
import numpy as np
from transformers import Trainer

from ..data.get_dataset import get_dataset
from ..pipelines.get_embeddings import get_bert_embeddings
from ..transform.transform import pipeline_preprocess
from ..train.train import model_load


def pipeline_evaluate(config_path: str, data_path: str = None) -> list:
    """
    Пайплайн выполнения предсказаний из файла обученной модели
    :param config_path: конфигурационный файл
    :param data_path: путь к файлу
    :return:
    """
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    preprocessing_config = config["preprocessing"]
    train_config = config["train"]
    test_config = config["test"]

    # Если указан пусть к файлу
    if data_path:
        test_data = get_dataset(data_path)

    # Предобработка тестового датасета
    test_data = pipeline_preprocess(test_data, flg2eval=True)
    test_dataset = get_bert_embeddings(test_data, train_config)
    # Загрузка модели Bert
    model = model_load(test_config)
    test_trainer = Trainer(model)

    # Выполнение предсказаний
    raw_pred, _, _ = test_trainer.predict(test_dataset)
    y_pred = np.argmax(raw_pred, axis=1)

    prediction = y_pred.tolist()

    return prediction
