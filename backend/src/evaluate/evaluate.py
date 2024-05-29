"""
Программа выполнения предсказаний из файла
Версия: 1.0
"""

import yaml
import numpy as np
from transformers import Trainer, TextClassificationPipeline, BertTokenizer
from collections import Counter
from typing import Tuple

from ..data.get_dataset import get_dataset
from ..pipelines.get_embeddings import get_bert_embeddings
from ..transform.transform import pipeline_preprocess
from ..train.train import model_load


def pipeline_evaluate(
        config_path: str, data_path: str = None
) -> Tuple[list, dict]:
    """
    Пайплайн выполнения предсказаний из файла обученной модели
    :param config_path: конфигурационный файл
    :param data_path: путь к файлу
    :return predictions: предсказания
    :return stats: словарь со статистикой предсказаний
    """
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

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

    predictions = y_pred.tolist()
    stats = get_sentiment_stats(predictions)

    return predictions, stats


def evaluate_from_input(config_path: str, text: str) -> Tuple[dict, str]:
    """
    Предсказание тональности введенного пользователем отзыва
    :param config_path: пусть к конфигурационному файлу
    :param text: текст отзыва
    :return probabilities: словарь с вероятностями принадлежности к классам
    :return predicted_label: предсказанный класс
    """

    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    train_config = config["train"]
    test_config = config["test"]

    pipe = TextClassificationPipeline(
        model=model_load(test_config),
        tokenizer=BertTokenizer.from_pretrained(train_config["tokenizer_path"]),
        return_all_scores=True,
    )
    raw_probs = pipe(text)[0]
    probabilities = {item["label"]: item["score"] for item in raw_probs}
    predicted_label = max(probabilities, key=probabilities.get)

    return probabilities, predicted_label


def get_sentiment_stats(preds: list) -> dict:
    """
    Получение статистики классификации модели по данным из файла
    :param preds: предсказания модели
    :return: словарь со относительным количеством позитивных/негативных предсказаний
    """
    counter = Counter(preds)
    return {
        "Negative": f"{round(counter[0] / len(preds) * 100, 1)} %",
        "Positive": f"{round(counter[1] / len(preds) * 100, 1)} %",
    }
