"""
Программа: Получение метрик
Версия: 1.0
"""
import json

import numpy as np
import yaml
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
)
import evaluate
from scipy.special import softmax
import transformers
from ..tokenize.bert_embeddings import CustomDataset


def create_dict_metrics(
    y_test: list, y_predict: np.ndarray, y_probability: np.ndarray
) -> dict:
    """
    Получение словаря с метриками для задачи классификации и запись в словарь
    :param y_test: реальные данные
    :param y_predict: предсказанные значения
    :param y_probability: предсказанные вероятности
    :return словарь с метриками
    """
    dict_metrics = {
        "roc_auc": round(roc_auc_score(y_test, y_probability[:, 1]), 3),
        "precision": round(precision_score(y_test, y_predict), 3),
        "recall": round(recall_score(y_test, y_predict), 3),
        "f1": round(f1_score(y_test, y_predict), 3),
        "logloss": round(log_loss(y_test, y_probability), 3),
    }
    return dict_metrics


def compute_metrics(eval_preds) -> dict:
    """
    Расчет метрики roc-auc для обучения модели
    :param eval_preds: предсказания модели
    :return: словарь с метрикой ROC-AUC:
    """

    metric = evaluate.load("roc_auc")
    logits, labels = eval_preds
    predictions = softmax(logits)[:, 1]
    res = metric.compute(prediction_scores=predictions, references=labels)

    return {"roc_auc": res["roc_auc"]}


def save_metrics(
    test_data: CustomDataset, trainer: transformers.Trainer, metric_path: str
) -> None:
    """
    Сохранение метрик
    :param test_data: тестовый датасет
    :param trainer: Trainer
    :param metric_path: путь для сохранения метрик

    """
    y_pred_proba, _, _ = trainer.predict(test_data)
    y_pred = np.argmax(y_pred_proba, axis=1)

    result_metrics = create_dict_metrics(
        y_test=test_data.labels,
        y_predict=y_pred,
        y_probability=y_pred_proba,
    )
    with open(metric_path, "w") as file:
        json.dump(result_metrics, file)


def load_metrics(config_path: str) -> str:
    """
    Получение метрик из файла
    :param config_path: путь до конфигурационного файла
    :return: метрики
    """
    # get params
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    with open(config["train"]["metrics_path"]) as json_file:
        metrics = json.load(json_file)

    return metrics
