"""
Программа тренировки модели Bert
Версия 1.0
"""
import torch
import transformers
from transformers import BertForSequenceClassification, Trainer, TrainingArguments

from ..train.metrics import compute_metrics, save_metrics
from ..tokenize.bert_embeddings import CustomDataset
import json


def get_device():
    """
    Использование GPU при возможности
    :return:
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    return device


def bert_training(
    train_config: dict,
    train_dataset: CustomDataset,
    eval_dataset: CustomDataset,
    test_dataset: CustomDataset,
    requires_grad: bool = False,
) -> transformers.Trainer:
    """
    Обучение модели Bert
    :param train_config: конфигурационный файл
    :param train_dataset: датасет для обучения
    :param eval_dataset: датасет для предсказания на валидационный выборке
    :param test_dataset: тестовый датасет для оценки метрик
    :param requires_grad: обновление весовых коэффициентов
    :return Trainer, включающий в себя обученную модель
    """

    # Параметры для обучения
    args = TrainingArguments(
        train_config["trainer_path"],
        evaluation_strategy="epoch",
        save_strategy="no",
        learning_rate=train_config["learning_rate"],
        num_train_epochs=train_config["epochs"],
        weight_decay=train_config["weight_decay"],
        push_to_hub=False,
        per_device_train_batch_size=train_config["per_device_batch_size"],
    )

    # Загрузка предобученной модели классификации
    model = BertForSequenceClassification.from_pretrained(train_config["model_path"])

    # Заморозка/разморозка весов в зависимости от параметра
    for param in model.bert.parameters():
        param.requires_grad = requires_grad

    # Перенос вычислений модели на GPU
    model.to(get_device())

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
    # Обучение
    trainer.train()

    # Сохранение метрик
    save_metrics(test_dataset, trainer, train_config["metrics_path"])

    # Cохранение логов обучения модели
    save_log_history(train_config, trainer)

    return trainer


def save_model(trainer: transformers.Trainer, test_config: dict) -> None:
    """
    Сохранение  обученной модели
    :param trainer: объект класса Trainer
    :param test_config: конфигурационный файл
    :return:
    """
    # Сохранение модели
    trainer.save_model(test_config["model_path"])

    # Изменение название классов
    id2label = {"0": "Negative", "1": "Positive"}
    label2id = {"Negative": 0, "Positive": 1}

    # Сохранение конфигурационного файла модели с новыми названиями классов
    with open(f'{test_config["model_path"]}/config.json', "r+") as json_file:
        params = json.load(json_file)
        params["id2label"] = id2label
        params["label2id"] = label2id
        json_file.seek(0)
        json.dump(params, json_file)
        json_file.truncate()


def model_load(test_config: dict) -> transformers.BertForSequenceClassification:
    """
    Загрузка обученной модели
    :param test_config: конфигурационный файл
    :return: model: обученная модель классификации Bert
    """
    model = BertForSequenceClassification.from_pretrained(test_config["model_path"])

    return model


def save_log_history(train_config: dict, trainer: transformers.Trainer) -> None:
    """
    :param train_config: конфигурационный файл
    :param trainer: объект trainer обучавший модель
    """

    auc = []
    eval_loss = []

    for step in trainer.state.log_history:
        try:
            auc.append(step["eval_roc_auc"])
            eval_loss.append(step["eval_loss"])

        except KeyError:
            continue

    stats = {
        "auc": auc,
        "eval_loss": eval_loss
    }

    with open(train_config["trainer_log_path"], "w") as json_file:
        json.dump(stats, json_file)
