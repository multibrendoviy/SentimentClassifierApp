"""
Программа выполнения пайплайна обучения Bert
Версия 1.0
"""
import yaml
from ..data.get_dataset import get_dataset
from ..data.split_data import split_train_test, get_train_test_data
from ..transform.transform import transform_labels, check_data, pipeline_preprocess
from ..train.train import bert_training, save_model
from ..pipelines.get_embeddings import get_bert_embeddings


def pipeline_training(config_path: dict, requires_grad: bool = False) -> None:
    """
    Пайплайн обучения модели Bert
    :param config_path: конфигурационный файл
    :param requires_grad: обновление весовых коэффициентов
    :return:
    """
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    preprocessing_config = config["preprocessing"]
    train_config = config["train"]
    test_config = config["test"]

    # Загрузка исходного датасета
    raw_data = get_dataset(preprocessing_config["raw_path"])
    data = transform_labels(raw_data)

    # Разделение на train/test
    split_train_test(data, preprocessing_config)

    # Загрузка предобработанных датасетов из файла
    train_df = get_dataset(preprocessing_config["train_path"])
    test_df = get_dataset(preprocessing_config["test_path"])

    train_df = pipeline_preprocess(train_df, flg2eval=False)
    test_df = pipeline_preprocess(test_df, flg2eval=False)

    # Разделение train датасета на train/validation
    train_df, val_df = get_train_test_data(train_df, preprocessing_config)

    # Получение объектов, содержащих bert-эмбеддинги
    train_dataset = get_bert_embeddings(train_df, train_config)
    val_dataset = get_bert_embeddings(val_df, train_config)
    test_dataset = get_bert_embeddings(test_df, train_config)

    # Обучение модели
    trainer = bert_training(
        train_config, train_dataset, val_dataset, test_dataset, requires_grad=True
    )

    # Cохранение обученной модели
    save_model(trainer, test_config)
