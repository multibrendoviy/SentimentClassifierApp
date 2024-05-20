import uvicorn
import logging
from logging.config import fileConfig
from fastapi import FastAPI, HTTPException
from fastapi import File
from fastapi import UploadFile
from pydantic import BaseModel, field_validator, ValidationError
from transformers import TextClassificationPipeline

from src.pipelines.pipeline import pipeline_training
from src.evaluate.evaluate import pipeline_evaluate
from src.train.metrics import load_metrics
from src.train.train import model_load
from src.data.get_reviews import get_reviews

import yaml
from transformers import BertTokenizer
import re

import warnings
warnings.filterwarnings("ignore")


fileConfig('log.ini')
logger = logging.getLogger(__name__)
logger.info('Starting API...')

app = FastAPI()


CONFIG_PATH = "../config/config.yaml"
config = yaml.load(open(CONFIG_PATH), Loader=yaml.FullLoader)
train_config = config['train']
test_config = config['test']


@app.get("/hello")
def welcome():
    """
    Hello
    :return: None
    """
    logger.info('Request for greeting message...')
    return {'message': 'Hello Data Scientist!'}


class SentimentRequest(BaseModel):
    """
    Класс для валидации введенного пользователем отзыва
    """
    text: str

    @field_validator('text')
    @classmethod
    def check_input(cls, s: str):
        # Проверка, что строка начинается с русской буквы
        if not re.match(r'^[а-яА-Я]', s):
            raise ValueError('Отзыв должен начинаться с русской буквы.')

        # Проверка, что в строке есть хотя бы одно русское слово не менее 4 букв
        if not re.search(r'\b[а-яА-Я]{4,}\b', s):
            raise ValueError('Отзыв должен содержать хотя бы одно русское слово не менее 4 букв.')

        return s


class URLRequest(BaseModel):
    """
    Класс для валидации введенного URL-запроса
    """
    url: str
    page_count: int

    @field_validator('url')
    @classmethod
    def check_url(cls, s: str):
        pattern = r'^https:\/\/otzovik\.com\/reviews\/[a-zA-Z0-9_-]+\/\d+\/$'
        if re.match(pattern, s):
            return s
        else:
            raise ValueError('Некорректный URL для парсинга')

    @field_validator('page_count')
    @classmethod
    def check_count(cls, p: int):
        if p > 100:
            raise ValueError('Превышено количество страниц для парсинга')
        else:
            return p


@app.post("/train")
def training():
    """
    Обучение модели, логирование метрик
    """
    logger.info('Starting training...')
    pipeline_training(config_path=CONFIG_PATH, requires_grad=True)
    metrics = load_metrics(config_path=CONFIG_PATH)

    return {"metrics": metrics}


@app.post("/predict")
def prediction(file: UploadFile = File(...)):
    """
    Предсказание модели по данным из файла
    """
    logger.info('Starting prediction from file...')
    predictions = pipeline_evaluate(
        config_path=CONFIG_PATH, data_path=file.file
    )

    assert isinstance(predictions, list), "Результат не соответствует типу list"

    return {"prediction": predictions[:5]}


@app.post("/predict_from_input")
def prediction_from_input(request: SentimentRequest):
        """
        Предсказание модели по введенному тексту
        """
        logger.info('Starting prediction from input...')
        try:

            pipe = TextClassificationPipeline(
                        model=model_load(test_config),
                        tokenizer=BertTokenizer.from_pretrained(train_config['tokenizer_path']),
                        return_all_scores=True
                    )

            return pipe(request.text)[0]

        except ValidationError as e:
            raise HTTPException(status_code=400, detail=e.errors())


@app.post("/scrape")
def scrape_data_from_url(request: URLRequest):
    """Получение отзывов с сайта Otzovik.com"""

    logger.info('Starting getting reviews...')

    try:
        get_reviews(config_path=CONFIG_PATH, url=request.url, page_count=request.page_count)

    except ValidationError as e:
        raise HTTPException(status_code=400, detail=e.errors)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=80)