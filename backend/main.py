import uvicorn
import io
import re
import json

from fastapi import FastAPI, HTTPException
from fastapi import File
from fastapi import UploadFile
from pydantic import BaseModel, field_validator, ValidationError

from src.pipelines.pipeline import pipeline_training
from src.pipelines.pipeline_eda import pipeline_eda
from src.evaluate.evaluate import pipeline_evaluate, evaluate_from_input
from src.train.metrics import load_metrics
from src.data.get_reviews import get_reviews

import warnings

warnings.filterwarnings("ignore")


app = FastAPI()

CONFIG_PATH = "../config/config.yaml"


@app.get("/hello")
def welcome():
    """
    Hello
    :return: message
    """

    return {"message": "Hello Data Scientist!"}


class SentimentRequest(BaseModel):
    """
    Валидация введенного пользователем отзыва
    """

    text: str

    @field_validator("text")
    @classmethod
    def check_input(cls, s: str):
        # Проверка, что строка начинается с русской буквы или цифры
        if not re.match(r"^[а-яА-Я0-9]", s):
            raise ValueError("Отзыв должен начинаться с русской буквы или цифры.")

        # Проверка, что в строке есть хотя бы одно русское слово не менее 4 букв
        if not re.search(r"\b[а-яА-Я]{4,}\b", s):
            raise ValueError(
                "Отзыв должен содержать хотя бы одно русское слово не менее 4 букв."
            )

        return s


class URLRequest(BaseModel):
    """
    Валидации введенного URL-запроса
    """

    url: str
    page_count: int

    @field_validator("url")
    @classmethod
    def check_url(cls, s: str):
        pattern = r"^https:\/\/otzovik\.com\/reviews\/[a-zA-Z0-9_-]+\/\d+\/$"
        if re.match(pattern, s):
            return s
        else:
            raise ValueError("Некорректный URL для парсинга")

    @field_validator("page_count")
    @classmethod
    def check_count(cls, p: int):
        if p > 100:
            raise ValueError("Превышено количество страниц для парсинга")
        else:
            return p


@app.post("/compute_eda")
def get_eda_stats():
    """
    Получение статистика для EDA
    """
    eda_data, words, count = pipeline_eda(config_path=CONFIG_PATH)
    buffer = io.StringIO()
    eda_data.to_csv(buffer, index=False)
    buffer.seek(0)

    words_json = json.dumps(words)
    count_json = json.dumps(count)

    return {"data_csv": buffer.getvalue(), "words": words_json, "count": count_json}


@app.post("/train")
def training():
    """
    Обучение модели, логирование метрик
    """

    pipeline_training(config_path=CONFIG_PATH, requires_grad=True)
    metrics = load_metrics(config_path=CONFIG_PATH)

    return {"metrics": metrics}


@app.post("/predict")
def prediction(file: UploadFile = File(...)):
    """
    Предсказание модели по данным из файла
    """

    predictions, stats = pipeline_evaluate(config_path=CONFIG_PATH, data_path=file.file)

    assert isinstance(predictions, list), "Результат не соответствует типу list"

    return {"predictions": predictions[:5], "stats": stats}


@app.post("/predict_from_input")
def prediction_from_input(request: SentimentRequest):
    """
    Предсказание модели по введенному тексту отзыва
    """

    try:
        probabilities, predicted_label = evaluate_from_input(
            config_path=CONFIG_PATH, text=request.text
        )

        return {"Probabilities": probabilities, "Predicted_label": predicted_label}

    except ValidationError as e:
        raise HTTPException(status_code=400, detail=e.errors())


@app.post("/scrape")
def scrape_data_from_url(request: URLRequest):
    """Получение отзывов с сайта Otzovik.com"""

    try:
        get_reviews(
            config_path=CONFIG_PATH, url=request.url, page_count=request.page_count
        )

    except ValidationError as e:
        raise HTTPException(status_code=400, detail=e.errors)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=80)
