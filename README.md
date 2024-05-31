<<<<<<< HEAD
# Описание
Веб-сервис для анализа тональности проекта отзывов клиентов онлайн-магазина "Ozon" (позитив/негатив).

В качестве классификатора тональности используется предобученная модель Bert, дообученная на корпусе
11к отзывов.

Источник отзывов - otzovik.com

<img src="demo/demo.gif" width="900" height="550" />

___

# Инструкция
=======
# Анализ тональности отзывов клиентов онлайн-магазина "Ozon"
>>>>>>> d94e685a1e34f27f4c0c984dfed002c1ad10da7c

## Клонирование

`git clone https://github.com/multibrendoviy/SentimentClassifierApp.git`


## Запуск приложения без Docker
Для запуска проекта на локальном хосте без докера предварительно необходимо раскоментировать
следующие строки в файле `config/config.yaml`:
```python
exploratory: 'http://localhost:8000/compute_eda'
train: 'http://localhost:8000/train'
prediction_input: 'http://localhost:8000/predict_from_input'
prediction_from_file: 'http://localhost:8000/predict'
scrape: 'http://localhost:8000/scrape'
 ```
и соответственно закомментировать следующие строки:
```python
#exploratory: 'http://fastapi:8000/compute_eda'
#train: 'http://fastapi:8000/train'
#prediction_input: 'http://fastapi:8000/predict_from_input'
#prediction_from_file: 'http://fastapi:8000/predict'
#scrape: 'http://fastapi:8000/scrape'
 ```
Cначала необходимо запустить Fastapi. Далее запускаете Streamlit.

Веб-сервис для анализа тональности отзывов клиентов на примере онлайн-магазина "Ozon".     
Тональность определялась как позитив/негатив (метки 1/0).
Отзывы, которые клиенты оценивали до 3 баллов включительно расценивались как негативные (0), 4,5 баллов - позитивные (1).
В качестве классификатора тональности используется предобученная модель Bert для русского языка rubert-tiny2, дообученная на более чем 11000
отзывах клиентов онлайн магазина "Ozon".

```bash
git clone https://github.com/multibrendoviy/SentimentClassifierApp.git

## Запуск FastAPI

- Запуск fastapi  

`cd backend`

`uvicorn main:app --host=0.0.0.0 --port=8000 --reload --log_config==log_config.yaml`

## Запуск Streamlit

`cd frontend`

`streamlit run main.py`

И приложение будет доступно по адресу http://localhost:8501 

___

Запуск с помощью Docker Compose. Предварительно убедиться что в файле `config/config.yaml`
раскоментированы эндпоинты с /fastapi/ и закоментированы эндпоинты с /localhost/.

## Docker

- Запуск образа Fastapi из директории mlops_sentiment_project

`docker build -t fastapi:ver1 backend -f backend/Dockerfile`

- Запуск образа Streamlit из директории mlops_sentiment_project

`docker build -t streamlit:ver1 frontend -f frontend/Dockerfile`

___

## Docker Compose

- Сборка сервисов из образов внутри backend/frontend и запуск контейнеров в автономном режиме

`docker compose up -d`

/

`docker compose up -d --build`


