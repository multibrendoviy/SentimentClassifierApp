# Анализ тональности отзывов клиентов онлайн-магазина "Ozon"

Веб-сервис для анализа тональности отзывов клиентов на примере онлайн-магазина "Ozon".        
Тональность определялась как позитив/негатив (метки 1/0). 
Отзывы, которые клиенты оценивали до 3 баллов включительно расценивались как негативные (0), 4,5 баллов - позитивные (1). 
В качестве классификатора тональности используется предобученная модель Bert для русского языка rubert-tiny2, дообученная на более чем 11000 
отзывах клиентов онлайн магазина "Ozon".

# Инструкция
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


