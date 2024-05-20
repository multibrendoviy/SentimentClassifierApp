import pandas as pd
import streamlit as st
import requests
from io import BytesIO

from ..plotting.charts import get_sentiment_emoji, create_sentiment_plot
from collections import Counter


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


def start_evaluate(endpoint: object, input: str) -> None:
    """
    Получение входных данных путем ввода в UI> вывод результата
    :param endpoint: endpoint
    :param input: введенный текст
    :return:
    """
    data = {"text": input}
    try:
        response = requests.post(endpoint, timeout=8000, json=data)
        response.raise_for_status()
        output = response.json()

        max_label = max(output, key=lambda x: x["score"])
        # Предсказанная метка
        predicted_label = max_label["label"]

        col1, col2, col3 = st.columns([2, 2, 2])
        st.markdown(
            """
            <style>
            .column {
                float: left;
                width: 33.33%;
                padding: 10px;
                box-sizing: border-box;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        with col1:
            # Вывод эмоджи
            st.write(f"Результат: {max_label['label']}!")
            sentiment_emoji = get_sentiment_emoji(predicted_label, size=120)
            st.markdown(f"{sentiment_emoji}", unsafe_allow_html=True)

        with col2:
            # Вывод словаря с вероятностями принадлежности к классам
            st.write(*output)

        with col3:
            # Построение барплота вероятностей
            probabilities = {item["label"]: item["score"] for item in output}
            fig = create_sentiment_plot(
                probabilities,
                width=200,
                height=300,
                title="Sentiment probabilities",
                xaxis="Sentiments",
                yaxis="Probability",
            )

            st.plotly_chart(fig, use_container_width=True)

    except requests.exceptions.RequestException:
        st.error(
            f"Отзыв должен начинаться с русской буквы и содержать хотя бы одно слово, содержащее"
            f"не менее 4 букв"
        )


def start_evaluate_from_file(data: pd.DataFrame, endpoint: object, files: BytesIO):
    """
    Получение входных данных путем загрузки из файла> вывод результата
    :param data: датасет
    :param endpoint: endpoint
    :param files:
    :return:
    """

    button_ok = st.button("Predict")
    if button_ok:
        # заглушка так как не выводим все предсказания
        data_ = data[:5]
        with st.spinner("Making predictions..."):

            output = requests.post(endpoint, files=files, timeout=8000)
            data_["predict"] = output.json()["prediction"]
            stats = get_sentiment_stats(data_.predict.tolist())
            st.write(data_.head())
            fig = create_sentiment_plot(
                stats,
                width=500,
                height=600,
                title="Sentiment stats",
                xaxis="Sentiments",
                yaxis="Percent",
            )

            st.plotly_chart(fig, use_container_width=True)
