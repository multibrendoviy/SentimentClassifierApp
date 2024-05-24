import pandas as pd
import streamlit as st
import requests


def start_scraping(config: dict, endpoint: object) -> None:

    with st.form("url"):
        url = st.text_input(
            "Вставьте URL-запрос со страницы с "
            "тизерами отзывов с  сайта otzovik.com "
        )
        page = st.number_input(
            "Укажите число страниц с отзывами",
            min_value=1,
            max_value=50,
            value=3,
            step=1,
        )
        submit = st.form_submit_button("Get data")
        st.write("Например: https://otzovik.com/reviews/ozon_ru_online_shop/200/")

    if submit:
        with st.spinner("Getting data..."):
            try:
                data = {"url": url, "page_count": page}
                response = requests.post(endpoint, timeout=8000, json=data)
                response.raise_for_status()
                st.write(
                    "Парсинг успешно завершен."
                    " Для выполнения предсказания перейдите"
                    "в раздел Prediction from file"
                )
                df = pd.read_csv(config['preprocessing']['scrape_path'])
                st.write(df.head())
                csv = df.to_csv(index=False)
                st.download_button(
                    label='Download File',
                    data=csv,
                    file_name='sparse_dataframe.csv',
                    mime='text/csv'
                )

            except requests.exceptions.RequestException:
                if response.status_code == 422:
                    error_msg = response.json()["detail"]
                    st.error(error_msg)
                else:
                    st.error("Ошибка парсинга")
