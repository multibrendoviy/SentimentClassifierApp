import streamlit as st
import yaml
import os

from src.train.training import start_training
from src.evaluate.evaluate import start_evaluate, start_evaluate_from_file
from src.data.get_data import load_data, get_data_for_eda, get_most_freq_words
from src.data.get_reviews import start_scraping
from src.plotting.charts import plot_bars, plot_boxplot, plot_kdeplot, plot_barplot


CONFIG_PATH = "../config/config.yaml"


def main_page():
    """
    Страница с описанием проекта
    """
    st.image(
        "https://static.vecteezy.com/system/resources/previews/026/226/883/original/"
        "sentiment-analysis-icon-illustration-vector.jpg",
        width=400,
    )

    st.markdown("# Описание проекта")
    st.title("MLOps project:  Sentiment Analysis Ozon Customers ")
    st.write(
        """
        Определение тональности отзывов клиентов онлайн-магазина "Ozon". 

        Источник отзывов - otzovik.com
        
        Тональность определялась как позитив/негатив (метки 1/0). 
        Отзывы, которые клиенты оценивали до 3 баллов 
        включительно расценивались как негативные (0), 4,5 баллов - позитивные (1).
        
        В качестве классификатора тональности используется предобученная модель Bert
         для русского языка rubert-tiny2, дообученная на более чем 11000 
         отзывах клиентов онлайн магазина "Ozon".
        """
    )

    # name of the columns
    st.markdown(
        """
        ### Описание полей 
        reviewText - содержание отзывов   
        label - пользовательская оценка
    """
    )


def exploratory():
    """
    Exploratory data analysis
    """
    st.markdown("# Exploratory data analysis")
    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    data = get_data_for_eda(dataset_path=config["preprocessing"]["train_path"])
    st.write(data.head())
    review_stats = st.sidebar.checkbox("Статистика отзывов")
    count_word_sent = st.sidebar.checkbox("Количество слов и предложений в отзывах")
    review_length = st.sidebar.checkbox("Длина отзывов")
    mean_word_len = st.sidebar.checkbox("Длина слов")
    mean_sent_len = st.sidebar.checkbox("Длина предложений")
    most_freq = st.sidebar.checkbox("Наиболее часто встречающиеся слова")

    if review_stats:
        st.pyplot(
            plot_bars(
                data=data,
            )
        )
        st.write(
            "В датасете присутсвует дисбаланс классов в сторону негативных отзывов."
        )

    if count_word_sent:
        col1, col2 = st.columns([0.5, 0.5])
        with col1:
            st.pyplot(
                plot_boxplot(data, "Words_count", "target", "Количество слов"),
                use_container_width=True,
            )

        with col2:
            st.pyplot(
                plot_boxplot(
                    data, "Sentences_count", "target", "Количество предложений"
                ),
                use_container_width=True,
            )
        st.write(
            "Негативные отзывы содержат больше слов и предложений, что кажется логичным: человек, "
            "довольный опытом покупки, отзыв либо не оставляет, либо делает это лаконично."
            " Негативные отзывы же содержат больше текста, ввиду того что покупатель "
            "старается описать суть проблемы, либо выплескивает эмоции."
        )

    if review_length:
        col1, col2 = st.columns([0.5, 0.5])
        with col1:
            st.pyplot(
                plot_boxplot(
                    data,
                    width=4,
                    height=5,
                    col_name="Review_length",
                    hue_name="target",
                    plot_name="Длина отзыва (количество символов)",
                ),
                use_container_width=True,
            )
        with col2:
            st.pyplot(
                plot_kdeplot(
                    data,
                    width=4,
                    height=5,
                    col_name="Review_length",
                    hue_name="target",
                    plot_name="Длина отзыва (количество символов)",
                )
            )

        st.write(
            "Логично, что и длина негативного отзыва, выраженная количеством символов, "
            "больше чем в позитивном."
        )

    if mean_word_len:
        col1, col2 = st.columns([0.5, 0.5])
        with col1:
            st.pyplot(
                plot_boxplot(data, "Mean_word_length", "target", "Средняя длина слов"),
                use_container_width=True,
            )

        with col2:
            st.pyplot(
                plot_kdeplot(data, "Mean_word_length", "target", "Средняя длина слов"),
                use_container_width=True,
            )
        st.write(
            "Средняя длина слова в позитивном отзыве чуть больше чем в негативном."
        )
    if mean_sent_len:
        col1, col2 = st.columns([0.5, 0.5])
        with col1:
            st.pyplot(
                plot_boxplot(
                    data, "Mean_sentence_length", "target", "Средняя длина предложений"
                ),
                use_container_width=True,
            )

        with col2:
            st.pyplot(
                plot_kdeplot(
                    data, "Mean_sentence_length", "target", "Средняя длина предложений"
                ),
                use_container_width=True,
            )
        st.write(
            "Визуальной разницы в длинах предложений отзывов, в разрезе целевой переменной, нет."
        )

    if most_freq:

        words, count = get_most_freq_words(data)
        st.pyplot(plot_barplot(data, words, count), use_container_width=True)

        st.write(
            "В топ-30 попали разные формы слов, касаемых непосредственно процесса покупки товаров, слов. "
            "Среди них отсуствуют слова, вносящие сами по себе резко негативный/позитивный оттенок"
        )


def training():
    st.markdown("# Training Bert")
    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    endpoint = config["endpoints"]["train"]
    if st.button("Start training"):
        start_training(config=config, endpoint=endpoint)


def prediction():
    st.markdown("# Prediction")
    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    endpoint = config["endpoints"]["prediction_input"]
    form = st.form(key="sentiment")
    user_input = form.text_area("Введите текст отзыва")
    predict = form.form_submit_button("Predict")

    if predict:
        if os.path.exists(config["test"]["model_path"]):
            start_evaluate(endpoint=endpoint, input=user_input)
        else:
            st.error("Сначала обучите модель")


def prediction_from_file():
    st.markdown("# Prediction from file")
    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    endpoint = config["endpoints"]["prediction_from_file"]

    upload_file = st.file_uploader("", type=["csv"],
                                   accept_multiple_files=False)
    if upload_file:
        dataset_csv_df, files = load_data(data=upload_file, type_data="test")
        # проверка на наличие сохраненной модели
        if os.path.exists(config["test"]["model_path"]):
            start_evaluate_from_file(
                data=dataset_csv_df, endpoint=endpoint, files=files
            )
        else:
            st.error("Сначала обучите модель")


def get_some_reviews():
    st.markdown("# Get some reviews from otzovik")
    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    endpoint = config["endpoints"]["scrape"]
    start_scraping(endpoint=endpoint)


def main():

    page_names_to_funcs = {
        "Описание проекта": main_page,
        "Exploratory data analysis": exploratory,
        "Training model": training,
        "Prediction": prediction,
        "Prediction from file": prediction_from_file,
        "Get some reviews": get_some_reviews,
    }
    selected_page = st.sidebar.selectbox("Выберите пункт",
                                         page_names_to_funcs.keys())
    page_names_to_funcs[selected_page]()


if __name__ == "__main__":
    main()
