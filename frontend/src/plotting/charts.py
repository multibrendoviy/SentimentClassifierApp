import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import plotly.graph_objects as go
import seaborn as sns
import json


def plot_text(ax: plt.Axes):
    """
    Выводит процентные значения на столбчатой диаграмме.

    :param: ax:
    :returns: fig
        ax (plt.Axes): Объект Axes, на котором отображается
        столбчатая диаграмма.
    """

    # Перебираем каждый столбец на диаграмме

    for p in ax.patches:
        # Вычисляем процентное значение высоты столбца
        percentage = "{:.1f}%".format(p.get_height())
        # Аннотируем столбец с процентным значением
        ax.annotate(
            percentage,  # Текст аннотации
            # Координаты аннотации (по центру столбца)
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            # Выравнивание текста
            ha="center",
            va="center",
            # Смещение текста относительно координаты
            xytext=(0, 10),
            # Использование смещения в "точках" (points)
            textcoords="offset points",
            fontsize=10,
        )


def plot_bars(data: pd.DataFrame) -> matplotlib.figure.Figure:
    """
    Выводит столбчатую диаграмму для данных DataFrame.

    :param: data: DataFrame с данными для построения столбчатой диаграммы
    :return: fig
    """

    sns.set_style("whitegrid")
    # Вычисляем процентное распределение меток

    norm_target = (
        data["target"]
        .value_counts(normalize=True)
        .mul(100)
        .rename("percent")
        .reset_index()
    )

    fig = plt.figure(figsize=(10, 7))
    # Строим столбчатую диаграмму
    ax = sns.barplot(x="target", y="percent", data=norm_target, palette="viridis")

    # Добавляем процентные значения к столбцам
    plot_text(ax)
    # Устанавливаем заголовок графика
    ax.set_title("Оценки пользователей", fontsize=14)
    # Устанавливаем подпись оси x
    ax.set_xlabel("Оценки пользователей", fontsize=12)
    # Устанавливаем подпись оси y
    ax.set_ylabel("Доля в процентах", fontsize=12)

    return fig


def plot_boxplot(
    data, col_name: str, hue_name: str, plot_name: str, width: int = 4, height: int = 4
) -> matplotlib.figure.Figure:
    """
    Построение бокплота
    :param data: датафрейм
    :param col_name:
    :param hue_name:
    :param plot_name:
    :param width:
    :param height:
    :return: fig
    """

    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(width, height))
    ax = sns.boxplot(data=data, y=col_name, hue=hue_name, palette="magma")
    ax.set_title(plot_name)

    return fig


def plot_kdeplot(
    data, col_name, hue_name: str, plot_name: str, width: int = 4, height: int = 4
) -> matplotlib.figure.Figure:
    """
    Построение kdeplot
    :param data:
    :param col_name:
    :param hue_name:
    :param plot_name:
    :param width:
    :param height:
    :return: fig
    """

    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(width, height))
    ax = sns.kdeplot(
        {
            "target 0": data[data[hue_name] == 0][col_name],
            "target 1": data[data[hue_name] == 1][col_name],
        },
        common_norm=False,
        palette="viridis",
        fill=True,
    )

    ax.set_xlabel(col_name)
    ax.set_title(plot_name)

    return fig


def plotting_trainer_stats(config: str) -> matplotlib.figure.Figure:
    """
    Построение графиков
    :param config: конфигурационный файл
    :return:
    """
    with open(config["trainer_log_path"], "r") as json_file:
        steps = json.load(json_file)

    auc = []
    eval_loss = []

    for step in steps:
        try:
            auc.append(step["eval_roc_auc"])
            eval_loss.append(step["eval_loss"])

        except KeyError:
            continue

    fig, axes = plt.subplots(ncols=2, figsize=(10, 5))
    sns.lineplot(eval_loss, ax=axes[0], color="orange")
    sns.lineplot(auc, ax=axes[1])

    axes[0].set_title("Validation Loss")
    axes[1].set_title("ROC-AUC SCORE")
    axes[0].set(xlabel="Epochs")
    axes[1].set(xlabel="Epochs")

    return fig


def plot_barplot(data, y, x):

    fig = plt.figure(figsize=(8, 6))
    ax = sns.barplot(y=y, x=x, orient="h", palette="rocket")

    return fig


def create_sentiment_plot(
    probabilities,
    width=500,
    height=400,
    colors=None,
    title=None,
    xaxis=None,
    yaxis=None,
) -> matplotlib.figure.Figure:
    """
    Создает график вероятностей сентиментов.

    Args:
    probabilities (dict): Словарь с метками сентиментов и их вероятностями.
    width (int): Ширина графика в пикселях. По умолчанию 800.
    height (int): Высота графика в пикселях. По умолчанию 600.
    colors (list): Список цветов для каждого столбца. По умолчанию None.

    Returns:
    plotly.graph_objects.Figure: Объект Figure для отображения графика.
    """
    if colors is None:
        colors = ["#FF6666", "#33D5E1"]  # Стандартные цвета

    # Создание графика
    fig = go.Figure(
        data=[
            go.Bar(
                x=list(probabilities.keys()),
                y=list(probabilities.values()),
                marker=dict(color=colors),
            )
        ]
    )

    # Настройка размеров и других параметров графика
    fig.update_layout(
        title={"text": title, "x": 0.5, "xanchor": "center"},
        xaxis_title=xaxis,
        yaxis_title=yaxis,
        width=width,  # ширина графика в пикселях
        height=height,  # высота графика в пикселях
    )

    return fig


def get_sentiment_emoji(label, size):
    if label == "Positive":
        return f'<span style="font-size: {size}px;">😊</span>'  # Веселый эмодзи
    elif label == "Negative":
        return f'<span style="font-size: {size}px;">😔</span>'  # Грустный эмодзи
    else:
        return ""
