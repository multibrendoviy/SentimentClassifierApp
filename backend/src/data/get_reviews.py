"""
Программа получения отзывов
Версия 1.0
"""
import json
import re

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm.auto import tqdm
import time
import numpy as np
import yaml
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def get_request_params(url: str) -> tuple:
    """
    Получение тела URL-запроса и номера текущей страницы
    :param url: URL-запрос
    :return: base_url: тело URL-запроса
    :return: num_page: номер страницы
    """
    pattern_url = r"^(.*)/\d+/$"
    pattern_num = r"/(\d+)/$"
    match = re.search(pattern_url, url)

    if match:
        base_url = match.group(1)
    else:
        raise ValueError()
    match = re.search(pattern_num, url)

    if match:
        num_page = match.group(1)
    else:
        num_page = 1

    return base_url, num_page


def get_session() -> requests.session:
    """
    Создание сессии
    :return: session
    """
    session = requests.Session()
    retry = Retry(connect=3, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session


def get_reviews(config_path: str, url: str, page_count: int = 2) -> None:
    """
    Получение отзывов с сайта отзовик
    :param: config_path: путь к конфигурационному файлу
    :param: url: URL-запрос
    :param: page_count: количество страниц для парсинга
    """

    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    preprocessing_params = config["preprocessing"]

    data_path = preprocessing_params["scrape_path"]

    with open(preprocessing_params['connection_path'], 'r') as json_file:
        params = json.load(json_file)

        cookies = params['cookies']
        headers = params['headers']
        HOST = params['HOST']

    data = []
    base_url, page_num = get_request_params(url)

    for i in tqdm(range(int(page_count))):

        url = f"{base_url}/{int(page_num) + i}/"
        session = get_session()
        response = session.get(url, headers=headers, cookies=cookies)

        soup = BeautifulSoup(response.text, 'html.parser')
        href_list = soup.find_all("a", class_="review-btn review-read-link")

        for j, link in enumerate(href_list):

            pause = np.random.uniform(0, 2)
            time.sleep(pause)

            # Проход циклом по всем ссылкам, полученным со страницы
            href = str(HOST + link["href"])
            response = session.get(href, cookies=cookies, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')

            # Получение текста отзыва
            try:
                review_text = soup.find("div", class_="review-body description").getText()
                data.append(review_text)

            except AttributeError:
                continue

    df = pd.Series(data, name='reviewText')
    df.to_csv(data_path, index=False)
