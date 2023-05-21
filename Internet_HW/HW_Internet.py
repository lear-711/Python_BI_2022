#!/usr/bin/env python
# coding: utf-8

# # Задание 1 (6 баллов)

# В данном задании мы будем работать со [списком 250 лучших фильмов IMDb](https://www.imdb.com/chart/top/?ref_=nv_mp_mv250)
# 
# 1. Выведите топ-4 *фильма* **по количеству оценок пользователей** и **количество этих оценок** (1 балл)
# 2. Выведите топ-4 лучших *года* (**по среднему рейтингу фильмов в этом году**) и **средний рейтинг** (1 балл)
# 3. Постройте отсортированный **barplot**, где показано **количество фильмов** из списка **для каждого режисёра** (только для режиссёров с более чем 2 фильмами в списке) (1 балл)
# 4. Выведите топ-4 самых популярных *режиссёра* (**по общему числу людей оценивших их фильмы**) (2 балла)
# 5. Сохраните данные по всем 250 фильмам в виде таблицы с колонками (name, rank, year, rating, n_reviews, director) в любом формате (2 балла)
# 
# Использовать можно что-угодно, но полученные данные должны быть +- актуальными на момент сдачи задания


import requests
from bs4 import BeautifulSoup


# ### 1. Топ-4 фильма по количеству оценок пользователей и количество этих оценок:

# Сортируем фильмы по количеству оценок пользователей:


response_nv = requests.get("https://www.imdb.com/chart/top", params={"sort": "nv,desc", "mode": "simple"})
soup_nv = BeautifulSoup(response_nv.content, "lxml")


# Вытаскиваем необходимую информацию с фильмами и отзывами:

tbody = soup_nv.find_all("tbody", class_ = "lister-list")[0]


# Берем первые 4 фильма:

trs = tbody.find_all("tr")[:4]


for tr in trs:
    film_name = tr.find("td", class_ ="titleColumn").find("a").text
    number_of_ratings = int(tr.find("strong").attrs['title'].split("on")[-1].split("user")[0].replace(",", ""))
    print(f'Фильм: {film_name} - количество оценок = {number_of_ratings}')


# ### 2. Топ-4 лучших года (по среднему рейтингу фильмов в этом году) и средний рейтинг:

response = requests.get("https://www.imdb.com/chart/top/?ref_=nv_mp_mv250")
soup = BeautifulSoup(response.content, "lxml")

tbody = soup.find_all("tbody", class_ = "lister-list")[0]


# Создаем словарь с годами и рейтингами фильмов, вышедших в этот год:

year_all_rates_dict = {}

for i in range(250):
    year = int(tbody.find_all("tr")[i].find_all("span", class_ ="secondaryInfo")[0].text[1:-1])
    rate = float(tbody.find_all("tr")[i].find("strong").attrs['title'].split(" based ")[0])
    if year in year_all_rates_dict.keys():
        year_all_rates_dict[year].append(rate)
    else:
        year_all_rates_dict[year] = []
        year_all_rates_dict[year].append(rate)


# Создаем словарь с годами и средним рейтингом фильмов, вышедших в этот год:

import numpy as np

year_mean_rates_dict = {}

for key, value in year_all_rates_dict.items():
    mean_value = np.mean(value)
    year_mean_rates_dict[key] = mean_value


best_years = sorted(year_mean_rates_dict.items(), key=lambda x:x[1], reverse=True)[:4]
for year in best_years:
    print(f'Год: {year[0]}, средний рейтинг: {year[1]}')


# ### 3. Barplot с количеством фильмов из списка для каждого режисёра:

director = tbody.find_all("td", class_ ="titleColumn")[0].find("a").attrs["title"].split(' (dir.)')[0]


# Создаем словарь с режиссерами и количеством фильмов у них:

director_films_dict = {}

for i in range(250):
    director = tbody.find_all("td", class_ ="titleColumn")[i].find("a").attrs["title"].split(' (dir.)')[0]
    if director in director_films_dict.keys():
        director_films_dict[director] += 1
    else:
        director_films_dict[director] = 1


# Удаляем из словаря режиссеров с 1 или 2 фильмами:

keys_for_delete = []

for key, value in director_films_dict.items():
    if (value == 1) or (value == 2):
        keys_for_delete.append(key)


for key in keys_for_delete:
    del director_films_dict[key]


# Сортируем словарь:

sorted_director_dict = dict(sorted(director_films_dict.items(), key=lambda x:x[1]))


import matplotlib.pyplot as plt
import seaborn as sns

sns.set(rc={'figure.figsize':(8,6)})

keys = list(sorted_director_dict.keys())
vals = list(sorted_director_dict.values())
sns.barplot(x=keys, y=vals)

plt.xticks(rotation=90)
plt.tight_layout()


# ### 4. Топ-4 самых популярных режиссёра (по общему числу людей оценивших их фильмы):

# Создаем словарь с режиссерами и общим количеством людей их оценивших:


director_ratings = {}

for i in range(250):
    director = tbody.find_all("td", class_ ="titleColumn")[i].find("a").attrs["title"].split(' (dir.)')[0]
    number_of_ratings = int(tbody.find_all("tr")[i].find("strong").attrs['title'].split("on")[-1].split("user")[0].replace(",", ""))
    if director in director_ratings.keys():
        director_ratings[director] += number_of_ratings
    else:
        director_ratings[director] = number_of_ratings



sorted_director_rate_dict = sorted(director_ratings.items(), key=lambda x:x[1], reverse=True)


for direct in sorted_director_rate_dict[:4]:
    print(f'Режиссер: {direct[0]}, общее количество оценок: {direct[1]}')


# ### 5. Сохранение данных по всем 250 фильмам в виде таблицы

film_names = []
ranks = []
years = []
ratings = []
n_reviews = []
directors = []

for i in range(250):
    film_names.append(tbody.find_all("td", class_ ="titleColumn")[i].find("a").text)
    ranks.append(i + 1)
    years.append(int(tbody.find_all("tr")[i].find_all("span", class_ ="secondaryInfo")[0].text[1:-1]))
    ratings.append(float(tbody.find_all("tr")[i].find("strong").attrs['title'].split(" based ")[0]))
    n_reviews.append(int(tbody.find_all("tr")[i].find("strong").attrs['title'].split("on")[-1].split("user")[0].replace(",", "")))
    directors.append(tbody.find_all("td", class_ ="titleColumn")[i].find("a").attrs["title"].split(' (dir.)')[0])



import pandas as pd

top_250_table = pd.DataFrame(
    {
        "Film name": pd.Series(film_names),
        "Ranks": pd.Series(ranks),
        "Year": pd.Series(years),
        "Ratings": pd.Series(ratings),
        "Number of reviews": pd.Series(n_reviews),
        "Director": pd.Series(directors)
    }
)


top_250_table





# # Задание 2 (10 баллов)

# Напишите декоратор `telegram_logger`, который будет логировать запуски декорируемых функций и отправлять сообщения в телеграм.
# 
# 
# Вся информация про API телеграм ботов есть в официальной документации, начать изучение можно с [этой страницы](https://core.telegram.org/bots#how-do-bots-work) (разделы "How Do Bots Work?" и "How Do I Create a Bot?"), далее идите в [API reference](https://core.telegram.org/bots/api)
# 
# **Основной функционал:**
# 1. Декоратор должен принимать **один обязательный аргумент** &mdash; ваш **CHAT_ID** в телеграме. Как узнать свой **CHAT_ID** можно найти в интернете
# 2. В сообщении об успешно завершённой функции должны быть указаны её **имя** и **время выполнения**
# 3. В сообщении о функции, завершившейся с исключением, должно быть указано **имя функции**, **тип** и **текст ошибки**
# 4. Ключевые элементы сообщения должны быть выделены **как код** (см. скриншот), форматирование остальных элементов по вашему желанию
# 5. Время выполнения менее 1 дня отображается как `HH:MM:SS.μμμμμμ`, время выполнения более 1 дня как `DDD days, HH:MM:SS`. Писать форматирование самим не нужно, всё уже где-то сделано за вас
# 
# **Дополнительный функционал:**
# 1. К сообщению также должен быть прикреплён **файл**, содержащий всё, что декорируемая функция записывала в `stdout` и `stderr` во время выполнения. Имя файла это имя декорируемой функции с расширением `.log` (**+3 дополнительных балла**)
# 2. Реализовать предыдущий пункт, не создавая файлов на диске (**+2 дополнительных балла**)
# 3. Если функция ничего не печатает в `stdout` и `stderr` &mdash; отправлять файл не нужно
# 
# **Важные примечания:**
# 1. Ни в коем случае не храните свой API токен в коде и не загружайте его ни в каком виде свой в репозиторий. Сохраните его в **переменной окружения** `TG_API_TOKEN`, тогда его можно будет получить из кода при помощи `os.getenv("TG_API_TOKEN")`. Ручное создание переменных окружения может быть не очень удобным, поэтому можете воспользоваться функцией `load_dotenv` из модуля [dotenv](https://pypi.org/project/python-dotenv/). В доке всё написано, но если коротко, то нужно создать файл `.env` в текущей папке и записать туда `TG_API_TOKEN=<your_token>`, тогда вызов `load_dotenv()` создаст переменные окружения из всех переменных в файле. Это довольно часто используемый способ хранения ключей и прочих приватных данных
# 2. Функцию `long_lasting_function` из примера по понятным причинам запускать не нужно. Достаточно просто убедится, что большие временные интервалы правильно форматируются при отправке сообщения (как в примерах)
# 3. Допустима реализация логирования, когда логгер полностью перехватывает запись в `stdout` и `stderr` (то есть при выполнении функций печать происходит **только** в файл)
# 4. В реальной жизни вам не нужно использовать Telegram API при помощи ручных запросов, вместо этого стоит всегда использовать специальные библиотеки Python, реализующие Telegram API, они более высокоуровневые и удобные. В данном задании мы просто учимся работать с API при помощи написания велосипеда.
# 5. Обязательно прочтите часть конспекта лекции про API перед выполнением задания, так как мы довольно поверхностно затронули это на лекции
# 
# **Рекомендуемые к использованию модули:**
# 1. os
# 2. sys
# 3. io
# 4. datetime
# 5. requests
# 6. dotenv
# 
# **Запрещённые модули**:
# 1. Любые библиотеки, реализующие Telegram API в Python (*python-telegram-bot, Telethon, pyrogram, aiogram, telebot* и так далле...)
# 2. Библиотеки, занимающиеся "перехватыванием" данных из `stdout` и `stderr` (*pytest-capturelog, contextlib, logging*  и так далле...)
# 
# 
# 
# Результат запуска кода ниже должен быть примерно такой:
# 
# ![image.png](attachment:620850d6-6407-4e00-8e43-5f563803d7a5.png)
# 
# ![image.png](attachment:65271777-1100-44a5-bdd2-bcd19a6f50a5.png)
# 
# ![image.png](attachment:e423686d-5666-4d81-8890-41c3e7b53e43.png)

# In[27]:


import io
# следующие две ячейки откровенно говоря, танцы с бубном:
# чтобы перенаправить в вывод stdout и stderr в псевдо-файл и прикрутить к нему название файла <function>.log

class NamedBytesIO(io.BufferedReader):
    def __init__(self, buffer, name=None, **kwargs):
        vars(self)['name'] = name
        super().__init__(buffer, **kwargs)

    def __getattribute__(self, name):
        if name == 'name':
            return vars(self)['name']
        return super().__getattribute__(name)




class RedirectedStd:
    def __init__(self):
        self._stdout = None
        self._string_io = io.StringIO()
        self._stderr = None

    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._string_io
        self._stderr = sys.stderr
        sys.stderr = self._string_io
        return self

    def __exit__(self, type, value, traceback):
        sys.stdout = self._stdout
        sys.stderr = self._stderr

    def __str__(self):
        return self._string_io.getvalue()
    
    def get_file(self, function_name):
        data = self._string_io
        data = io.BytesIO(bytes(data.getvalue(), 'utf-8'))
        file = NamedBytesIO(data, f"{function_name}.log")
        return file




import sys
import time
import datetime
import requests
from dotenv import load_dotenv
import os

load_dotenv()

chat_id = "38847190"
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")


def send_result_to_telegram(chat_id, message, function_name, log_file):
    files = None
    restricted_tokens = ['.', '_', '!']
    for rt in restricted_tokens:
        message = message.replace(f"{rt}", f"\\{rt}")
    params = {
        'chat_id': chat_id,
        'parse_mode': "MarkdownV2",
    }
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/"
    if len(str(log_file)):
        # есть логи, надо использовать отправку документа
        url += "sendDocument"
        params['caption'] = message
        log_file = log_file.get_file(function_name)
        files = {'document': log_file}
    else:
        # логов нет, используем обычную отправку
        url += "sendMessage"
        params['text'] = message
    res = requests.get(url, params, files=files)


def telegram_logger(chat_id):
    # Ваш код здесь
    def wrapper(func):
        def inner(*args, **kwargs):
            function_name = func.__name__
            telegram_msg = f"Function `{function_name}` "
            
            try:
                with RedirectedStd() as my_std:
                    start = time.time()
                    result = func(*args, **kwargs)
                    end = time.time()
                telegram_msg += "successfully finished in: "
                time_delta = datetime.datetime.fromtimestamp(end) - datetime.datetime.fromtimestamp(start)
                time_str = str(time_delta)
                if time_delta.days:
                    time_str = time_str.split(".")[0]
                telegram_msg += time_str
                send_result_to_telegram(chat_id, f"🎉 {telegram_msg}", function_name, my_std)
            except Exception as e:
                telegram_msg += f"failed with an exception:\n\n{type(e).__name__}: {e}"
                send_result_to_telegram(chat_id, f"☹️ {telegram_msg}", function_name, my_std)
                raise e
            return result
        return inner
    return wrapper





@telegram_logger(chat_id)
def good_function():
    print("This goes to stdout")
    print("And this goes to stderr", file=sys.stderr)
    time.sleep(2)
    print("Wake up, Neo")

@telegram_logger(chat_id)
def bad_function():
    print("Some text to stdout")
    time.sleep(2)
    print("Some text to stderr", file=sys.stderr)
    raise RuntimeError("Ooops, exception here!")
    print("This text follows exception and should not appear in logs")
    
@telegram_logger(chat_id)
def long_lasting_function():
    time.sleep(200000000)



good_function()

try:
    bad_function()
    print("_____")
except Exception:
    pass




# long_lasting_function()


# # Задание 3
# 
# В данном задании от вас потребуется сделать Python API для какого-либо сервиса
# 
# В задании предложено два варианта: простой и сложный, **выберите только один** из них.
# 
# Можно использовать только **модули стандартной библиотеки** и **requests**. Любые другие модули можно по согласованию с преподавателем.

# ❗❗❗ В **данном задании** требуется оформить код в виде отдельного модуля (как будто вы пишете свою библиотеку). Код в ноутбуке проверяться не будет ❗❗❗

# ## Вариант 1 (простой, 10 баллов)
# 
# В данном задании вам потребуется сделать Python API для сервиса http://hollywood.mit.edu/GENSCAN.html
# 
# Он способен находить и вырезать интроны в переданной нуклеотидной последовательности. Делает он это не очень хорошо, но это лучше, чем ничего. К тому же у него действительно нет публичного API.
# 
# Реализуйте следующую функцию:
# `run_genscan(sequence=None, sequence_file=None, organism="Vertebrate", exon_cutoff=1.00, sequence_name="")` &mdash; выполняет запрос аналогичный заполнению формы на сайте. Принимает на вход все параметры, которые можно указать на сайте (кроме Print options). `sequence` &mdash; последовательность в виде строки или любого удобного вам типа данных, `sequence_file` &mdash; путь к файлу с последовательностью, который может быть загружен и использован вместо `sequence`. Функция должна будет возвращать объект типа `GenscanOutput`. Про него дальше.
# 
# Реализуйте **датакласс** `GenscanOutput`, у него должны быть следующие поля:
# + `status` &mdash; статус запроса
# + `cds_list` &mdash; список предсказанных белковых последовательностей с учётом сплайсинга (в самом конце результатов с сайта)
# + `intron_list` &mdash; список найденных интронов. Один интрон можно представить любым типом данных, но он должен хранить информацию о его порядковом номере, его начале и конце. Информацию о интронах можно получить из первой таблицы в результатах на сайте.
# + `exon_list` &mdash; всё аналогично интронам, но только с экзонами.
# 
# По желанию можно добавить любые данные, которые вы найдёте в результатах

# In[ ]:


# Не пиши код здесь, сделай отдельный модуль


# Демонстрация работы модуля:


from Gen_scan import GenscanOutput

genscan = GenscanOutput()

genscan.run_genscan(sequence_file = '../../sequence.fasta')


genscan.status

genscan.cds_list[0]


# ## Вариант 2 (очень сложный, 20 дополнительных баллов)

# В этом варианте от вас потребуется сделать Python API для BLAST, а именно для конкретной вариации **tblastn** https://blast.ncbi.nlm.nih.gov/Blast.cgi?PROGRAM=tblastn&PAGE_TYPE=BlastSearch&LINK_LOC=blasthome
# 
# Хоть у BLAST и есть десктопное приложение, всё-таки есть одна область, где API может быть полезен. Если мы хотим искать последовательность в полногеномных сборках (WGS), а не в базах данных отдельных генов, у нас могут возникнуть проблемы. Так как если мы хотим пробластить нашу последовательность против большого количества геномов нам пришлось бы или вручную отправлять запросы на сайте, или скачивать все геномы и делать поиск локально. И тот и другой способы не очень удобны, поэтому круто было бы иметь способ сделать автоматический запрос, не заходя в браузер.
# 
# Необходимо написать функцию для запроса, которая будет принимать 3 обязательных аргумента: **белковая последовательность**, которую мы бластим, **базу данных** (в этом задании нас интересует только WGS, но по желанию можете добавить какую-нибудь ещё), **таксон**, у которого мы ищем последовательность, чаще всего &mdash; конкретный вид. По=желанию можете добавить также любые другие аргументы, соответствующие различным настройкам поиска на сайте. 
# 
# Функция дожна возвращать список объектов типа `Alignment`, у него должны быть следующие атрибуты (всё согласно результатам в браузере, удобно посмотреть на рисунке ниже), можно добавить что-нибудь своё:
# 
# ![Alignment.png](attachment:e45d0969-ff95-4d4b-8bbc-7f5e481dcda3.png)
# 
# 
# Самое сложное в задании - правильно сделать запрос. Для этого нужно очень глубоко погрузиться в то, что происходит при отправке запроса при помощи инструмента для разработчиков. Ещё одна проблема заключается в том, что BLAST не отдаёт результаты сразу, какое-то время ваш запрос обрабатывается, при этом изначальный запрос не перекидывает вас на страницу с результатами. Задание не такое простое как кажется из описания!

# In[ ]:


# Не пиши код здесь, сделай отдельный модуль

