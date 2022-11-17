
import re
import requests
import seaborn as sns
import pandas as pd


# # 1

# ### Извлекаем из файла ftp ссылки и записываем их в файл ftps

response = requests.get("https://raw.githubusercontent.com/Serfentum/bf_course/master/15.re/references")
data = response.text


pattern = r"ftp\.[^;\s]*"
string = data

result = re.findall(pattern, string)
result = list(sorted(set(result)))
for i in result:
    print(i)


with open("ftps", "w") as ftp_file:
    for line in result:
        ftp_file.write(f"{line}\n")


# # 2

# ### Извлекаем из рассказа все числа

story_link = requests.get("https://raw.githubusercontent.com/Serfentum/bf_course/master/15.re/2430AD")
story = story_link.text


pattern = r"\d+\.*\d+"
string = story

result_story = re.findall(pattern, string)

print(result_story)


# # 3

# ### Извлекаем из рассказа все слова, где есть буква "а" (регистр не важен)

pattern = r"\w*[aA]\w*"
string = story

result_story_a = re.findall(pattern, string)

print(result_story_a)


# # 4

# ### Извлекаем из рассказа восклицательные предложения

pattern = r"[^.\"]*!"
string = story

result_story_exc = re.findall(pattern, string)

print(result_story_exc)


# # 5

pattern = r"\w+'\w+|\w+"
string = story

result_story_hist = re.findall(pattern, string)
print(result_story_hist)


# ### Отбираем только уникальные слова и числа

new_result = []
for i in result_story_hist:
    i = i.lower()
    new_result.append(i)
new_result = set(new_result)


# ### Делаем массив с длинами уникальных слов

len_result = []
for i in new_result:
    len_i = len(i)
    len_result.append(len_i)

#print(new_result)
#print(len_result)


set_len = set(len_result)
list_len = []
for i in set_len:
    cnt = len_result.count(i)
    list_len.append(cnt)


sum_len = sum(list_len)
part_list = []

for i in list_len:
    part = i/sum_len
    part = round(part, 3)
    part_list.append(part)


# ### Создаем датафрейм с длинами слов, их количеством и долей

len_df = pd.DataFrame(list(zip(list(set_len), list_len, part_list)), columns = ['Word length', 'Values', 'Part'])
len_df


# ### Строим barplot 

sns.barplot(data = len_df, x = 'Word length', y = 'Part')


# # 6

# ### Функция для перевода с русского языка на "кирпичный" язык

def rus_to_brick(string):
    
    # Отбираем слова из текста
    pattern_1 = r"\w+"
    string_1 = 'Введите строку на русском языке для перевода'
    result_translate = re.findall(pattern_1, string_1)
    
    # Функция для наждения значения по ключу (значение match_obj) в объявленном словаре
    def my_match(match_obj):
        dict_translate = {
        'а':'КА',
        'у':'КУ',
        'о':'КО',
        'е':'КЕ',
        'и':'КИ',
        'я':'КЯ',
        'ю':'КЮ',
        'ё':'КЁ',
        'э':'КЭ',
        'ы':'КЫ',
        }
        return match_obj.group(0) + dict_translate.get(match_obj.group(0), "")
    
    # Отбираем гласные буквы в словах и ставим после них значения словаря по ключу
    pattern_2 = r"([ауоеияюёэы])"
    result_vowel = []

    for s in result_translate:
        vowel = re.sub(pattern_2, my_match, s)
        result_vowel.append(vowel)
    
    # Записываем результат в строку
    string_result = ''
    for w in result_vowel:
        string_result += w
        string_result += ' '
        
    return string_result



# # 7

# ### Находим в тексте предложения с заданным количеством слов

def extract_n_words(string, count):
    
    pattern = r"[^.]*."
    string = string
    result_1 = re.findall(pattern, string)

    result = ()
    pattern_2 = r"\w+"
    for res in result_1:
        result_2 = re.findall(pattern_2, res)
        if len(result_2) == n:
            result = result + (result_2,)

    return result

