#!/usr/bin/env python
# coding: utf-8

# # Задание 1 (2 балла)

# Напишите класс `MyDict`, который будет полностью повторять поведение обычного словаря, за исключением того, что при итерации мы должны получать и ключи, и значения.
# 
# **Модули использовать нельзя**


class MyDict(dict):
    
    def __init__(self, data):
        super().__init__(data)
        self.__current_key_index = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.__current_key_index == len(self.keys()):
            raise StopIteration
        next_key = list(self.keys())[self.__current_key_index]
        next_item = self[next_key]
        self.__current_key_index += 1
        return next_key, next_item



dct = MyDict({"a": 1, "b": 2, "c": 3, "d": 25})
dct


dct = MyDict({"a": 1, "b": 2, "c": 3, "d": 25})
dct['f'] = 7
for key, value in dct:
    print(key, value) 


dct = MyDict({"a": 1, "b": 2, "c": 3, "d": 25})
for key, value in dct:
    print(key, value)   


for key, value in dct.items():
    print(key, value)


for key in dct.keys():
    print(key)


dct["c"] + dct["d"]


# # Задание 2 (2 балла)

# Напишите функцию `iter_append`, которая "добавляет" новый элемент в конец итератора, возвращая итератор, который включает изначальные элементы и новый элемент. Итерироваться по итератору внутри функции нельзя, то есть вот такая штука не принимается
# ```python
# def iter_append(iterator, item):
#     lst = list(iterator) + [item]
#     return iter(lst)
# ```
# 
# **Модули использовать нельзя**


def iter_append(iterator, item):
    try:
        while True:
            yield next(iterator)
    except:
        yield item
    
    
my_iterator = iter([1, 2, 3])
new_iterator = iter_append(my_iterator, 4)

for element in new_iterator:
    print(element)


# # Задание 3 (5 баллов)

# Представим, что мы установили себе некоторую библиотеку, которая содержит в себе два класса `MyString` и `MySet`, которые являются наследниками `str` и `set`, но также несут и дополнительные методы.
# 
# Проблема заключается в том, что библиотеку писали не очень аккуратные люди, поэтому получилось так, что некоторые методы возвращают не тот тип данных, который мы ожидаем. Например, `MyString().reverse()` возвращает объект класса `str`, хотя логичнее было бы ожидать объект класса `MyString`.
# 
# Найдите и реализуйте удобный способ сделать так, чтобы подобные методы возвращали экземпляр текущего класса, а не родительского. При этом **код методов изменять нельзя**
# 
# **+3 дополнительных балла** за реализацию того, чтобы **унаследованные от `str` и `set` методы** также возвращали объект интересующего нас класса (то есть `MyString.replace(..., ...)` должен возвращать `MyString`). **Переопределять методы нельзя**
# 
# **Модули использовать нельзя**

# #### Изменяю возвращаемый тип декораторами методов: 


def make_class_mystring(func):
    def inner_function(*args, **kwargs):
        result_inter = func(*args, **kwargs)
        if type(result_inter) != str:
            return result_inter
        result = MyString(result_inter)
        return result
    return inner_function


def make_class_myset(func):
    def inner_function(*args, **kwargs):
        result_inter = func(*args, **kwargs)
        if type(result_inter) != set:
            return result_inter
        result = MySet(result_inter)
        return result
    return inner_function


class MyString(str):

    @make_class_mystring
    def reverse(self):
        return self[::-1]
    
    @make_class_mystring
    def make_uppercase(self):
        return "".join([chr(ord(char) - 32) if 97 <= ord(char) <= 122 else char for char in self])
    
    @make_class_mystring
    def make_lowercase(self):
        return "".join([chr(ord(char) + 32) if 65 <= ord(char) <= 90 else char for char in self])
    
    @make_class_mystring
    def capitalize_words(self):
        return " ".join([word.capitalize() for word in self.split()])
    
    
class MySet(set):
    @make_class_myset
    def is_empty(self):
        return len(self) == 0
    
    def has_duplicates(self):
        return len(self) != len(set(self))
    
    @make_class_myset
    def union_with(self, other):
        return self.union(other)
    
    @make_class_myset
    def intersection_with(self, other):
        return self.intersection(other)
    
    @make_class_myset
    def difference_with(self, other):
        return self.difference(other)



string_example = MyString("Aa Bb Cc")
set_example_1 = MySet({1, 2, 3, 4})
set_example_2 = MySet({3, 4, 5, 6, 6})

print(type(string_example.reverse()))
print(type(string_example.make_uppercase()))
print(type(string_example.make_lowercase()))
print(type(string_example.capitalize_words()))
print()
print(type(set_example_1.is_empty()))
print(type(set_example_2.has_duplicates()))
print(type(set_example_1.union_with(set_example_2)))
print(type(set_example_1.difference_with(set_example_2)))


# #### Изменяю возвращаемый тип декораторами классов:


def change_return_type(cls):
    fixed_attr_dict = cls.__dict__.copy()
    if cls.__name__ == "MyString":
        addition_dict = str.__dict__.copy()
    if cls.__name__ == "MySet":
        addition_dict = set.__dict__.copy()
    
    for attribute_name in addition_dict:
        if not attribute_name.startswith("_"):
            func = addition_dict[attribute_name]
            if cls.__name__ == "MyString":
                result = make_class_mystring(func)
            elif cls.__name__ == "MySet":
                result = make_class_myset(func)
            else:
                result = func
            setattr(cls, attribute_name, result)
    
    for attribute_name in fixed_attr_dict:
        if not attribute_name.startswith("_"):
            func = fixed_attr_dict[attribute_name]
            if cls.__name__ == "MyString":
                result = make_class_mystring(func)
            elif cls.__name__ == "MySet":
                result = make_class_myset(func)
            else:
                result = func
            setattr(cls, attribute_name, result)
    return cls



@change_return_type
class MyString(str):

    def reverse(self):
        return self[::-1]
    
    def make_uppercase(self):
        return "".join([chr(ord(char) - 32) if 97 <= ord(char) <= 122 else char for char in self])
    
    def make_lowercase(self):
        return "".join([chr(ord(char) + 32) if 65 <= ord(char) <= 90 else char for char in self])
    
    def capitalize_words(self):
        return " ".join([word.capitalize() for word in self.split()])
    
    
@change_return_type
class MySet(set):
    
    def is_empty(self):
        return len(self) == 0
    
    def has_duplicates(self):
        return len(self) != len(set(self))
    
    def union_with(self, other):
        return self.union(other)
    
    def intersection_with(self, other):
        return self.intersection(other)
    
    def difference_with(self, other):
        return self.difference(other)



string_example = MyString("Aa Bb Cc")
set_example_1 = MySet({1, 2, 3, 4})
set_example_2 = MySet({3, 4, 5, 6, 6})

print(type(string_example.reverse()))
print(type(string_example.make_uppercase()))
print(type(string_example.make_lowercase()))
print(type(string_example.capitalize_words()))
print()
print(type(set_example_1.is_empty()))
print(type(set_example_2.has_duplicates()))
print(type(set_example_1.union_with(set_example_2)))
print(type(set_example_1.difference_with(set_example_2)))
print()
print(type(string_example.replace("A", "M")))


# # Задание 4 (5 баллов)

# Напишите декоратор `switch_privacy`:
# 1. Делает все публичные **методы** класса приватными
# 2. Делает все приватные методы класса публичными
# 3. Dunder методы и защищённые методы остаются без изменений
# 4. Должен работать тестовый код ниже, в теле класса писать код нельзя
# 
# **Модули использовать нельзя**


# Ваш код здесь

def method_swither(cls):
    fixed_attr_dict = cls.__dict__.copy()
    for attribute_name in fixed_attr_dict:
        if attribute_name.startswith(f"_{cls.__name__}__"):
            # private
            new_name = attribute_name.replace(f"_{cls.__name__}__", "")
            func = fixed_attr_dict[attribute_name]
            setattr(cls, new_name, func)
            delattr(cls, attribute_name)
        if not attribute_name.startswith("_"):
            # public
            new_name = f"_{cls.__name__}__{attribute_name}"
            func = fixed_attr_dict[attribute_name]
            setattr(cls, new_name, func)
            delattr(cls, attribute_name)
    return cls
    

@method_swither
class ExampleClass:
    # Но не здесь
    def public_method(self):
        return 1
    
    def _protected_method(self):
        return 2
    
    def __private_method(self):
        return 3
    
    def __dunder_method__(self):
        pass


test_object = ExampleClass()

test_object._ExampleClass__public_method()   # Публичный метод стал приватным

test_object.private_method()   # Приватный метод стал публичным

test_object._protected_method()   # Защищённый метод остался защищённым

test_object.__dunder_method__()   # Дандер метод не изменился

hasattr(test_object, "public_method"), hasattr(test_object, "private")   # Изначальные варианты изменённых методов не сохраняются


# # Задание 5 (7 баллов)

# Напишите [контекстный менеджер](https://docs.python.org/3/library/stdtypes.html#context-manager-types) `OpenFasta`
# 
# Контекстные менеджеры это специальные объекты, которые могут работать с конструкцией `with ... as ...:`. В них нет ничего сложного, для их реализации как обычно нужно только определить только пару dunder методов. Изучите этот вопрос самостоятельно
# 
# 1. Объект должен работать как обычные файлы в питоне (наследоваться не надо, здесь лучше будет использовать **композицию**), но:
#     + При итерации по объекту мы должны будем получать не строку из файла, а специальный объект `FastaRecord`. Он будет хранить в себе информацию о последовательности. Важно, **не строки, а именно последовательности**, в fasta файлах последовательность часто разбивают на много строк
#     + Нужно написать методы `read_record` и `read_records`, которые по смыслу соответствуют `readline()` и `readlines()` в обычных файлах, но они должны выдавать не строки, а объект(ы) `FastaRecord`
# 2. Конструктор должен принимать один аргумент - **путь к файлу**
# 3. Класс должен эффективно распоряжаться памятью, с расчётом на работу с очень большими файлами
#     
# Объект `FastaRecord`. Это должен быть **датакласс** (см. про примеры декораторов в соответствующей лекции) с тремя полями:
# + `seq` - последовательность
# + `id_` - ID последовательности (это то, что в фаста файле в строке, которая начинается с `>` до первого пробела. Например, >**GTD326487.1** Species anonymous 24 chromosome) 
# + `description` - то, что осталось после ID (Например, >GTD326487.1 **Species anonymous 24 chromosome**)
# 
# 
# Напишите демонстрацию работы кода с использованием всех написанных методов, обязательно добавьте файл с тестовыми данными в репозиторий (не обязательно большой)
# 
# **Можно использовать модули из стандартной библиотеки**

# In[19]:


from dataclasses import dataclass

@dataclass
class FastaRecord:
    seq: str
    id_: str
    description: str
    


class OpenFasta:
    
    def __init__(self, filename, mode = 'r'):
        self.filename = filename
        self.mode = mode
        self.__first_line_symb = ""
    
    
    def __enter__(self):
        self.__file = open(self.filename, self.mode)
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        if not self.__file.closed:
            self.__file.close()
        return False
    
    # readline()
    def read_record(self):
        first_line = self.__file.readline()  # читаем первую строку с id и описанием
        if first_line == "":  # прочитали пустую строку: конец файла или пустой файл
            return None
        array = first_line.split(' ')
        id_, description = array[0], " ".join(array[1:])[:-1]
        if self.__first_line_symb != ">":  # учитываем случай: когда ">" с новой строки мы уже прочитали
            id_ = id_[1:]
        # id и description прочитали
        
        # здесь начинаем читать sequence
        self.__first_line_symb = self.__file.read(1)
        seq = ""
        while (self.__first_line_symb != ""):  # пока не дошли до конца файла – первый символ в строке = пустая строка
            if self.__first_line_symb == ">":  # зашли на следующую запись: возвращаем FastaRecord
                return FastaRecord(seq=seq, id_=id_, description=description)
            # не дошли до конца - спокойно можем читать последовательность до конца строки
            seq += self.__first_line_symb
            seq += self.__file.readline()[:-1]
            self.__first_line_symb = self.__file.read(1)
        return FastaRecord(seq=seq, id_=id_, description=description)
    
    # readlines()
    def read_records(self):
        line = self.read_record()
        lines = []
        while line != None:
            lines.append(line)
            line = fasta.read_record()
        return lines



# Ваш код здесь
import os

with OpenFasta(os.path.join("data", "Parapallasea_18.fa")) as fasta:
    line = fasta.read_record()
    while line != None:
        print(line)
        line = fasta.read_record()



lines = []
with OpenFasta(os.path.join("data", "Parapallasea_18.fa")) as fasta:
    # Ваш код здесь
    lines = fasta.read_records()


lines


# # Задание 6 (7 баллов)

# 1. Напишите код, который позволит получать все возможные (неуникальные) генотипы при скрещивании двух организмов. Это может быть функция или класс, что вам кажется более удобным.
# 
# Например, все возможные исходы скрещивания "Aabb" и "Aabb" (неуникальные) это
# 
# ```
# AAbb
# AAbb
# AAbb
# AAbb
# Aabb
# Aabb
# Aabb
# Aabb
# Aabb
# Aabb
# Aabb
# Aabb
# aabb
# aabb
# aabb
# aabb
# ```
# 
# 2. Напишите функцию, которая вычисляет вероятность появления определённого генотипа (его ожидаемую долю в потомстве).
# Например,
# 
# ```python
# get_offspting_genotype_probability(parent1="Aabb", parent2="Aabb", target_genotype="Aabb")   # 0.5
# 
# ```
# 
# 3. Напишите код, который выводит все уникальные генотипы при скрещивании `'АаБбввГгДдЕеЖжЗзИиЙйккЛлМмНн'` и `'АаббВвГгДДЕеЖжЗзИиЙйКкЛлМмНН'`, которые содержат в себе следующую комбинацию аллелей `'АаБбВвГгДдЕеЖжЗзИиЙйКкЛл'`
# 4. Напишите код, который расчитывает вероятность появления генотипа `'АаБбввГгДдЕеЖжЗзИиЙйккЛлМмНн'` при скрещивании `АаБбВвГгДдЕеЖжЗзИиЙйКкЛлМмНн` и `АаБбВвГгДдЕеЖжЗзИиЙйКкЛлМмНн`
# 
# Важные замечания:
# 1. Порядок следования аллелей в случае гетерозигот всегда должен быть следующим: сначала большая буква, затем маленькая (вариант `AaBb` допустим, но `aAbB` быть не должно)
# 2. Подзадачи 3 и 4 могут потребовать много вычислительного времени (до 15+ минут в зависимости от железа), поэтому убедитесь, что вы хорошо протестировали написанный вами код на малых данных перед выполнением этих задач. Если ваш код работает **дольше 20 мин**, то скорее всего ваше решение не оптимально, попытайтесь что-нибудь оптимизировать. Если оптимальное решение совсем не получается, то попробуйте из входных данных во всех заданиях убрать последний ген (это должно уменьшить время выполнения примерно в 4 раза), но **за такое решение будет снято 2 балла**
# 3. Несмотря на то, что подзадания 2, 3 и 4 возможно решить математически, не прибегая к непосредственному получению всех возможных генотипов, от вас требуется именно brute-force вариант алгоритма
# 
# **Можно использовать модули из стандартной библиотеки питона**, но **за выполнение задания без использования модулей придусмотрено +3 дополнительных балла**



# Ваш код здесь (1 и 2 подзадание)

import itertools


# Функция, меняющая буквы местами в tuple гетерозигот (чтобы было Aa, но не было aA)
def upper_lower_change(result):

    result_list = []
    cnt = 0

    for i in result:
        i = list(i)
        result_list.append(i)
        if (i[0].islower() and i[1].isupper()):
            result_list[cnt][0], result_list[cnt][1] = result_list[cnt][1], result_list[cnt][0]
        cnt += 1

    return result_list


# Возможные (неуникальные) генотипы при скрещивании двух организмов
def all_combinations_of_genotypes(parent_1, parent_2, test_genotype = None, target_genotype = None):
    
    comb_dict = {}
    cnt = 0
    result_list = []
    
    for i in range(len(set(parent_1.lower()))):
        comb_dict[i] = upper_lower_change(list(itertools.product(parent_1[cnt:cnt+2], parent_2[cnt:cnt+2])))
        cnt += 2
    
            
    result = itertools.product(*list(comb_dict.values()))
    
    target_count = 0
    total_count = 0
    for i in result:
        res = tuple(itertools.chain(*i))
        res_string = ''.join(res)
        if target_genotype:
            total_count += 1
            if target_genotype == res_string:
                target_count += 1
        else:
            if test_genotype:
                if test_genotype in res_string:
                    result_list.append(res_string)
            else:
                result_list.append(res_string)
    
    if target_genotype:
        return target_count, total_count
    return result_list


# Вычисление вероятности появления определённого генотипа
def get_offspting_genotype_probability(parent_1, parent_2, target_genotype):
    
    target_count, total_count = all_combinations_of_genotypes(parent_1, parent_2, target_genotype = target_genotype)
    result = target_count / total_count
    return result


all_combinations_3 = all_combinations_of_genotypes(parent_1="AabbCc", parent_2="Aabbcc")


get_offspting_genotype_probability(parent_1="Aabb", parent_2="Aabb", target_genotype="Aabb")


# Ваш код здесь (3 подзадание)

all_combinations_14 = all_combinations_of_genotypes(parent_1 = 'АаБбввГгДдЕеЖжЗзИиЙйккЛлМмНн', parent_2 = 'АаббВвГгДДЕеЖжЗзИиЙйКкЛлМмНН', 
                                                    test_genotype = 'АаБбВвГгДдЕеЖжЗзИиЙйКкЛл')


# Ваш код здесь (4 подзадание)

get_offspting_genotype_probability(parent_1="АаБбВвГгДдЕеЖжЗзИиЙйКкЛлМмНн", 
                                   parent_2="АаБбВвГгДдЕеЖжЗзИиЙйКкЛлМмНн", 
                                   target_genotype="АаБбввГгДдЕеЖжЗзИиЙйккЛлМмНн")




