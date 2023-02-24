#!/usr/bin/env python
# coding: utf-8

# # Задание 1 (5 баллов)

# Напишите классы **Chat**, **Message** и **User**. Они должны соответствовать следующим требованиям:
# 
# **Chat**:
# + Должен иметь атрибут `chat_history`, где будут храниться все сообщения (`Message`) в обратном хронологическом порядке (сначала новые, затем старые)
# + Должен иметь метод `show_last_message`, выводящий на экран информацию о последнем сообщении
# + Должен иметь метод `get_history_from_time_period`, который принимает два опциональных аргумента (даты с которой и по какую мы ищем сообщения и выдаём их). Метод также должен возвращать объект типа `Chat`
# + Должен иметь метод `show_chat`, выводящий на экран все сообщения (каждое сообщение в таком же виде как и `show_last_message`, но с разделителем между ними)
# + Должен иметь метод `recieve`, который будет принимать сообщение и добавлять его в чат
# 
# **Message**:
# + Должен иметь три обязательных атрибута
#     + `text` - текст сообщения
#     + `datetime` - дата и время сообщения (встроенный модуль datetime вам в помощь). Важно! Это должна быть не дата создания сообщения, а дата его попадания в чат! 
#     + `user` - информация о пользователе, который оставил сообщение (какой тип данных использовать здесь, разберётесь сами)
# + Должен иметь метод `show`, который печатает или возвращает информацию о сообщении с необходимой информацией (дата, время, юзер, текст)
# + Должен иметь метод `send`, который будет отправлять сообщение в чат
# 
# **User**:
# + Класс с информацией о юзере, наполнение для этого класса придумайте сами
# 
# Напишите несколько примеров использования кода, которое показывает взаимодействие между объектами.
# 
# В тексте задания намерено не указано, какие аргументы должны принимать методы, пускай вам в этом поможет здравый смысл)
# 
# В этом задании не стоит флексить всякими продвинутыми штуками, для этого есть последующие
# 
# В этом задании можно использовать только модуль `datetime`


import datetime


class Chat:
    def __init__(self, chat_history, chat_history_time_period):
        self.chat_history = chat_history
        self.chat_history_time_period = chat_history_time_period
        
    def show_last_message(self):
        self.chat_history[0].show()
    
    def get_history_from_time_period(self, first_date, last_date):
        for message in range(len(self.chat_history)):
            if (self.chat_history[message].datetime > first_date) and (self.chat_history[message].datetime < last_date):
                self.chat_history_time_period.insert(0, self.chat_history[message])
                self.chat_history[message].show()
                print("\n–––––––––––\n")
        return self.chat_history_time_period
    
    
    def show_chat(self):
        for message in self.chat_history:
            message.show()
            print("\n–––––––––––\n")
    
    def recieve(self, message):
        message.send()
        self.chat_history.insert(0, message) 
    


class Message:
    def __init__(self, text, datetime, user):
        self.text = text
        self.datetime = datetime
        self.user = user
        
    def show(self):
        showed_message = f'Message: {self.text}, \nDate and time: {self.datetime}, \nUser: {self.user}'
        print(showed_message)


    def send(self):
        self.datetime = datetime.datetime.now()
        return self
    


class User:
    def __init__(self, name, login, age):
        self.name = name
        self.login = login
        self.age = age
    
    def __repr__(self):
        return f"name: {self.name} | login: {self.login} | age {self.age}"
    


# ### Примеры использования:

# #### Пример 1: Receive messages to chat, show it and show last message

chat_1 = Chat(chat_history = [], chat_history_time_period = [])


user_1 = User(name = 'Ann', login = 'Anuta_wow', age = 19)
user_2 = User(name = 'Max', login = 'Maximus', age = 22)
user_3 = User(name = 'Jack', login = 'King_Jack', age = 15)


message_1 = Message(text = 'Hello! How are you?', datetime = datetime.datetime.now(), user = user_1)
message_2 = Message(text = 'My homework is hard(', datetime = datetime.datetime.now(), user = user_2)
message_3 = Message(text = 'I am going for a walk', datetime = datetime.datetime.now(), user = user_3)


chat_1.recieve(message_1)

chat_1.recieve(message_2)

chat_1.recieve(message_3)

chat_1.show_chat()


# #### Пример 2: Show different messages

message_1.show()

message_3.show()


# #### Пример 3: Show last message

chat_1.show_last_message()


# #### Пример 4: Get history from time period

chat_1 = Chat(chat_history = [], chat_history_time_period = [])

user_1 = User(name = 'Ann', login = 'Anuta_wow', age = 19)
user_2 = User(name = 'Max', login = 'Maximus', age = 22)
user_3 = User(name = 'Jack', login = 'King_Jack', age = 15)

message_1 = Message(text = 'Hello! How are you?', datetime = datetime.datetime.now(), user = user_1)
message_2 = Message(text = 'My homework is hard(', datetime = datetime.datetime.now(), user = user_2)
message_3 = Message(text = 'I am going for a walk', datetime = datetime.datetime.now(), user = user_3)


chat_1.recieve(message_1)


# После добавления первого сообщения в чат фиксируем дату и время, начиная с которой будем будем искать сообщения (то есть первое сообщение не должно попасть в этот промежуток)


date_1 = datetime.datetime.now()

# Добавляем в чат еще 2 сообщения:

chat_1.recieve(message_2)

chat_1.recieve(message_3)

# Фиксируем вторую дату:

date_2 = datetime.datetime.now()


# Смотрим сообщения, входящие в данный временной промежуток:

chat_1.get_history_from_time_period(date_1, date_2)


# Вывелись только сообщения 2 и 3 и вернулся объект типа Chat

# # Задание 2 (3 балла)

# В питоне как-то слишком типично и неинтересно происходят вызовы функций. Напишите класс `Args`, который будет хранить в себе аргументы, а функции можно будет вызывать при помощи следующего синтаксиса.
# 
# Использовать любые модули **нельзя**, да и вряд-ли это как-то поможет)

class Args:
    
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
    
    def __rlshift__(self, other):
        return other(*self.args, **self.kwargs)



sum << Args([1, 2])

(lambda a, b, c: a**2 + b + c) << Args(1, 2, c=50)


# # Задание 3 (5 баллов)

# Сделайте класс наследник `float`. Он должен вести себя как `float`, но также должен обладать некоторыми особенностями:
# + При получении атрибутов формата `<действие>_<число>` мы получаем результат такого действия над нашим числом
# + Создавать данные атрибуты в явном виде, очевидно, не стоит
# 
# Подсказка: если в процессе гуглёжки, вы выйдете на такую тему как **"Дескрипторы", то это НЕ то, что вам сейчас нужно**
# 
# Примеры использования ниже


class StrangeFloat(float):
    def __init__(self, value):
        self.value = value
    
    def __getattribute__(self, attribute):
        if ("add" in attribute) or ("subtract" in attribute) or ("multiply" in attribute) or ("divide" in attribute):
            [action, value] = attribute.split("_")
            tmp = self.value
            if action == "add":
                tmp += float(value)
            if action == "subtract":
                tmp -= float(value)
            if action == "multiply":
                tmp *= float(value)
            if action == "divide":
                tmp /= float(value)
            result = StrangeFloat(tmp)
            return result
        else:
            return super().__getattribute__(attribute)



number = StrangeFloat(3.5)

number.add_1

number.subtract_20

number.multiply_5

number.divide_25

number.add_1.add_2.multiply_6.divide_8.subtract_9

getattr(number, "add_-2.5")   # Используем getattr, так как не можем написать number.add_-2.5 - это SyntaxError

number + 8   # Стандартные для float операции работают также

number.as_integer_ratio()   # Стандартные для float операции работают также  (это встроенный метод float, писать его НЕ НАДО)


# # Задание 4 (3 балла)

# В данном задании мы немного отдохнём и повеселимся. От вас требуется заменить в данном коде максимально возможное количество синтаксических конструкций на вызовы dunder методов, dunder атрибутов и dunder переменных.
# 
# Маленькая заметка: полностью всё заменить невозможно. Например, `function()` можно записать как `function.__call__()`, но при этом мы всё ещё не избавляемся от скобочек, так что можно делать так до бесконечности `function.__call__.__call__.__call__.__call__.....__call__()` и при всём при этом мы ещё не избавляемся от `.` для доступа к атрибутам. В общем, замените всё, что получится, не закапываясь в повторы, как в приведённом примере. Чем больше разных методов вы найдёте и используете, тем лучше и тем выше будет балл
# 
# Код по итогу дожен работать и печатать число **4420.0**, как в примере. Структуру кода менять нельзя, просто изменяем конструкции на синонимичные
# 
# И ещё маленькая подсказка. Заменить здесь можно всё кроме:
# + Конструкции `for ... in ...`:
# + Синтаксиса создания лямбда функции
# + Оператора присваивания `=`
# + Конструкции `if-else`

import numpy as np


matrix = []
for idx in range(0, 100, 10):
    matrix += [list(range(idx, idx + 10))]
    
selected_columns_indices = list(filter(lambda x: x in range(1, 5, 2), range(len(matrix))))
selected_columns = map(lambda x: [x[col] for col in selected_columns_indices], matrix)

arr = np.array(list(selected_columns))

mask = arr[:, 1] % 3 == 0
new_arr = arr[mask]

product = new_arr @ new_arr.T

if (product[0] < 1000).all() and (product[2] > 1000).any():
    print(product.mean())





import numpy as np


matrix = []
for idx in range(0, 100, 10):
    matrix.__iadd__([list(range(idx, idx.__add__(10)))])
    
selected_columns_indices = list(filter(lambda x: range(1, 5, 2).__contains__(x), range(len(matrix))))
selected_columns = map(lambda x: [x.__getitem__(col) for col in selected_columns_indices], matrix)

arr = np.array(list(selected_columns))

mask = arr[:, 1].__mod__(3).__eq__(0)
new_arr = arr.__getitem__(mask)

product = new_arr.__matmul__(new_arr.T)

if (product.__getitem__(0).__lt__(1000)).all().__and__((product.__getitem__(2).__gt__(1000)).any()):
    print(product.mean())


# # Задание 5 (10 баллов)

# Напишите абстрактный класс `BiologicalSequence`, который задаёт следующий интерфейс:
# + Работа с функцией `len`
# + Возможность получать элементы по индексу и делать срезы последовательности (аналогично строкам)
# + Вывод на печать в удобном виде и возможность конвертации в строку
# + Возможность проверить алфавит последовательности на корректность
# 
# Напишите класс `NucleicAcidSequence`:
# + Данный класс реализует интерфейс `BiologicalSequence`
# + Данный класс имеет новый метод `complement`, возвращающий комплементарную последовательность
# + Данный класс имеет новый метод `gc_content`, возвращающий GC-состав (без разницы, в процентах или в долях)
# 
# Напишите классы наследники `NucleicAcidSequence`: `DNASequence` и `RNASequence`
# + `DNASequence` должен иметь метод `transcribe`, возвращающий транскрибированную РНК-последовательность
# + Данные классы не должны иметь <ins>публичных методов</ins> `complement` и метода для проверки алфавита, так как они уже должны быть реализованы в `NucleicAcidSequence`.
# 
# Напишите класс `AminoAcidSequence`:
# + Данный класс реализует интерфейс `BiologicalSequence`
# + Добавьте этому классу один любой метод, подходящий по смыслу к аминокислотной последовательности. Например, метод для нахождения изоэлектрической точки, молекулярного веса и т.д.
# 
# Комментарий по поводу метода `NucleicAcidSequence.complement`, так как я хочу, чтобы вы сделали его опредедённым образом:
# 
# При вызове `dna.complement()` или условного `dna.check_alphabet()` должны будут вызываться соответствующие методы из `NucleicAcidSequence`. При этом, данный метод должен обладать свойством полиморфизма, иначе говоря, внутри `complement` не надо делать условия а-ля `if seuqence_type == "DNA": return self.complement_dna()`, это крайне не гибко. Данный метод должен опираться на какой-то общий интерфейс между ДНК и РНК. Создание экземпляров `NucleicAcidSequence` не подразумевается, поэтому код `NucleicAcidSequence("ATGC").complement()` не обязан работать, а в идеале должен кидать исключение `NotImplementedError` при вызове от экземпляра `NucleicAcidSequence`
# 
# Вся сложность задания в том, чтобы правильно организовать код. Если у вас есть повторяющийся код в сестринских классах или родительском и дочернем, значит вы что-то делаете не так.
# 
# 
# Маленькое замечание: По-хорошему, между классом `BiologicalSequence` и классами `NucleicAcidSequence` и `AminoAcidSequence`, ещё должен быть класс-прослойка, частично реализующий интерфейс `BiologicalSequence`, но его писать не обязательно, так как задание и так довольно большое (правда из-за этого у вас неминуемо возникнет повторяющийся код в классах `NucleicAcidSequence` и `AminoAcidSequence`)



from abc import ABC, abstractmethod


class BiologicalSequence(ABC):
    
    @abstractmethod
    def __len__():
        pass
    
    @abstractmethod
    def __getitem__():
        pass
    
    @abstractmethod
    def __repr__():
        pass
    
    @abstractmethod
    def check_alphabet():
        pass
    
    
    def __getattribute__(self, attribute):
        if "forbiden_methods" in attribute:
            try:
                return super().__getattribute__(attribute)
            except AttributeError:
                return []
        forbiden_methods = self._forbiden_methods
        for method in forbiden_methods:
            if method in attribute:
                raise NotImplementedError(f"Method {attribute}() is forbiden")
        return super().__getattribute__(attribute)




class NucleicAcidSequence(BiologicalSequence):
    
    def __init__(self, sequence):
        self.sequence = sequence
        self._forbiden_methods = ["complement", "check_alphabet"]
    
    
    def __len__(self):
        return len(self.sequence)
    
    
    def __getitem__(self, slc):
        return self.sequence[slc]
    
    
    def __repr__(self):
        return self.sequence
    
    
    def check_alphabet(self):
        self.sequence = self.sequence.upper()
        true_seq = 'AGTC'
        if set(self.sequence) <= set(true_seq):
            print('It is okey!')
        else:
            raise AlphabetError(f'{self.sequence} is not a true biological nucleotide sequence!')
    
    def complement(self):
        self.sequence = self.sequence.upper()
        complement_dict = {'A':'T', 'T':'A', 'C':'G', 'G':'C'}
        seq_complemented = ''
        for i in self.sequence:
            seq_complemented += str(complement_dict.get(i))
        return seq_complemented
    
    def gc_content(self):
        self.sequence = self.sequence.upper()
        len_seq = len(self.sequence)
        count_gc = self.sequence.count('G') + self.sequence.count('C')
        result_gc = round(count_gc/len_seq, 3)
        return result_gc





class DNASequence(NucleicAcidSequence):
    
    def __init__(self, sequence):
        self.sequence = sequence
    
    def transcribe(self):
        self.sequence = self.sequence.upper()
        transcribe_dict = {'A':'A', 'T':'U', 'C':'C', 'G':'G'}
        seq_transcribed = ''
        for i in self.sequence:
            seq_transcribed += str(transcribe_dict.get(i))
        print(seq_transcribed)
        

class RNASequence(NucleicAcidSequence):
    
    def __init__(self, sequence):
        self.sequence = sequence




class AminoAcidSequence(BiologicalSequence):
    
    def __init__(self, sequence):
        self.sequence = sequence

        
    def __len__(self):
        return len(self.sequence)
    
    
    def __getitem__(self, slc):
        return self.sequence[slc]
    
    
    def __repr__(self):
        return self.sequence
    
    
    def check_alphabet(self):
        self.sequence = self.sequence.upper()
        true_seq = 'ACDEFGHIKLMNPQRSTVWY'
        if set(self.sequence) <= set(true_seq):
            print('It is okey!')
        else:
            raise AlphabetError(f'{self.sequence} is not a true biological aminoacid sequence!')
            
    
    def molecular_weight(self):
        self.sequence = self.sequence.upper()
        amino_weights = {"A":89.09, "R":174.20, "N":132.12, 
                 "D":133.10, "C":121.15, "Q":146.15, 
                 "E":147.13, "G":75.07, "H":155.16, 
                 "I":131.17, "L":131.17, "K":146.19, 
                 "M":149.21, "F":165.19, "P":115.13, 
                 "S":105.09, "T":119.12, "W":204.23, 
                 "Y":181.19, "V":117.15} 
        weight = 0
        for aa in self.sequence:
            weight += amino_weights.get(aa)
        return weight
        

