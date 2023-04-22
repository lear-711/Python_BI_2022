#!/usr/bin/env python
# coding: utf-8

# В формулировке заданий будет использоваться понятие **worker**. Это слово обозначает какую-то единицу параллельного выполнения, в случае питона это может быть **поток** или **процесс**, выбирайте то, что лучше будет подходить к конкретной задаче
# 
# В каждом задании нужно писать подробные аннотиции типов для:
# 1. Аргументов функций и классов
# 2. Возвращаемых значений
# 3. Классовых атрибутов (если такие есть)
# 
# В каждом задании нужно писать докстроки в определённом стиле (какой вам больше нравится) для всех функций, классов и методов

# # Задание 1 (7 баллов)

# В одном из заданий по ML от вас требовалось написать кастомную реализацию Random Forest. Её проблема состоит в том, что она работает медленно, так как использует всего один поток для работы. Добавление параллельного программирования в код позволит получить существенный прирост в скорости обучения и предсказаний.
# 
# В данном задании от вас требуется добавить возможность обучать случайный лес параллельно и использовать параллелизм для предсказаний. Для этого вам понадобится:
# 1. Добавить аргумент `n_jobs` в метод `fit`. `n_jobs` показывает количество worker'ов, используемых для распараллеливания
# 2. Добавить аргумент `n_jobs` в методы `predict` и `predict_proba`
# 3. Реализовать функционал по распараллеливанию в данных методах
# 
# В результате код `random_forest.fit(X, y, n_jobs=2)` и `random_forest.predict(X, y, n_jobs=2)` должен работать в ~1.5-2 раза быстрее, чем `random_forest.fit(X, y, n_jobs=1)` и `random_forest.predict(X, y, n_jobs=1)` соответственно
# 
# Если у вас по каким-то причинам нет кода случайного леса из ДЗ по ML, то вы можете написать его заново или попросить у однокурсника. *Детали* реализации ML части оцениваться не будут, НО, если вы поломаете логику работы алгоритма во время реализации параллелизма, то за это будут сниматься баллы
# 
# В задании можно использовать только модули из **стандартной библиотеки** питона, а также функции и классы из **sklearn** при помощи которых вы изначально писали лес



import multiprocessing as mp


# переключаемся в режим работы fork для создания новых процессов – по скольку в MacOS по-умолчанию spawn
mp.set_start_method("fork")


from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from typing import Optional
import numpy as np
import random


class RandomForestClassifierCustom(BaseEstimator):
    
    """
    This class implements a random forest classifier. It is a meta estimator that fits a number of 
    decision tree classifiers on various sub-samples of the dataset
    """
    
    def __init__(
        self, 
        n_estimators: int = 10, 
        max_depth: Optional[int] = None,
        max_features: Optional[int] = None, 
        random_state: int = 1
    ):
        
        """
        Variable initialization.

        Parameters
        ----------
        n_estimators : int
            The number of trees in the forest.
        max_depth : int or None
            The maximum depth of the tree.
        max_features : int or None
            The number of features to consider when looking for the best split.
        random_state : int
            Controls both the randomness of the bootstrapping of the samples used when building trees
            and the sampling of the features to consider when looking for the best split at each node.

        """
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state

        self.trees = []
        self.feat_ids_by_tree = []

        
        
    def single_fit(self, X: list, y: list, estimator: int):

        """
        Fitting one of n_estimators
        
        Parameters
        ----------
        X : list
            Train features.
        Y : list
            Train target.
        estimator : int
            Index of estimator.

        """

        random.seed(self.random_state + estimator)

        # выбираем признаки для модели
        current_features_ids = random.sample([j for j in range(len(X[0]))], self.max_features)

        # делаем превдовыборку
        bootstrap_sample = []
        for X_i in X:
            X_chosen = [X_i[cfi] for cfi in current_features_ids] # выбранные признаки для каждой строки
            bootstrap_sample.append(X_chosen)

        # создаем и обучаем модель
        dtc = DecisionTreeClassifier(max_depth = self.max_depth, 
                                    max_features = self.max_features, 
                                    random_state = self.random_state) 
        dtc.fit(bootstrap_sample, y)
        return current_features_ids, dtc, estimator
        
        
    def fit(self, X: list, y: list, n_jobs: int):
        
        """
        Build a forest of trees from the training set.

        Parameters
        ----------
        X : list
            Train features.
        Y : list
            Train target.
        n_jobs: int
            The number of jobs to run in parallel.

        Returns
        -------
        RandomForestClassifierCustom
            RandomForestClassifierCustom class instance (self).
        """
        
        self.classes_ = sorted(np.unique(y))
            
        with mp.Pool(processes = n_jobs) as pool:
            results = pool.starmap(self.single_fit, [(X, y, estimator) for estimator in range(self.n_estimators)])
            
            for i, result in enumerate(results):
                current_features_ids, dtc, estimator = result[0], result[1], result[2]
                self.feat_ids_by_tree.append(current_features_ids)
                self.trees.append(dtc)

        
        return self

    
    def single_predict_proba(self, X: list, index: int):
        
        """
        Run single predict proba.

        Parameters
        ----------
        X : list
            Test features.
        index : int
            Index of features and model.

        Returns
        -------
        list
            The class probabilities of the input sample.
        """
        
        model = self.trees[index]
        feat_ids = self.feat_ids_by_tree[index]

        bootstrap_sample = []
        for X_i in X:
            X_chosen = [X_i[cfi] for cfi in feat_ids]
            bootstrap_sample.append(X_chosen)
        proba = model.predict_proba(bootstrap_sample)
        
        return proba

    
    
    def predict_proba(self, X: list, n_jobs: int):
        
        """
        Predict class probabilities for X.

        Parameters
        ----------
        X : list
            Test features.
        n_jobs: int
            The number of jobs to run in parallel of proba prediction.

        Returns
        -------
        list
            The class probabilities of the input samples.
        """
        
        probas_trees = [ [0]*len(self.classes_) for _ in X ]
        
        with mp.Pool(processes = n_jobs) as pool:
            proba_results = pool.starmap(self.single_predict_proba, [(X, estimator) for estimator in range(self.n_estimators)])
            
            for row, proba in enumerate(proba_results):
                for c in self.classes_:
                    probas_trees[row][c] += proba[row][c]
                
        for row in range(len(probas_trees)):
            for c in self.classes_:
                probas_trees[row][c] /= len(self.trees)

        return probas_trees
            
    
    def predict(self, X: list, n_jobs: int):
        
        
        """
        Predict class for X.

        Parameters
        ----------
        X : list
            Test features.
        n_jobs: int
            The number of jobs to run in parallel.

        Returns
        -------
        list
            The predicted classes.
        """
        
        
        probas = self.predict_proba(X, n_jobs)
        predictions = np.argmax(probas, axis=1)
        
        return predictions


    
X, y = make_classification(n_samples=100000)


random_forest = RandomForestClassifierCustom(max_depth=30, n_estimators=10, max_features=2, random_state=42)


get_ipython().run_cell_magic('time', '', '\n_ = random_forest.fit(X, y, n_jobs=1)\n')


get_ipython().run_cell_magic('time', '', '\npreds_1 = random_forest.predict(X, n_jobs=1)\n')


random_forest = RandomForestClassifierCustom(max_depth=30, n_estimators=10, max_features=2, random_state=42)


get_ipython().run_cell_magic('time', '', '\n_ = random_forest.fit(X, y, n_jobs=2)\n')


get_ipython().run_cell_magic('time', '', '\npreds_2 = random_forest.predict(X, n_jobs=2)\n')


(preds_1 == preds_2).all()   # Количество worker'ов не должно влиять на предсказания


# #### Какие есть недостатки у вашей реализации параллельного Random Forest (если они есть)? Как это можно исправить? Опишите словами, можно без кода (+1 дополнительный балл)

# 
# Сейчас n_jobs надо задавать принудительно. Можно сделать параметр `n_jobs=-1` по-умолчанию и определять его как максимальное число процессоров. 
# 
# Использование `fork` как способа запуска процесса не является безопастным: есть риск получить процесс-зомби, если не завершить дочерний процесс. Хотя это контролируется контекстным менеджером `with`. Подробнее о проблеме тут: https://docs.python.org/3/library/multiprocessing.html#the-spawn-and-forkserver-start-methods



# # Задание 2 (9 баллов)

# Напишите декоратор `memory_limit`, который позволит ограничивать использование памяти декорируемой функцией.
# 
# Декоратор должен принимать следующие аргументы:
# 1. `soft_limit` - "мягкий" лимит использования памяти. При превышении функцией этого лимита должен будет отображён **warning**
# 2. `hard_limit` - "жёсткий" лимит использования памяти. При превышении функцией этого лимита должно будет брошено исключение, а функция должна немедленно завершить свою работу
# 3. `poll_interval` - интервал времени (в секундах) между проверками использования памяти
# 
# Требования:
# 1. Потребление функцией памяти должно отслеживаться **во время выполнения функции**, а не после её завершения
# 2. **warning** при превышении `soft_limit` должен отображаться один раз, даже если функция переходила через этот лимит несколько раз
# 3. Если задать `soft_limit` или `hard_limit` как `None`, то соответствующий лимит должен быть отключён
# 4. Лимиты должны передаваться и отображаться в формате `<number>X`, где `X` - символ, обозначающий порядок единицы измерения памяти ("B", "K", "M", "G", "T", ...)
# 5. В тексте warning'ов и исключений должен быть указан текщий объём используемой памяти и величина превышенного лимита
# 
# В задании можно использовать только модули из **стандартной библиотеки** питона, можно писать вспомогательные функции и/или классы
# 
# В коде ниже для вас предопределены некоторые полезные функции, вы можете ими пользоваться, а можете не пользоваться



import os
import psutil
import time
import warnings
from typing import Union


def get_memory_usage(pid: int = os.getpid()):    # Показывает текущее потребление памяти процессом
    
    """
    Shows the current memory consumption of the process.
    
    Parameters
    ----------
    pit : int
        Process ID.

    Returns
    -------
    int
        Physical non-paging memory used by the process.
    """
    
    process = psutil.Process(pid)
    mem_info = process.memory_info()
    return mem_info.rss



def bytes_to_human_readable(n_bytes: int):
    
    """
    Converts the number of bytes passed to the input into a short readable record.
    
    Parameters
    ----------
    n_bytes : int
        Number of bytes you need to convert.

    Returns
    -------
    str
        Number of bytes or kilobytes or megabytes or gigabytes or terabytes or petabytes
        or exabytes or zettabytes or yottabytes with its symobol ('K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y').
    """
    
    symbols = ('K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y')
    prefix = {}
    for idx, s in enumerate(symbols):
        prefix[s] = 1 << (idx + 1) * 10
    for s in reversed(symbols):
        if n_bytes >= prefix[s]:
            value = float(n_bytes) / prefix[s]
            return f"{value:.2f}{s}"
    return f"{n_bytes}B"



def check_memory_by_pid(pid: int, 
                        soft_limit: Union[str, None] = None, 
                        hard_limit: Union[str, None] = None):
    
    """
    Checks the memory used by the process and compares it with soft limit and hard limit.
    
    Parameters
    ----------
    pid : int
        Process ID.
    soft_limit : str or None
        "Soft" memory usage limit.
    hard_limit: str or None
        "Hard" memory usage limit.

    Returns
    -------
    int
        One of the numbers 0, 1 or 2:
        0 - memory limit has not been reached
        1 - memory soft limit exceeded
        2 - memory hard limit exceeded
    """
    
    symbols = ('K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y')

    mem_use = get_memory_usage(pid)
    mem_use_readable = bytes_to_human_readable(mem_use)

    
    if (soft_limit is not None) and (hard_limit is not None):
    
        # Если превышает soft limit
        if (
            (symbols.index(soft_limit[-1]) < symbols.index(mem_use_readable[-1])) or 
            ((symbols.index(soft_limit[-1]) == symbols.index(mem_use_readable[-1])) and 
             (float(soft_limit[:-1]) < float(mem_use_readable[:-1])))
            ):

            # Проверяем hard limit

            # Если не превышает hard limit
            if ((symbols.index(hard_limit[-1]) > symbols.index(mem_use_readable[-1])) or 
                ((symbols.index(hard_limit[-1]) == symbols.index(mem_use_readable[-1])) and 
                 (float(hard_limit[:-1]) >= float(mem_use_readable[:-1])))
               ):

                return 1

            # Если превышает hard limit
            else: 
                return 2

        else:
            return 0
        
        
    elif (soft_limit is not None) and (hard_limit is None):
        # Если превышает soft limit
        if (
            (symbols.index(soft_limit[-1]) < symbols.index(mem_use_readable[-1])) or 
            ((symbols.index(soft_limit[-1]) == symbols.index(mem_use_readable[-1])) and 
             (float(soft_limit[:-1]) < float(mem_use_readable[:-1])))
            ):
            return 1
        
        
    elif (soft_limit is None) and (hard_limit is not None):
        # Если превышает hard limit
        if ((symbols.index(hard_limit[-1]) < symbols.index(mem_use_readable[-1])) or 
            ((symbols.index(hard_limit[-1]) == symbols.index(mem_use_readable[-1])) and 
             (float(hard_limit[:-1]) < float(mem_use_readable[:-1])))
           ):
            return 2
        
            

def memory_limit(soft_limit: Union[int, float, None] = None, 
                 hard_limit: Union[int, float, None] = None, 
                 poll_interval=1):
    
    """
    Decorator , which allows you to limit the memory usage of the function being decorated.
    
    Parameters
    ----------
    soft_limit : int, float or None
        "Soft" memory usage limit.
    hard_limit: int, float or None
        "Hard" memory usage limit.
    poll_interval: 
        Time interval (in seconds) between memory usage checks.

    Returns
    -------
    func
        Function with limited memory usage.
    """
    
    def func_result_to_queue(*args, **kwargs):
        
        """
        Function that runs target func and puts result into queue.
        """

        queue = args[0]
        func = args[1]
        real_args = args[2]
        result = func(*real_args, **kwargs)
        queue.put(result)
        return
    

    def wrap(func):
        def inner_function(*args, **kwargs):
            with mp.Manager() as manager:
                queue = manager.Queue()
                
                # здесь в аргументы args передаём нашу очередь и функцию, которую надо запустить
                p = mp.Process(target = func_result_to_queue, 
                               args = (queue, func, args), 
                               kwargs = kwargs)
                p.start()
                while p.is_alive():
                    result_memory = check_memory_by_pid(p.pid, soft_limit = soft_limit, hard_limit = hard_limit)
                    if result_memory == 1:
                        warnings.warn("Warning: You have exceeded the soft memory limit!")
                    elif result_memory == 2:
                        raise MemoryError('You have exceeded the hard memory limit!')
                    time.sleep(poll_interval)
                    
                # забираем результат работы функции, он лежит в очереди
                result = queue.get()
                p.join()
                
                return result
        return inner_function
    return wrap




def memory_increment():
    """
    Функция для тестирования
    
    В течение нескольких секунд достигает использования памяти 1.89G
    Потребление памяти и скорость накопления можно варьировать, изменяя код
    """
    lst = []
    for i in range(50000000):
        if i % 500000 == 0:
            time.sleep(0.1)
        lst.append(i)
    return lst




@memory_limit(soft_limit="212M", hard_limit="800M", poll_interval=0.1)
def memory_increment():
    """
    Функция для тестирования
    
    В течение нескольких секунд достигает использования памяти 1.89G
    Потребление памяти и скорость накопления можно варьировать, изменяя код
    """
    lst = []
    for i in range(50000000):
        if i % 500000 == 0:
            time.sleep(0.1)
        lst.append(i)
    return lst


memory_increment()


@memory_limit(soft_limit=None, hard_limit=None, poll_interval=1)
def memory_increment():
    """
    Функция для тестирования
    
    В течение нескольких секунд достигает использования памяти 1.89G
    Потребление памяти и скорость накопления можно варьировать, изменяя код
    """
    lst = []
    for i in range(50000): #000
        if i % 500 == 0: #000
            time.sleep(0.1)
        lst.append(i)
    return lst



memory_increment()



# # Задание 3 (11 баллов)

# Напишите функцию `parallel_map`. Это должна быть **универсальная** функция для распараллеливания, которая эффективно работает в любых условиях.
# 
# Функция должна принимать следующие аргументы:
# 1. `target_func` - целевая функция (обязательный аргумент)
# 2. `args_container` - контейнер с позиционными аргументами для `target_func` (по-умолчанию `None` - позиционные аргументы не передаются)
# 3. `kwargs_container` - контейнер с именованными аргументами для `target_func` (по-умолчанию `None` - именованные аргументы не передаются)
# 4. `n_jobs` - количество workers, которые будут использованы для выполнения (по-умолчанию `None` - количество логических ядер CPU в системе)
# 
# Функция должна работать аналогично `***PoolExecutor.map`, применяя функцию к переданному набору аргументов, но с некоторыми дополнениями и улучшениями
#     
# Поскольку мы пишем **универсальную** функцию, то нам нужно будет выполнить ряд требований, чтобы она могла логично и эффективно работать в большинстве ситуаций
# 
# 1. `target_func` может принимать аргументы любого вида в любом количестве
# 2. Любые типы данных в `args_container`, кроме `tuple`, передаются в `target_func` как единственный позиционный аргумент. `tuple` распаковываются в несколько аргументов
# 3. Количество элементов в `args_container` должно совпадать с количеством элементов в `kwargs_container` и наоборот, также значение одного из них или обоих может быть равно `None`, в иных случаях должна кидаться ошибка (оба аргумента переданы, но размеры не совпадают)
# 
# 4. Функция должна выполнять определённое количество параллельных вызовов `target_func`, это количество зависит от числа переданных аргументов и значения `n_jobs`. Сценарии могут быть следующие
#     + `args_container=None`, `kwargs_container=None`, `n_jobs=None`. В таком случае функция `target_func` выполнится параллельно столько раз, сколько на вашем устройстве логических ядер CPU
#     + `args_container=None`, `kwargs_container=None`, `n_jobs=5`. В таком случае функция `target_func` выполнится параллельно **5** раз
#     + `args_container=[1, 2, 3]`, `kwargs_container=None`, `n_jobs=5`. В таком случае функция `target_func` выполнится параллельно **3** раза, несмотря на то, что `n_jobs=5` (так как есть всего 3 набора аргументов для которых нам нужно получить результат, а лишние worker'ы создавать не имеет смысла)
#     + `args_container=None`, `kwargs_container=[{"s": 1}, {"s": 2}, {"s": 3}]`, `n_jobs=5`. Данный случай аналогичен предыдущему, но здесь мы используем именованные аргументы
#     + `args_container=[1, 2, 3]`, `kwargs_container=[{"s": 1}, {"s": 2}, {"s": 3}]`, `n_jobs=5`. Данный случай аналогичен предыдущему, но здесь мы используем и позиционные, и именованные аргументы
#     + `args_container=[1, 2, 3, 4]`, `kwargs_container=None`, `n_jobs=2`. В таком случае в каждый момент времени параллельно будет выполняться **не более 2** функций `target_func`, так как нам нужно выполнить её 4 раза, но у нас есть только 2 worker'а.
#     + В подобных случаях (из примера выше) должно оптимизироваться время выполнения. Если эти 4 вызова выполняются за 5, 1, 2 и 1 секунды, то параллельное выполнение с `n_jobs=2` должно занять **5 секунд** (не 7 и тем более не 10)
# 
# 5. `parallel_map` возвращает результаты выполнения `target_func` **в том же порядке**, в котором были переданы соответствующие аргументы
# 6. Работает с функциями, созданными внутри других функций
# 
# Для базового решения от вас не ожидается **сверххорошая** оптимизация по времени и памяти для всех возможных случаев. Однако за хорошо оптимизированную логику работы можно получить до **+3 дополнительных баллов**
# 
# Вы можете сделать класс вместо функции, если вам удобнее
# 
# В задании можно использовать только модули из **стандартной библиотеки** питона
# 
# Ниже приведены тестовые примеры по каждому из требований



import types
import functools
import os
from typing import Callable


# чтобы сделать возможным вложенный запуск parallel_map и не делать ручной pickle 
# сделаем функцию: которая будет копировать функцию
def copy_func(f: Callable, 
              name: Optional[str] = None):
    '''
    Return a function with same code, globals, defaults, closure, and 
    name (or provide a new name).
    
    Parameters
    ----------
    f : Callable
        Any function to be copied.
    name: str or None
        Name of function or same as function if None.

    Returns
    -------
    func
        Copy of function f.
    '''
    
    fn = types.FunctionType(f.__code__, f.__globals__, name or f.__name__,
        f.__defaults__, f.__closure__)
    # in case f was given attrs (note this dict is a shallow copy):
    fn.__dict__.update(f.__dict__) 
    return fn


def target_func_kwargs_mapper(kwargs: dict):
    
    """
    Function wrapper that applies kwargs to traget function.
    
    Parameters
    ----------
    kwargs : dict
        Dict of kwargs and the function itself stored in key `___target_func`.

    Returns
    -------
    result
        Result of the target function.
    """

    # достаем функцию которую надо запустить
    target_func = kwargs.pop('___target_func')
    return target_func(**kwargs)


def parallel_map(target_func: Callable,
                 args_container: Union[list, list[tuple], None] = None,
                 kwargs_container: Optional[list[dict]] = None,
                 n_jobs: Optional[int] = None,
                 use_imap: Optional[bool] = True,  # использование ленивого распределения на потоки
                 log_amount_of_jobs: Optional[bool] = False
                ):
    
    """
    Generic function for parallelization.
    
    Parameters
    ----------
    target_func : Callable
        Function to run in parallel
    args_container: list, list[tuple] or None
        List of args to be applied to target_func in parallel.
    kwargs_container: list[dict] or None
        List of kwargs to be applied to target_func in parallel.
    n_jobs: int or None
        Number of workers.
    use_imap: bool or None
        Use lazy optimization for argument mapping on processes. Is on by default.
    log_amount_of_jobs: bool or None 
        Flag to log the amout of workers to be used. Is off by default.

    Returns
    -------
    results
        List of function results.
    """   
    
    # Оптимизации:
    # * вычисление минимального количества одновременно допустимых запущенных процессов: len(args) ? n_jobs
    
    # По скольку делать свой менеджер количества запущенных процессов не охота, и лучше это доверить with
    # а способа граммотно прокинуть args и kwargs нет, но есть pool.starmap(), который применит позиционные аргументы
    # проще сначала грамотно скомпановать из args и kwargs один набор kwargs, 
    # передать эти kwargs в некую функцию-обёртку, которая передаст в target_func наши kwargs
    # это нужно потому что imap умеет передавать в нужную функцию только один аргумент
    # а затем использовать imap() для ленивого запуска, поскольку он гораздо эффективнее по памяти, чем map()
    # 
    
    working_with_nested_function_flag = False
    if target_func.__qualname__ != target_func.__name__:
        # имеем дело с вложенной функцией
        target_func = copy_func(target_func)
        globals()[target_func.__name__] = target_func
        working_with_nested_function_flag = True
    
    if (args_container is not None) and (kwargs_container is not None):
        if len(args_container) != len(kwargs_container):
            raise Exception("args and kwargs container lengths does not match.")
    
    if (kwargs_container is None) and (args_container is not None):
        # чтобы в zip все нормально выполнилось когда нет kwargs_container,
        # сделаем так, чтобы их "размерности" совпадали
        kwargs_container = []
        for each in args_container:
            kwargs_container.append({})
        
    if (kwargs_container is not None) and (args_container is None):
        # аналогично для args_container
        args_container = []
        for each in kwargs_container:
            args_container.append(None)
    
    if (kwargs_container is None) and (args_container is None):
        # обработаем случай, когда оба пустые
        # с учетом того, что надо запустить функцию n_jobs, раз
        # такое поведение описано в примерах 4.1 и 4.2
        min_runs = os.cpu_count()
        if n_jobs is not None:
            min_runs = min(n_jobs, min_runs)
        kwargs_container = [{}] * min_runs
        args_container = [None] * min_runs
    
    var_names = target_func.__code__.co_varnames
    kwargs_to_parallel_execution = []
    for args, kwargs in zip(args_container, kwargs_container):
        new_kwargs = {}
        if args is not None:
            if type(args) != tuple:
                # нам передали один позиционный аргумент, то есть первый из позиционных аргументов
                new_kwargs[var_names[0]] = args
            else:
                # нам передали кортеж позиционных аргументов, значит надо их всех добавить
                for i, arg in enumerate(args):
                    new_kwargs[var_names[i]] = arg
        
        for key in kwargs:
            # добавим значения из kwargs в new_kwargs, даже если это значение уже есть – перезапишем.
            # будем считать kwargs главнее
            new_kwargs[key] = kwargs[key]
        
        # в kwaргументы добавляем нашу целевую функцию, чтобы в target_func_kwargs_mapper её потом достать
        # поскольку добавление функции, это лишь ссылка не неё: по памяти мы не сильно проиграем (отыграем за счет imap)
        new_kwargs["___target_func"] = target_func
        kwargs_to_parallel_execution.append(new_kwargs)

    results = [] 
    # определяем количество минимально необходимых воркеров, чтобы не плодить лишнего
    min_n_jobs = 1
    if n_jobs is None:
        # ничего не передали: значит по-умолчанию будет количество логических CPU
        min_n_jobs = os.cpu_count()
    else:
        # если передали, то возьмем минимально необходимое
        # выбираем его исходя из количества переданных аргументов, n_jobs, и количества логических ядер в принципе
        min_n_jobs = min(len(kwargs_to_parallel_execution), n_jobs, os.cpu_count())
    
    if log_amount_of_jobs:
        print(f"Running with pool of {min_n_jobs} processes.")
    with mp.Pool(processes=n_jobs) as pool:
        # используем ленивое распределение на потоки imap, но оставляем возможность сравнить скорость с map
        if use_imap:
            for result in pool.imap(target_func_kwargs_mapper, kwargs_to_parallel_execution):
                results.append(result)
        else:
            results = pool.map(target_func_kwargs_mapper, kwargs_to_parallel_execution)
    
    # если работали с вложенной функцией – подчищаем
    if working_with_nested_function_flag:
        del globals()[target_func.__name__]
    return results




import time


# Это только один пример тестовой функции, ваша parallel_map должна уметь эффективно работать с ЛЮБЫМИ функциями
# Поэтому обязательно протестируйте код на чём-нибудбь ещё
def test_func(x=1, s=2, a=1, b=1, c=1):
    time.sleep(s)
    return a*x**2 + b*x + c


get_ipython().run_cell_magic('time', '', '\n# Пример 0.1\n# Возможна одновременная передача args_container и kwargs_container, но количества элементов в них должны быть равны\nparallel_map(test_func,\n             args_container=[i for i in range(20)],\n             kwargs_container=[{"s": 3}]*20,\n             use_imap=True,\n            )\n')


get_ipython().run_cell_magic('time', '', '\n# Пример 0.2\n# Возможна одновременная передача args_container и kwargs_container, но количества элементов в них должны быть равны\nparallel_map(test_func,\n             args_container=[i for i in range(20)],\n             kwargs_container=[{"s": 3}]*20,\n             use_imap=False,\n            )\n')


# Судя по результату imap знатно так срезал время работы, аж на одну итерацию из 20 \
# (Акутально для процессов с малым количеством логических ядер))


get_ipython().run_cell_magic('time', '', '\n# Пример 2.1\n# Отдельные значения в args_container передаются в качестве позиционных аргументов\nparallel_map(test_func, args_container=[1, 2.0, 3j-1, 4])   # Здесь происходят параллельные вызовы: test_func(1) test_func(2.0) test_func(3j-1) test_func(4)\n')



get_ipython().run_cell_magic('time', '', '\n# Пример 2.2\n# Элементы типа tuple в args_container распаковываются в качестве позиционных аргументов\nparallel_map(test_func, [(1, 1), (2.0, 2), (3j-1, 3), 4])    # Здесь происходят параллельные вызовы: test_func(1, 1) test_func(2.0, 2) test_func(3j-1, 3) test_func(4)\n')



get_ipython().run_cell_magic('time', '', '\n# Пример 3.1\n# Возможна одновременная передача args_container и kwargs_container, но количества элементов в них должны быть равны\nparallel_map(test_func,\n             args_container=[1, 2, 3, 4],\n             kwargs_container=[{"s": 3}, {"s": 3}, {"s": 3}, {"s": 3}])\n\n# Здесь происходят параллельные вызовы: test_func(1, s=3) test_func(2, s=3) test_func(3, s=3) test_func(4, s=3)\n')



get_ipython().run_cell_magic('time', '', '\n# Пример 3.2\n# args_container может быть None, а kwargs_container задан явно\nparallel_map(test_func,\n             kwargs_container=[{"s": 3}, {"s": 3}, {"s": 3}, {"s": 3}])\n')



get_ipython().run_cell_magic('time', '', '\n# Пример 3.3\n# kwargs_container может быть None, а args_container задан явно\nparallel_map(test_func,\n             args_container=[1, 2, 3, 4])\n')



get_ipython().run_cell_magic('time', '', '\n# Пример 3.4\n# И kwargs_container, и args_container могут быть не заданы\nparallel_map(test_func)\n')



get_ipython().run_cell_magic('time', '', '\n# Пример 3.4\n# И kwargs_container, и args_container могут быть не заданы\nparallel_map(test_func)\n')



get_ipython().run_cell_magic('time', '', '\n# Пример 3.5\n# При несовпадении количеств позиционных и именованных аргументов кидается ошибка\nparallel_map(test_func,\n             args_container=[1, 2, 3, 4],\n             kwargs_container=[{"s": 3}, {"s": 3}, {"s": 3}])\n')



get_ipython().run_cell_magic('time', '', '\n# Пример 4.1\n# Если функция не имеет обязательных аргументов и аргумент n_jobs не был передан, то она выполняется параллельно столько раз, сколько ваш CPU имеет логических ядер\n# В моём случае это 24, у вас может быть больше или меньше\nparallel_map(test_func, log_amount_of_jobs=True)\n')



get_ipython().run_cell_magic('time', '', '\n# Пример 4.2\n# Если функция не имеет обязательных аргументов и передан только аргумент n_jobs, то она выполняется параллельно n_jobs раз\nparallel_map(test_func, n_jobs=2, log_amount_of_jobs=True)\n')



get_ipython().run_cell_magic('time', '', "\n# Пример 4.3\n# Если аргументов для target_func указано МЕНЬШЕ, чем n_jobs, то используется такое же количество worker'ов, сколько было передано аргументов\nparallel_map(test_func,\n             args_container=[1, 2, 3],\n             n_jobs=5,\n             log_amount_of_jobs=True\n            )   # Здесь используется 3 worker'a\n")



get_ipython().run_cell_magic('time', '', '\n# Пример 4.4\n# Аналогичный предыдущему случай, но с именованными аргументами\nparallel_map(test_func,\n             kwargs_container=[{"s": 3}, {"s": 3}, {"s": 3}],\n             n_jobs=5,\n             log_amount_of_jobs=True\n            )   # Здесь используется 3 worker\'a\n')



get_ipython().run_cell_magic('time', '', '\n# Пример 4.5\n# Комбинация примеров 4.3 и 4.4 (переданы и позиционные и именованные аргументы)\nparallel_map(test_func,\n             args_container=[1, 2, 3],\n             kwargs_container=[{"s": 3}, {"s": 3}, {"s": 3}],\n             n_jobs=5,\n             log_amount_of_jobs=True\n            )   # Здесь используется 3 worker\'a\n')



get_ipython().run_cell_magic('time', '', "\n# Пример 4.6\n# Если аргументов для target_func указано БОЛЬШЕ, чем n_jobs, то используется n_jobs worker'ов\nparallel_map(test_func,\n             args_container=[1, 2, 3, 4],\n             kwargs_container=None,\n             n_jobs=2,\n             log_amount_of_jobs=True\n            )   # Здесь используется 2 worker'a\n")



get_ipython().run_cell_magic('time', '', '\n# Пример 4.7\n# Время выполнения оптимизируется, данный код должен отрабатывать за 5 секунд\nparallel_map(test_func,\n             kwargs_container=[{"s": 5}, {"s": 1}, {"s": 2}, {"s": 1}],\n             n_jobs=2,\n             log_amount_of_jobs=True\n            )\n')




def test_func2(string, sleep_time=1):
    time.sleep(sleep_time)
    return string

# Пример 5
# Результаты возвращаются в том же порядке, в котором были переданы соответствующие аргументы вне зависимости от того, когда завершился worker
arguments = ["first", "second", "third", "fourth", "fifth"]
parallel_map(test_func2,
             args_container=arguments,
             kwargs_container=[{"sleep_time": 5}, {"sleep_time": 4}, {"sleep_time": 3}, {"sleep_time": 2}, {"sleep_time": 1}])





get_ipython().run_cell_magic('time', '', '\n\ndef test_func3():\n    def inner_test_func(sleep_time):\n        time.sleep(sleep_time)\n        return sleep_time  # добавила, чтобы проверить, что результат тоже возвращается\n    return parallel_map(inner_test_func, args_container=[1, 2, 3])\n\ntest_func3.inner_test_func = None\n\n# Пример 6\n# Работает с функциями, созданными внутри других функций\ntest_func3()\n')



