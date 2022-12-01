
# # 1

def sequential_map(*args):
    l = len(args) - 1
    c = args[-1]
    for i in args[0:l]:
        c = i(c)
    return list(c)


# # 2

def consensus_filter(*args):
    l = len(args) - 1
    c = args[-1]
    
    for i in args[0:l]:
        cnt = -1
        res = [i(x) for x in c]
        for j in res:
            cnt += 1
            if not j:
                c.remove(c[cnt])
                cnt -= 1
    return c


# # 3

def conditional_reduce(*args):
    func_1 = args[0]
    func_2 = args[1]
    cont = args[2]
    
    res = [func_1(x) for x in cont]
    cnt = -1
    for i in res:
        cnt += 1
        if not i:
            val = cont[cnt]
            cont.remove(val)
            cnt -= 1
    
    el = cont[0]
    for j in range(1, len(cont)):
        el = func_2(el, cont[j])
    return el


# # 4

def func_chain(*args):
    new_func = lambda prev_f: (f(prev_f) for f in args)
    def res_func(value):
        result_chain = new_func(value) # вызов каждой из функций на значении value
        result = None
        for r in result_chain:
            result = r
        return result
    return res_func


# # 5

def multiple_partial(*args, **kwargs):
    
    def my_partial(func, **kwargs):
        def new_func(*new_args, **new_kwargs):
            return func(*new_args, **new_kwargs, **kwargs)
        return new_func
    
    result = []
    for i in args:
        result.append(
            my_partial(i, **kwargs)
        )
    return result


# # 6

import sys

def my_print(*args):
    for a in args:
        sys.stdout.write(str(a) + " ")


