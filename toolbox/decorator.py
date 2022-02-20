import time
def timer(func):
    def func_wrapper(*arg, **kwargs):
        t0 = time.time()
        print('begin to run function "{}"'.format(func.__name__))
        result = func(*arg, **kwargs)
        t1 = time.time()
        t_diff = t1 - t0
        msg = 'finish function "{}", time used: {}m, {}s'
        print(msg.format(func.__name__, t_diff // 60, t_diff % 60))
        return result

    return func_wrapper