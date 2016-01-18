def timer(f):
    import functools

    @functools.wraps(f)
    def f2(*args, **kwargs):
        import time
        import inspect
        t0 = time.time()
        result = f(*args, **kwargs)
        t1 = time.time()
        fname = inspect.stack()[1][4][0].split('(')[0].strip()
        print 'time for %s = %.2f' % (fname, t1-t0)
        return result
    return f2
