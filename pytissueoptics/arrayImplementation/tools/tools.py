import time


def orderOfMagnitude(a, order=0):
    b = a / 10
    if b < 1:
        return order
    else:
        order += 1
        return orderOfMagnitude(b, order)

def nanoFormat(nanoTimeDelta):
    oom = orderOfMagnitude(nanoTimeDelta)
    substractor = oom % 3
    divider = oom-substractor
    prefix = ["ns", "us", "ms", "s"]
    prefixIndex = int(divider / 3)
    if prefixIndex < 0:
        prefix = "ns"
    elif prefixIndex > 3:
        prefix = "s"
    else:
        prefix = prefix[prefixIndex]
    return(nanoTimeDelta/(10**divider), prefix)


def timeit(function):
    def wrapper(*args, **kwargs):
        t0 = time.time_ns()
        function(*args, **kwargs)
        t1 = time.time_ns()
        dt, prefix = nanoFormat(t1-t0)
        print(f"{dt}{prefix} to run '{function.__name__}'")
    return wrapper