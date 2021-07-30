import numpy as np


def averageposition(object, position):
    elemsum = 0
    elemsumposition = 0

    # i, j, k = np.argmax(object)
    # x = a/object.shape[0] + b
    # y =
    # z =...

    for i in range(object.shape[0]):
        for j in range(object.shape[1]):
            for k in range(object.shape[2]):
                energy = object[i, j, k]
                elemsum += energy
                if position == 0:
                    stats = None
                    a = 8  # stats.max[0] - stats.min[0]
                    b = -4  # stats.min[0]
                    elemsumposition += energy * (i * a / object.shape[0] + b)
                elif position == 1:
                    a = 8  # stats.max[0] - stats.min[0]
                    b = -4  # stats.min[0]
                    elemsumposition += energy * (j * a / object.shape[0] + b)
                else:
                    a = 20  # stats.max[0] - stats.min[0]
                    b = 0  # stats.min[0]
                    elemsumposition += energy * (k * a / object.shape[0] + b)

    if elemsum != 0:
        averageposition = elemsumposition / elemsum
    else:
        averageposition = None

    return averageposition


def averagepositionSquare(object, position):
    elemsum = 0
    elemsumpositionSquare = 0

    for i in range(object.shape[0]):
        for j in range(object.shape[1]):
            for k in range(object.shape[2]):

                elemsum += object[i, j, k]
                if position == 0:
                    elemsumpositionSquare += object[i, j, k] * i * i
                elif position == 1:
                    elemsumpositionSquare += object[i, j, k] * j * j
                else:
                    elemsumpositionSquare += object[i, j, k] * k * k

    if elemsum != 0:
        averagepositionSquare = elemsumpositionSquare / elemsum
    else:
        averagepositionSquare = None

    return averagepositionSquare


def rmsposition(object, position):
    avgIdx = averageposition(object, position)
    avgIdx2 = averagepositionSquare(object, position)

    if avgIdx is not None and avgIdx2 is not None:
        rms = np.sqrt(avgIdx2 - avgIdx * avgIdx)
    else:
        rms = None
    return rms


def rmspositiones(object):
    return rmsposition(object, position=0), rmsposition(object, position=1), rmsposition(object, position=2)


if __name__ == "__main__":
    good_representation_of_volume = (1, 1, 1)
    array = np.ones(shape=(3, 3, 3))
    di, dj, dk = good_representation_of_volume
    print("∆i = {0}, ∆j = {1}, ∆k = {2}".format(di, dj, dk))

    print(averageposition(array, position=0))
    print(averageposition(array, position=1))
    print(averageposition(array, position=2))

    print(averagepositionSquare(array, position=0))
    print(averagepositionSquare(array, position=1))
    print(averagepositionSquare(array, position=2))

    iavg = averageposition(array, position=0)
    i2avg = averagepositionSquare(array, position=0)

    delta = np.sqrt(i2avg - iavg * iavg)
    print(delta, averageposition)

