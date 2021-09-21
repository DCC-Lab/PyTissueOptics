from pytissueoptics.arrayImplementation.vectors import *
from time import time_ns

time0 = time_ns()

batches = 6
N = 500000

position = CupyVectors(N=N)
direction = CupyVectors([[0, 0, 1]]*N)
er = CupyVectors([[1, 0, 0]]*N)
weight = CupyScalars([1.0]*N)


isAlive = True
time1 = time_ns()
print(f"Initialization   ::  {(time1-time0)/1000000000}s")
time2 = time_ns()

for i in range(batches):

    time3 = time_ns()
    count = 0

    while isAlive:
        count += 1

        theta = CupyScalars.random(N=N) * 2 * np.pi
        phi = CupyScalars.random(N=N) * 2 * np.pi
        d = CupyScalars.random(N=N)

        er.rotateAround(direction, phi)
        direction.rotateAround(er, theta)
        position = position + direction*d
        weight *= 0.9
        isAlive = (weight.v > 0.001).any()

    position = CupyVectors(N=N)
    direction = CupyVectors([[0, 0, 1]] * N)
    er = CupyVectors([[1, 0, 0]] * N)
    weight = CupyScalars([1.0] * N)
    isAlive = True

    time4 = time_ns()
    print(f"Batch #{i}   ::  {(time4-time3)/1000000000}s ::  {(i+1)*N/1000000}M photons/{N*batches/1000000}M    ::  ({int((i+1)*N*100/(N*batches))}%)")

time5 = time_ns()
print(f"Calculations Only    ::  {N*batches/1000000}M photons    ::  {(time5-time2)/1000000000}s")
print(f"Complete Process    ::  {N*batches/1000000}M photons    ::  {(time5-time0)/1000000000}s")
print(f"Performances    ::  {((time5-time0)/1000)/(N*batches)} us/photon")
