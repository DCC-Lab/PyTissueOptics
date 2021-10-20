import numpy as np
from numpy.random import rand
import time


# TEST 1
N = 10000000
v1 = list(rand(N))
v2 = list(rand(N))
v3 = list(rand(N))

v1n = np.array(v1)
v2n = np.array(v2)
v3n = np.array(v3)

b = 9

t0 = time.time_ns()
v3f = [v3[i] if v1[i]==v2[i] else b for i in range(len(v1))]
t1 = time.time_ns()
v4f = np.where(v1n==v2, v3, b)
t2 = time.time_ns()
v5f = []
for value1, value2, value3 in list(zip(v1, v2, v3)):
    if value1==value2:
        v5f.append(value3)
    else:
        v5f.append(b)
t3 = time.time_ns()

print("Function efficiency of (==) Operator.\n ==========")
print(f"Verification between answers is: {np.all(np.equal(v3f, v4f)) and np.all(np.equal(v4f, v5f))}")
print(f"Efficiency comparison between methods with {N:.2e} numbers")
print("List Comprehension (==):", str((t1-t0)/1000000)+"ms")
print("Numpy Where Function (==):", str((t2-t1)/1000000)+"ms")
print("Explicit loops (==):", str((t3-t2)/1000000)+"ms")


# TEST 2 (>=)
v1 = list(rand(N))
v2 = list(rand(N))
v3 = list(rand(N))

v1n = np.array(v1)
v2n = np.array(v2)
v3n = np.array(v3)

b = 9

t0 = time.time_ns()
v3f = [v3[i] if v1[i]!=v2[i] else b for i in range(len(v1))]
t1 = time.time_ns()
v4f = np.where(v1n!=v2, v3, b)
t2 = time.time_ns()
v5f = []
for value1, value2, value3 in list(zip(v1, v2, v3)):
    if value1!=value2:
        v5f.append(value3)
    else:
        v5f.append(b)
t3 = time.time_ns()

print("\nFunction efficiency of (!=) Operator.\n ==========")
print(f"Verification between answers is: {np.all(np.equal(v3f, v4f)) and np.all(np.equal(v4f, v5f))}")
print(f"Efficiency comparison between methods with {N:.2e} numbers")
print("List Comprehension (!=):", str((t1-t0)/1000000)+"ms")
print("Numpy Where Function (!=):", str((t2-t1)/1000000)+"ms")
print("Explicit loops (!=):", str((t3-t2)/1000000)+"ms")