
# import time
# from math import sqrt
# ndim = 1000000
# t1 = time.time()
# [sqrt(i ** 2) for i in range(ndim)]
# t2 = time.time()
# print("Time consumed: ", t2 - t1)
# from joblib import Parallel, delayed
# t1 = time.time()
# Parallel(n_jobs = 2)(delayed(sqrt)(i ** 2) for i in range(ndim))
# t2 = time.time()
# print("time consumed: ", t2 - t1)



import numpy as np
def test(t):
    print(t)
    ndim = 4000
    x = np.random.rand(ndim, ndim)
    T = np.random.rand(ndim, 1)
    # y = x ** x
    # x = np.random.rand(ndim, ndim)
    y = np.linalg.solve(x, T)
    return y

import time
t1 = time.time()
for i in range(1):
    test(1)
t2 = time.time()
print("Time consumed before multiprocessing: ", t2 - t1)

from multiprocessing import Process
# t1 = time.time()
# processes = []
# for i in range(4):
#     p = Process(target=test, args = [1])
#     processes.append(p)
#     p.start()
# for process in processes:
#     process.join()
# t2 = time.time()
# print("Time consumed after multiprocessing: ", t2 - t1)

from multiprocessing import Pool
import multiprocessing
t1 = time.time()
print("CPU number: ", multiprocessing.cpu_count())
pool = Pool(multiprocessing.cpu_count())
result = pool.map(test, [1])
t2 = time.time()
print("Time consumed after multiprocessing using pool: ", t2 - t1)



