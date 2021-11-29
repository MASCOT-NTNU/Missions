import multiprocessing
from usr_func import *


a_pool = multiprocessing.Pool(processes=10)


result = a_pool.map(sum_up_to, range(10))

res = []
ndim = 40000
import time
t1 = time.time()
for i in range(ndim):
    res.append(sum_up_to(i))
t2 = time.time()
print("Time consumed: ", t2 - t1)

t1 = time.time()
result = a_pool.map(sum_up_to, range(ndim))
t2 = time.time()
print("Time consumed: ", t2 - t1)
