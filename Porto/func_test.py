import h5py
import numpy as np
import time


datapath = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Prior/Nov_Prior/Merged/Merged_North_Mild_2019_sal_2.h5"
t1 = time.time()
data = h5py.File(datapath, 'r')
lat = np.array(data.get("lat"))
lon = np.array(data.get("lon"))
depth = np.array(data.get("depth"))
salinity = np.array(data.get("salinity"))
t2 = time.time()
print("Time consuemed: ", t2 - t1)

#%%
print(lat.shape)
print(salinity.shape)
print(depth.shape)
print(lon.shape)
#%%
import matplotlib.pyplot as plt

plt.scatter(lon[:, :, 0], lat[:, :, 0], c = salinity[100, :, :, 0], cmap = "Paired")
plt.colorbar()
plt.show()
#%%

[print(np.nanmean(depth[:, :, :, i])) for i in range(depth.shape[3])]



#%%
import multiprocessing
from Porto.usr_func import test

pool = multiprocessing.Pool(4)
print(pool.map(test, [[1, 3], [2, 3], [3, 4], [3, 5], [5, 6]]))

#%%
import time
from math import sqrt
t1 = time.time()
[sqrt(i ** 2) for i in range(10)]
t2 = time.time()
print("Time consumed: ", t2 - t1)
from joblib import Parallel, delayed
t1 = time.time()
Parallel(n_jobs = 2)(delayed(sqrt)(i ** 2) for i in range(10))
t2 = time.time()
print("time consumed: ", t2 - t1)



#%%
from Porto.usr_func import daemon, non_daemon
import multiprocessing
import sys
import time

d = multiprocessing.Process(name = "Daemon", target = daemon)
d.daemon = True

n = multiprocessing.Process(name = "non-daemon", target = non_daemon)
n.daemon = False

d.start()
time.sleep(.1)
n.start()



#%%
from Porto.usr_func import worker, my_service
import multiprocessing
import time

service = multiprocessing.Process(name = "my_service", target = my_service)
worker_1 = multiprocessing.Process(name = "worker 1", target = worker)
worker_2 = multiprocessing.Process(target = worker)

worker_1.start()
worker_2.start()
service.start()

worker_1.join()
worker_2.join()
service.join()

#%%
import numpy as np
from Porto.usr_func import test
import time
ndim = 2000

t = np.random.rand(ndim, ndim)
t1 = time.time()
for i in range(4):
    np.linalg.inv(t)
t2 = time.time()
print(t2 - t1)
import multiprocessing
t1 = time.time()
processes = []
for i in range(4):
    processes.append(multiprocessing.Process(target = test, args = [t]))

for process in processes:
    process.start()

# for process in processes:
#     process.join()
t2 = time.time()
print(t2 - t1)
#%%
import numpy as np
def test():
    ndim = 3000
    x = np.random.rand(ndim, ndim)
    y = np.linalg.inv(x)
    return y

import time
t1 = time.time()
for i in range(4):
    test()
t2 = time.time()
print("Time consumed: ", t2 - t1)

from multiprocessing import Process
t1 = time.time()
processes = []
for i in range(4):
    processes.append(Process(target=test))
for process in processes:
    process.start()
for process in processes:
    process.join()
t2 = time.time()
print("Time consumed: ", t2 - t1)


#%% Test of parallisation
import multiprocessing
from multiprocessing import Pool
import time
from Porto.usr_func import f

with Pool(5) as p:
    print(p.map(f, [1, 2, 3]))

#%%
from multiprocessing import Process, Queue
from Porto.usr_func import f

colors = ["red", "green", "blue", "black"]
cnt = 1
queue = Queue()
print("pushing items to queue: ")
for color in colors:
    print("item no: ", cnt, " ", color)
    queue.put(color)
    cnt += 1

print("\npopping items from queue:")
cnt = 0
while not queue.empty():
    print("item no: ", cnt, " ", queue.get())
    cnt += 1

#%% testing of multi processing

from multiprocessing import Lock, Process, Queue, current_process
from Porto.usr_func import do_job
import time

def main():
    number_of_task = 10
    number_of_processes = 4
    tasks_to_accomplish = Queue()
    tasks_that_are_done = Queue()
    processes = []

    t1 = time.time()
    for i in range(number_of_task):
        tasks_to_accomplish.put("Task no " + str(i))
    t2 = time.time()
    print("Time consunmed: ", t2 - t1)
    # creating processes
    for w in range(number_of_processes):
        p = Process(target = do_job, args = (tasks_to_accomplish, tasks_that_are_done))
        processes.append(p)
        p.start()

    # completing process
    for p in processes:
        p.join()

    while not tasks_that_are_done.empty():
        print(tasks_that_are_done.get())

    return True

main()

#%% test of multiple processing

from multiprocessing import Process, Queue
import random
from Porto.usr_func import rand_num
queue = Queue()
processes = [Process(target = rand_num, args = ()) for x in range(4)]
for p in processes:
    p.start()

for p in processes:
    p.join()

#%%
from multiprocessing import Pool
from Porto.usr_func import my_func
import multiprocessing
pool = Pool(multiprocessing.cpu_count())
result = pool.map(my_func, [4, 2, 3, 4, 5, 23, 23])
result_set_2 = pool.map(my_func, [2, 3, 4, 5, 5, 6, 7, 78])
print(result)
print(result_set_2)


#%%
import numpy as np

def filterNaN(val):
    ncol = val.shape[1]
    temp = np.empty((0, ncol))
    for i in range(val.shape[0]):
        indicator = 0
        for j in range(val.shape[1]):
            if not np.isnan(val[i, j]):
                indicator = indicator + 1
            if indicator == ncol:
                temp = np.append(temp, val[i, :].reshape(1, -1), axis=0)
            else:
                pass
    return temp

a = np.array([[1, 2, 3], [2, 2, 3], [np.nan, 4, 5]])
print(a)

b = filterNaN(a)
print(b)


#%%
import time
from progress.bar import IncrementalBar
mylist = [1,2,3,4,5,6,7,8]
bar = IncrementalBar('Countdown', max = len(mylist))
for item in mylist:
    bar.next()
    time.sleep(1)
bar.finish()

#%%
import os
import h5py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime



data_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Delft3D/Delft3D.hdf5"
import time
os.system("say import data, long time ahead")
t1 = time.time()
Delft3D = h5py.File(data_path, 'r')
lon = np.array(Delft3D.get("lon"))
lat = np.array(Delft3D.get("lat"))
timestamp_data = np.array(Delft3D.get("timestamp"))
salinity = np.array(Delft3D.get("salinity"))
depth = np.array(Delft3D.get("depth"))
t2 = time.time()
os.system('say It takes only {:.2f} seconds to import data, congrats'.format(t2 - t1))
#%%
from Adaptive_script.Porto.Grid import Grid
a = Grid()
lat_grid = a.grid_coord[:, 0].reshape(-1, 1)
lon_grid = a.grid_coord[:, 1].reshape(-1, 1)
depth_grid = np.ones_like(lat_grid) * 1
color_grid = np.zeros_like(lat_grid)
#%%
# wind_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/conditions/wind_Era5_douro_2012_a_2016.wnd"
wind_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/conditions/wind_Era5_douro_2017_a_2019.wnd"
wind_data = np.array(pd.read_csv(wind_path, sep = "\t ", header = None, engine = 'python'))
ref_t_wind = datetime(2005, 1, 1).timestamp()
timestamp_wind = wind_data[:, 0] * 60 + ref_t_wind
wind_speed = wind_data[:, 1]
wind_angle = wind_data[:, -1]

#%%
def filterNaN(val):
    temp = []
    for i in range(len(val)):
        if not np.isnan(val[i]):
            temp.append(val[i])
    val = np.array(temp).reshape(-1, 1)
    return val


def extractDelft3DFromLocation(Delft3D, location):
    lat, lon, depth, salinity = filterNaN(Delft3D)
    print(lat.shape)
    print(lon.shape)
    print(salinity.shape)
    Sal = []
    for i in range(location.shape[0]):
        lat_desired = location[i, 0].reshape(-1, 1)
        lon_desired = location[i, 1].reshape(-1, 1)
        depth_desired = location[i, 2].reshape(-1, 1)
        print(lat_desired, lon_desired, depth_desired)
        ind = np.argmin((lat - lat_desired) ** 2 + (lon - lon_desired) ** 2)
        print(ind)
        Sal.append(salinity[ind])
        print(salinity[ind])
    Sal = np.array(Sal).reshape(-1, 1)
    return Sal, lat, lon, depth, salinity

coordinates = a.grid_coord
depth = np.ones([coordinates.shape[0], 1]) * .15
location = np.hstack((coordinates, depth))

sal, lat, lon, depth, salinity = extractDelft3DFromLocation(sal_data, location)
