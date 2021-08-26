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
