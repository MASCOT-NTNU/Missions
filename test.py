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

print(lat.shape)
print(lon.shape)
print(depth.shape)
print(salinity.shape)

import matplotlib.pyplot as plt
plt.scatter(lon[:, :, 0], lat[:, :, 0], c = salinity[:, :, 0], cmap = "Paired")
plt.colorbar()
plt.show()

