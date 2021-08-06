import numpy as np
import matplotlib.pyplot as plt
from usr_func import *

data = np.loadtxt("data.txt", delimiter = ",")
timestamp = data[:, 0].reshape(-1, 1)
lat_auv_origin = rad2deg(data[:, 1]).reshape(-1, 1)
lon_auv_origin = rad2deg(data[:, 2]).reshape(-1, 1)
xauv = data[:, 3].reshape(-1, 1)
yauv = data[:, 4].reshape(-1, 1)
zauv = data[:, 5].reshape(-1, 1)
depth_auv = data[:, 6].reshape(-1, 1)
sal_auv = data[:, 7].reshape(-1, 1)
temp_auv = data[:, 8].reshape(-1, 1)


depth_obs = [0.5, 1.0, 1.5, 2.0, 2.5]  # planned depth to be observed
depth_tolerance = .25

lat_auv = lat_auv_origin + rad2deg(xauv * np.pi * 2.0 / circumference)
lon_auv = lon_auv_origin + rad2deg(yauv * np.pi * 2.0 / (circumference * np.cos(deg2rad(lat_auv))))

depthl = np.array(depth_obs) - depth_tolerance
depthu = np.array(depth_obs) + depth_tolerance

beta0 = np.zeros([len(depth_obs), 2])
beta1 = np.zeros([len(depth_obs), 2])
sal_residual = []
temp_residual = []
x_loc = []
y_loc = []
SINMOD_datapath = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Adaptive_script/samples_2020.05.01.nc"
import netCDF4
SINMOD_Data = netCDF4.Dataset(SINMOD_datapath)

for i in range(len(depth_obs)):
    # sort out AUV data
    ind_obs = (depthl[i] <= depth_auv) & (depth_auv <= depthu[i])
    lat_obs = lat_auv[ind_obs].reshape(-1, 1)
    lon_obs = lon_auv[ind_obs].reshape(-1, 1)
    sal_obs = sal_auv[ind_obs].reshape(-1, 1)
    temp_obs = temp_auv[ind_obs].reshape(-1, 1)

    # sort out SINMOD data
    salinity = np.mean(SINMOD_Data['salinity'][:, :, :, :], axis=0)  # time averaging of salinity
    temperature = np.mean(SINMOD_Data['temperature'][:, :, :, :], axis=0) - 273.15  # time averaging of temperature
    depth_sinmod = np.array(SINMOD_Data['zc'])  # depth from SINMOD
    lat_sinmod = np.array(SINMOD_Data['gridLats'][:, :]).reshape(-1, 1)  # lat from SINMOD
    lon_sinmod = np.array(SINMOD_Data['gridLons'][:, :]).reshape(-1, 1)  # lon from SINMOD
    sal_sinmod = np.zeros([sal_obs.shape[0], 1])
    temp_sinmod = np.zeros([temp_obs.shape[0], 1])

    for j in range(sal_obs.shape[0]):
        # print(depth_obs[i])
        ind_depth = np.where(np.array(depth_sinmod) == depth_obs[i])[0][0]
        # print(ind_depth)
        idx = np.argmin((lat_sinmod - lat_obs[j]) ** 2 + (lon_sinmod - lon_obs[j]) ** 2)
        sal_sinmod[j] = salinity[ind_depth].reshape(-1, 1)[idx]
        temp_sinmod[j] = temperature[ind_depth].reshape(-1, 1)[idx]
    # print(sal_sinmod.shape)

    X = np.hstack((np.ones_like(sal_sinmod), sal_sinmod))
    Beta = np.linalg.solve((X.T @ X), X.T @ sal_obs)
    print(Beta)

