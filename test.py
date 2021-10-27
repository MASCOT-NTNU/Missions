import mat73
import time
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime


path = "/home/ahomea/y/yaoling/MASCOT/Porto_Data_Processing/Data/"
# path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Prior/Nov_Prior/"
path_wind = "wind_data.txt"
path_tide = "tide.txt"
# path_wind = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Wind/wind_data.txt"
# path_tide = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Tide/tide.txt"
figpath = "/home/ahomea/y/yaoling/MASCOT/Porto_Data_Processing/Data/fig/"
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'font.style': 'oblique'})
lat_river_mouth, lon_river_mouth = 41.139024, -8.680089


wind = np.loadtxt(path_wind, delimiter=",")
tide = np.loadtxt(path_tide, delimiter=", ")

def angle2anle(angle):
    return 270 - angle

def s2uv(s, angle):
    angle = angle2anle(angle)
    u = s * np.cos(deg2rad(angle))
    v = s * np.sin(deg2rad(angle))
    return u, v

def rad2deg(rad):
    return rad / np.pi * 180

def deg2rad(deg):
    return deg / 180 * np.pi

counter = 0
files = os.listdir(path)
for file in files:
    if file.endswith(".mat"):
        print(file)
        t1 = time.time()
        data = mat73.loadmat(path + file)
        data = data["data"]
        lon = data["X"]
        lat = data["Y"]
        depth = data["Z"]
        Time = data['Time']
        timestamp_data = (Time - 719529) * 24 * 3600  # 719529 is how many days have passed from Jan1 0,
        # to Jan1 1970. Since 1970Jan1, is used as the starting index for datetime
        sal_data = data["Val"]
        string_date = datetime.fromtimestamp(timestamp_data[0]).strftime("%Y_%m")
        t2 = time.time()
        print("loading takes: ", t2 - t1, " seconds")
        for i in range(sal_data.shape[0]):
            plt.figure(figsize=(10, 10))
            im = plt.scatter(lon[:, :, 0], lat[:, :, 0], c=sal_data[i, :, :, 0], vmin=15, vmax=36, cmap="Paired")
            dist_wind = np.abs(timestamp_data[i] - wind[:, 0])
            ind_wind = np.where(dist_wind == np.nanmin(dist_wind))[0][0]

            dist_tide = np.abs(timestamp_data[i] - tide[:, 0])
            ind_tide = np.where(dist_tide == np.nanmin(dist_tide))[0][0]

            u, v = s2uv(wind[ind_wind, 1], wind[ind_wind, 2])

            plt.quiver(lon_river_mouth, lat_river_mouth, u, v)
            if tide[ind_tide, 2] == 1:
                if timestamp_data[i] >= tide[ind_tide, 0]:
                    plt.quiver(-8.65, 41.2, -1, 0)
                else:
                    plt.quiver(-8.65, 41.2, 1, 0)
            else:
                if timestamp_data[i] >= tide[ind_tide, 0]:
                    plt.quiver(-8.65, 41.2, 1, 0)
                else:
                    plt.quiver(-8.65, 41.2, -1, 0)
                # plt.quiver(-8.65, 41.2, -1, 0)
            plt.xlabel("Lon [deg]")
            plt.ylabel("Lat [deg]")
            plt.title("Surface salinity on " + datetime.fromtimestamp(timestamp_data[i]).strftime("%Y%m%d - %H:%M"))
            plt.colorbar(im)
            plt.savefig(figpath + "I_{:05d}.png".format(counter))
            counter = counter + 1
            print(counter)
            plt.close("all")
            # plt.show()

