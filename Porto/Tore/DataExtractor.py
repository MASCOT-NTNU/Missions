#!/usr/bin/env python3
__author__ = "Yaolin Ge"
__copyright__ = "Copyright 2021, The MASCOT Project, NTNU (https://wiki.math.ntnu.no/mascot)"
__credits__ = ["Yaolin Ge"]
__license__ = "MIT"
__version__ = "1.0.1"
__maintainer__ = "Yaolin Ge"
__email__ = "yaolin.ge@ntnu.no"
__status__ = "UnderDevelopment"

import numpy as np
import h5py
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'font.style': 'oblique'})
from datetime import datetime
from Porto.usr_func import *


class EXCEL2TXT:

    def __init__(self, datapath = '/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Nidelva/May27/Data/',
                 figpath = '/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Nidelva/July06/Adaptive/fig/'):
        self.datapath = datapath
        self.figpath = figpath
        print("hello world")
        pass

    def load_raw_data(self):
        # % Data extraction from the raw data
        self.rawTemp = pd.read_csv(self.datapath + "Temperature.csv", delimiter=', ', header=0, engine='python')
        self.rawLoc = pd.read_csv(self.datapath + "EstimatedState.csv", delimiter=', ', header=0, engine='python')
        self.rawSal = pd.read_csv(self.datapath + "Salinity.csv", delimiter=', ', header=0, engine='python')
        self.rawDepth = pd.read_csv(self.datapath + "Depth.csv", delimiter=', ', header=0, engine='python')
        # rawGPS = pd.read_csv(datapath + "GpsFix.csv", delimiter=', ', header=0, engine='python')
        # rawCurrent = pd.read_csv(datapath + "EstimatedStreamVelocity.csv", delimiter=', ', header=0, engine='python')
        print("Raw data is loaded successfully!")

    def group_raw_data_based_on_timestamp(self):

        # To group all the time stamp together, since only second accuracy matters
        self.rawSal.iloc[:, 0] = np.ceil(self.rawSal.iloc[:, 0])
        self.rawTemp.iloc[:, 0] = np.ceil(self.rawTemp.iloc[:, 0])
        self.rawCTDTemp = self.rawTemp[self.rawTemp.iloc[:, 2] == 78] # 'SmartX' because CTD sensor is SmartX
        self.rawLoc.iloc[:, 0] = np.ceil(self.rawLoc.iloc[:, 0])
        self.rawDepth.iloc[:, 0] = np.ceil(self.rawDepth.iloc[:, 0])
        self.rawDepth.iloc[:, 0] = np.ceil(self.rawDepth.iloc[:, 0])
        print("Raw data is grouped successfully!")

    def extract_all_data(self):
        self.depth_ctd = self.rawDepth[self.rawDepth.iloc[:, 2] == 'SmartX']["value (m)"].groupby(self.rawDepth["timestamp (seconds since 01/01/1970)"]).mean()
        self.depth_dvl = self.rawDepth[self.rawDepth.iloc[:, 2] == 'DVL']["value (m)"].groupby(self.rawDepth["timestamp (seconds since 01/01/1970)"]).mean()
        self.depth_est = self.rawLoc["depth (m)"].groupby(self.rawLoc["timestamp (seconds since 01/01/1970)"]).mean()

        # indices used to extract data
        self.lat_origin = self.rawLoc["lat (rad)"].groupby(self.rawLoc["timestamp (seconds since 01/01/1970)"]).mean()
        self.lon_origin = self.rawLoc["lon (rad)"].groupby(self.rawLoc["timestamp (seconds since 01/01/1970)"]).mean()
        self.x_loc = self.rawLoc["x (m)"].groupby(self.rawLoc["timestamp (seconds since 01/01/1970)"]).mean()
        self.y_loc = self.rawLoc["y (m)"].groupby(self.rawLoc["timestamp (seconds since 01/01/1970)"]).mean()
        self.z_loc = self.rawLoc["z (m)"].groupby(self.rawLoc["timestamp (seconds since 01/01/1970)"]).mean()
        self.depth = self.rawLoc["depth (m)"].groupby(self.rawLoc["timestamp (seconds since 01/01/1970)"]).mean()
        self.time_loc = self.rawLoc["timestamp (seconds since 01/01/1970)"].groupby(self.rawLoc["timestamp (seconds since 01/01/1970)"]).mean()
        self.time_sal= self.rawSal["timestamp (seconds since 01/01/1970)"].groupby(self.rawSal["timestamp (seconds since 01/01/1970)"]).mean()
        self.time_temp = self.rawCTDTemp["timestamp (seconds since 01/01/1970)"].groupby(self.rawCTDTemp["timestamp (seconds since 01/01/1970)"]).mean()
        self.dataSal = self.rawSal["value"].groupby(self.rawSal["timestamp (seconds since 01/01/1970)"]).mean()
        self.dataTemp = self.rawCTDTemp.iloc[:, -1].groupby(self.rawCTDTemp["timestamp (seconds since 01/01/1970)"]).mean()
        print("Data is extracted successfully!")

    def save_data(self, datapath_new):
        #% Rearrange data according to their timestamp
        self.load_raw_data()
        self.group_raw_data_based_on_timestamp()
        self.extract_all_data()
        data = []
        time_mission = []
        x = []
        y = []
        z = []
        d = []
        sal = []
        temp = []
        lat = []
        lon = []
        lat_origin = []
        lon_origin = []

        for i in range(len(self.time_loc)):
            if np.any(self.time_sal.isin([self.time_loc.iloc[i]])) and np.any(self.time_temp.isin([self.time_loc.iloc[i]])):
                time_mission.append(self.time_loc.iloc[i])
                x.append(self.x_loc.iloc[i])
                y.append(self.y_loc.iloc[i])
                z.append(self.z_loc.iloc[i])
                d.append(self.depth.iloc[i])
                lat_temp = rad2deg(self.lat_origin.iloc[i]) + rad2deg(self.x_loc.iloc[i] * np.pi * 2.0 / circumference)
                lat.append(lat_temp)
                lon.append(rad2deg(self.lon_origin.iloc[i]) + rad2deg(self.y_loc.iloc[i] * np.pi * 2.0 / (circumference * np.cos(deg2rad(lat_temp)))))
                sal.append(self.dataSal[self.time_sal.isin([self.time_loc.iloc[i]])].iloc[0])
                temp.append(self.dataTemp[self.time_temp.isin([self.time_loc.iloc[i]])].iloc[0])
                lat_origin.append(self.lat_origin.iloc[i])
                lon_origin.append(self.lon_origin.iloc[i])
            else:
                print(datetime.fromtimestamp(self.time_loc.iloc[i]))
                continue
        t1 = time.time()
        self.lat = np.array(lat).reshape(-1, 1)
        self.lon = np.array(lon).reshape(-1, 1)
        self.x = np.array(x).reshape(-1, 1)
        self.y = np.array(y).reshape(-1, 1)
        self.z = np.array(z).reshape(-1, 1)
        self.d = np.array(d).reshape(-1, 1)
        self.sal = np.array(sal).reshape(-1, 1)
        self.temp = np.array(temp).reshape(-1, 1)
        self.time_mission = np.array(time_mission).reshape(-1, 1)

        # self.datafile = h5py.File(datapath_new + 'data.h5', 'w')
        # self.datafile.create_dataset("lat", data = self.lat)
        # self.datafile.create_dataset("lon", data = self.lon)
        # self.datafile.create_dataset("x", data = self.x)
        # self.datafile.create_dataset("y", data = self.y)
        # self.datafile.create_dataset("z", data=self.z)
        # self.datafile.create_dataset("d", data=self.d)
        # self.datafile.create_dataset("salinity", data=self.sal)
        # self.datafile.create_dataset("temperature", data=self.temp)
        # self.datafile.create_dataset("timestamp", data=self.time_mission)

        self.datasheet = np.hstack((self.time_mission, self.lat, self.lon, self.x, self.y, self.z, self.d, self.sal, self.temp,
                                    np.array(lat_origin).reshape(-1, 1), np.array(lon_origin).reshape(-1, 1)))
        # np.savetxt(figpath + "../data.txt", datasheet, delimiter = ", ")
        np.savetxt(datapath_new + "data.txt", self.datasheet, delimiter=", ")

        t2 = time.time()
        print("Time consumed for creating data: ", t2 - t1, " seconds")
        print("Data is saved successfully!")

if __name__ == "__main__":
    datapath = '/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Porto/Tore/2021-11-05_douro/logs/lauv-xplore-1/20211105/Merged/mra/csv/'
    figpath = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Porto/Tore/fig/"
    datapath_new = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Porto/Tore/Data/"
    a = EXCEL2TXT(datapath, figpath)
    a.save_data(datapath_new)

    # data_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Nidelva/May27/"
    # a.save_data(data_path)

