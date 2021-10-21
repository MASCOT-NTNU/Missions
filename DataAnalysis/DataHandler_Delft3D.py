# ! /usr/bin/env python3
__author__ = "Yaolin Ge"
__copyright__ = "Copyright 2021, The MASCOT Project, NTNU (https://wiki.math.ntnu.no/mascot)"
__credits__ = ["Yaolin Ge"]
__license__ = "MIT"
__version__ = "1.0.1"
__maintainer__ = "Yaolin Ge"
__email__ = "yaolin.ge@ntnu.no"
__status__ = "UnderDevelopment"


from DataAnalysis.usr_func import *
from datetime import datetime
import time
import os
import h5py
import pandas as pd


class DataHandler_Delft3D:
    data_path = None
    data = None
    wind_path = None
    wind_data = None
    figpath = None
    ROUGH = True
    voiceControl = True

    def __init__(self, datapath, windpath, rough = True, voiceControl = True):
        self.ROUGH = rough
        self.voiceControl = voiceControl
        if self.ROUGH:
            if self.voiceControl:
                os.system("say Rough mode is activated")
            print("Rough mode is activated!")
        else:
            if self.voiceControl:
                os.system("say Fine mode is activated")
            print("Fine mode is activated!")
        self.data_path = datapath
        self.wind_path = windpath
        self.load_data()
        self.load_wind()
        self.associate_data_with_wind()

    def set_figpath(self, figpath):
        self.figpath = figpath
        print("figpath is set correctly, ", self.figpath)

    def load_data(self):
        t1 = time.time()
        self.data = h5py.File(self.data_path, 'r')
        self.lat = np.array(self.data.get("lat"))
        self.lon = np.array(self.data.get("lon"))
        self.depth = np.array(self.data.get("depth"))
        self.salinity = np.array(self.data.get("salinity"))
        self.timestamp_data = np.array(self.data.get("timestamp"))
        self.string_date = datetime.fromtimestamp(self.timestamp_data[0]).strftime("%Y-%m")
        t2 = time.time()
        print("Time consumed: ", t2 - t1, " seconds")
        if self.voiceControl:
            os.system("say loading data correctly, it takes {:.1f} seconds".format(t2 - t1))
        print("Lat: ", self.lat.shape)
        print("Lon: ", self.lon.shape)
        print("Depth: ", self.depth.shape)
        print("Salinity: ", self.salinity.shape)
        print("Date: ", self.string_date)
        print("Time: ", self.timestamp_data.shape)

    def load_wind(self):
        t1 = time.time()
        if "wind_times_serie_porto_obs_2015_2020.txt" in self.wind_path:
            print("Wind data: wind_times_serie_porto_obs_2015_2020.txt")
            self.wind_data = np.array(pd.read_csv(self.wind_path, sep="\t", engine='python'))
            self.wind_data = self.wind_data[:-3, :5]
            yr_wind = self.wind_data[:, 0]
            hr_wind = self.wind_data[:, 1]
            self.timestamp_wind = []
            for i in range(len(yr_wind)):
                year = int(yr_wind[i][6:])
                month = int(yr_wind[i][3:5])
                day = int(yr_wind[i][:2])
                hour = int(hr_wind[i][:2])
                self.timestamp_wind.append(datetime(year, month, day, hour).timestamp())
            self.wind_speed = self.wind_data[:, 3]
            self.wind_maxspeed = self.wind_data[:, 4]
            self.wind_angle = self.wind_data[:, 2]
        else:
            print("Wind data: ", self.wind_path[-13:])
            self.wind_data = np.loadtxt(self.wind_path, delimiter=",")
            self.timestamp_wind = self.wind_data[:, 0]
            self.wind_speed = self.wind_data[:, 1]
            self.wind_angle = self.wind_data[:, -1]
        print("Wind data is loaded correctly!")
        print("wind_data: ", self.wind_data.shape)
        print("wind speed: ", self.wind_speed.shape)
        print("wind angle: ", self.wind_angle.shape)
        print("wind timestamp: ", self.timestamp_wind.shape)
        t2 = time.time()
        print("Time consumed: ", t2 - t1, " seconds")
        if self.voiceControl:
            os.system("say Importing wind data takes {:.1f} seconds".format(t2 - t1))

    def windangle2direction(self, wind_angle):
        angles = np.arange(8) * 45 + 22.5
        self.directions = ['NorthEast', 'East', 'SouthEast', 'South',
                           'SouthWest', 'West', 'NorthWest', 'North']
        id = len(angles[angles < wind_angle]) - 1
        return self.directions[id]

    def windspeed2level(self, wind_speed):
        speeds = np.array([0, 2.5, 10])
        self.levels = ['Mild', 'Moderate', 'Heavy']
        id = len(speeds[speeds < wind_speed]) - 1
        return self.levels[id]

    def windangle2directionRough(self, wind_angle):
        angles = np.arange(4) * 90 + 45
        self.directions = ['East', 'South', 'West', 'North']
        id = len(angles[angles < wind_angle]) - 1
        return self.directions[id]

    def windspeed2levelRough(self, wind_speed):
        speeds = np.array([0, 2.5, 6])
        self.levels = ['Mild', 'Moderate', 'Heavy']
        id = len(speeds[speeds < wind_speed]) - 1
        return self.levels[id]

    def deg2rad(self, deg):
        return deg / 180 * np.pi

    def rad2deg(self, rad):
        return rad / np.pi * 180

    def angle2angle(self, nautical_angle):
        '''
        convert nautical angle to plot angle
        '''
        return self.deg2rad(270 - nautical_angle)

    def wind2uv(self, wind_speed, wind_angle): # convert wind to uv coord
        wind_angle = self.angle2angle(wind_angle)
        u = wind_speed * np.cos(wind_angle)
        v = wind_speed * np.sin(wind_angle)
        return u, v

    def uv2wind(self, u, v): # convert uv coord to wind again
        wind_speed = np.sqrt(u ** 2 + v ** 2)
        wind_angle = self.angle2angle(self.rad2deg(np.arctan2(v, u))) # v is the speed component in y, u is the speed component in x, cartisian normal
        return wind_speed, wind_angle

    def associate_data_with_wind(self):
        self.wind_v = []
        self.wind_dir = []
        self.wind_level = []
        for i in range(len(self.timestamp_data)):
            id_wind = (np.abs(self.timestamp_wind - self.timestamp_data[i])).argmin() # find the corresponding wind data
            self.wind_v.append(self.wind_speed[id_wind])
            if self.ROUGH:
                self.wind_dir.append(self.windangle2directionRough(self.wind_angle[id_wind])) # here one can choose whether
                self.wind_level.append(self.windspeed2levelRough(self.wind_speed[id_wind])) # to use rough or not
            else:
                self.wind_dir.append(self.windangle2direction(self.wind_angle[id_wind]))
                self.wind_level.append(self.windspeed2level(self.wind_speed[id_wind]))
        print("Data is merged correctly!!")
        print("wind levels: ", len(np.unique(self.wind_level)), np.unique(self.wind_level))
        print("wind directions: ", len(np.unique(self.wind_dir)), np.unique(self.wind_dir))


if __name__ == "__main__":

    data_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Prior/Nov_Prior/Nov2019_sal_1.h5"
    wind_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Wind/wind_data.txt"
    datahandler = DataHandler_Delft3D(data_path, wind_path, rough = True)
    datahandler.set_figpath("/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Porto/Delft3D/fig/")
    # datahandler.plot_grouppeddata()
    # datahandler.plot_grid_on_data(Grid())
    # datahandler.plotscatter3D(layers=1, frame = -1)
    # datahandler.plot3Danimation()
    # datahandler.plot_surface_timeseries() # it has problems, needs to be fixed

