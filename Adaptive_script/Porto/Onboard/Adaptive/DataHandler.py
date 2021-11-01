#! /usr/bin/env python3
__author__ = "Yaolin Ge"
__copyright__ = "Copyright 2021, The MASCOT Project, NTNU (https://wiki.math.ntnu.no/mascot)"
__credits__ = ["Yaolin Ge"]
__license__ = "MIT"
__version__ = "1.0.1"
__maintainer__ = "Yaolin Ge"
__email__ = "yaolin.ge@ntnu.no"
__status__ = "UnderDevelopment"


import os
import errno
from usr_func import *
# from pathlib import Path
from datetime import datetime

class DataHandler:
    data_salinity = []
    data_temperature = []
    data_path_waypoints = []
    data_timestamp = []
    path_onboard = ''

    def __init__(self):
        print("Data collector is initialised correctly")

    def append_salinity(self, value):
        DataHandler.data_salinity.append(value)

    def append_temperature(self, value):
        DataHandler.data_temperature.append(value)

    def append_path(self, value):
        DataHandler.data_path_waypoints.append(value)

    def append_timestamp(self, value):
        DataHandler.data_timestamp.append(value)

    def createDataPath(self, path_global):
        self.date_string = datetime.now().strftime("%Y_%m%d_%H%M")
        self.data_path_mission = path_global + "/Data/MASCOT/MissionData_on_" + self.date_string
        f_pre = open(path_global + "/Config/filepath_missiondata.txt", 'w')
        f_pre.write(self.data_path_mission)
        f_pre.close()
        if not os.path.exists(self.data_path_mission):
            self.mkdir_p(self.data_path_mission)
            # path = Path(self.data_path_mission)
            # path.mkdir(parents = True, exist_ok=True)
            print("New data path is created: ", self.data_path_mission)
        else:
            print("Folder is already existing, no need to create! ")

    def mkdir_p(self, path):
        try:
            os.makedirs(path)
        except OSError as exc:  # Python ≥ 2.5
            if exc.errno == errno.EEXIST and os.path.isdir(path):
                pass
            # possibly handle other errno cases here, otherwise finally:
            else:
                raise

    def save_data(self):
        self.data_salinity_saved = np.array(self.data_salinity).reshape(-1, 1)
        self.data_temperature_saved = np.array(self.data_temperature).reshape(-1, 1)
        self.data_path_waypoints_saved = np.array(self.data_path_waypoints).reshape(-1, 3)
        self.data_timestamp_saved = np.array(self.data_timestamp).reshape(-1, 1)
        np.savetxt(self.data_path_mission + "/data_salinity.txt", self.data_salinity_saved, delimiter=", ")
        np.savetxt(self.data_path_mission + "/data_temperature.txt", self.data_temperature_saved, delimiter=", ")
        np.savetxt(self.data_path_mission + "/data_path.txt", self.data_path_waypoints_saved, delimiter=", ")
        np.savetxt(self.data_path_mission + "/data_timestamp.txt", self.data_timestamp_saved, delimiter=", ")
        print("Data is saved successfully!")

    def vehpos2latlon(self, x, y, lat_origin, lon_origin):
        if lat_origin <= 10:
            lat_origin = rad2deg(lat_origin)
            lon_origin = rad2deg(lon_origin)
        lat = lat_origin + rad2deg(x * np.pi * 2.0 / circumference)
        lon = lon_origin + rad2deg(y * np.pi * 2.0 / (circumference * np.cos(deg2rad(lat))))
        return lat, lon



