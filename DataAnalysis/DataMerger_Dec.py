

import mat73
import h5py
import os
import time
from datetime import datetime

import logging
import matplotlib.path as mplPath  # used to determine whether a point is inside the grid or not
# logging.info('So should this')
# logging.warning('And this, too')
# logging.error('And non-ASCII stuff, too, like Øresund and Malmö')

# server = True
server = False
if server:
    from usr_func import *
else:
    from DataAnalysis.usr_func import *

class DataMerger:

    data_path = None # data_path contains the path for the mat file
    depth_layers = 8 # depth layers used for merging, only down to # layers
    if server == False:
        ebb_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Tide/ebb_dec.txt"
        wind_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Wind/wind_data.txt"
        log_path = "Log/"
        data_folder = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Prior/Dec_Prior/"
        data_folder_merged = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Prior/Dec_Prior/merged_test/"
    else:
        ebb_path = "/home/ahomea/y/yaoling/MASCOT/delft3d/data/ebb_dec.txt"
        wind_path = "/home/ahomea/y/yaoling/MASCOT/delft3d/data/wind_data.txt"
        log_path = "/home/ahomea/y/yaoling/MASCOT/delft3d/log/"
        data_folder = "/home/ahomea/y/yaoling/MASCOT/delft3d/data/Dec/"
        data_folder_merged = "/home/ahomea/y/yaoling/MASCOT/delft3d/data/Dec/Merged/"
    logging.basicConfig(filename=log_path + 'datamerger.log', level=logging.DEBUG, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    logging.debug('Test')


    def __init__(self):
        print("Welcome to use data merger...")
        self.load_wind()
        self.load_ebb()

    def loaddata(self, data_path):
        '''
        This loads the original data
        '''
        self.data_path = data_path
        print("Data path is initialised successfully! ", self.data_path)
        self.file_string = data_path[-17:-8]
        print("File string: ", self.file_string)
        # os.system("say it will take more than 100 seconds to import data")
        t1 = time.time()
        self.data = mat73.loadmat(self.data_path)
        data = self.data["data"]
        self.lon = data["X"]
        self.lat = data["Y"]
        self.depth = data["Z"]
        self.Time = data['Time']
        self.timestamp_data = (self.Time - 719529) * 24 * 3600  # 719529 is how many days have passed from Jan1 0,
        # to Jan1 1970. Since 1970Jan1, is used as the starting index for datetime
        self.salinity = data["Val"]
        self.string_date = datetime.fromtimestamp(self.timestamp_data[0]).strftime("%Y_%m")
        t2 = time.time()
        print("Data is loaded correctly!")
        print("Lat: ", self.lat.shape)
        print("Lon: ", self.lon.shape)
        print("Depth: ", self.depth.shape)
        print("salinity: ", self.salinity.shape)
        print("Date: ", self.string_date)
        print("Time consumed: ", t2 - t1, " seconds.")
        # os.system("say Congrats, it takes only {:.1f} seconds to import data.".format(t2 - t1))

    def load_ebb(self):
        print("Loading ebb...")
        self.ebb = np.loadtxt(self.ebb_path, delimiter=", ")
        print("Ebb is loaded correctly!")

    def load_wind(self):
        t1 = time.time()
        print("Wind data: ", self.wind_path)
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

    def windangle2direction(self, wind_angle):
        angles = np.arange(4) * 90 + 45
        self.directions = ['East', 'South', 'West', 'North']
        id = len(angles[angles < wind_angle]) - 1
        return self.directions[id]

    def windspeed2level(self, wind_speed):
        speeds = np.array([0, 2.5, 6])
        self.levels = ['Mild', 'Moderate', 'Heavy']
        id = len(speeds[speeds < wind_speed]) - 1
        return self.levels[id]

    def angle2angle(self, nautical_angle):
        '''
        convert nautical angle to plot angle
        '''
        return deg2rad(270 - nautical_angle)

    def associate_data_with_wind(self):
        self.wind_v = []
        self.wind_dir = []
        self.wind_level = []
        for i in range(len(self.timestamp_data)):
            id_wind = (np.abs(self.timestamp_wind - self.timestamp_data[i])).argmin() # find the corresponding wind data
            self.wind_v.append(self.wind_speed[id_wind])
            self.wind_dir.append(self.windangle2direction(self.wind_angle[id_wind])) # here one can choose whether
            self.wind_level.append(self.windspeed2level(self.wind_speed[id_wind])) # to use rough or not
        print("Data is merged correctly!!")
        print("wind levels: ", len(np.unique(self.wind_level)), np.unique(self.wind_level))
        print("wind directions: ", len(np.unique(self.wind_dir)), np.unique(self.wind_dir))

    def isEbb(self, timestamp): # check if it is in phase with ebb
        if len(np.where(timestamp < self.ebb[:, 0])[0]) > 0:
            ind = np.where(timestamp < self.ebb[:, 0])[0][0] - 1 # check the index for ebb start
            if timestamp < self.ebb[ind, 1]:
                return True
            else:
                return False
        else:
            return False

    def merge_data_for_wind(self, wind_dir, wind_level):
        print("Now the merging will start...")
        self.lat_merged = self.lat[:, :, :self.depth_layers] # only extract top layers
        self.lon_merged = self.lon[:, :, :self.depth_layers]
        self.ind_selected = np.where((np.array(self.wind_dir) == wind_dir) & (np.array(self.wind_level) == wind_level))[0] # indices for selecting the time frames

        if np.any(self.ind_selected):
            self.timestamp_selected = self.timestamp_data[self.ind_selected]
            self.ind_selected_ebb = []
            print("before ebb checking: ", len(self.ind_selected))
            print("len of selected timestamp: ", len(self.timestamp_selected))
            for i in range(len(self.timestamp_selected)):
                if self.isEbb(self.timestamp_selected[i]):
                    self.ind_selected_ebb.append(self.ind_selected[i])
            print("after ebb checking: ", len(self.ind_selected_ebb))

            if len(self.ind_selected_ebb) > 0:
                # print(self.ind_selected_ebb)
                logging.info("Found " + wind_dir + wind_level + " {:d} timeframes in are used to average".format(len(self.ind_selected_ebb)))
                print("Found ", wind_dir, wind_level, " {:d} timeframes are used to average".format(len(self.ind_selected_ebb)))
                self.salinity_merged = np.mean(self.salinity[self.ind_selected_ebb, :, :, :self.depth_layers], axis = 0)
                self.depth_merged = np.mean(self.depth[self.ind_selected_ebb, :, :, :self.depth_layers], axis = 0)
                t1 = time.time()
                if os.path.exists(self.data_folder_merged + "Merged_" + wind_dir + "_" + wind_level + "_" + self.file_string + ".h5"):
                    print("rm -rf " + self.data_folder_merged + "Merged_" + wind_dir + "_" + wind_level + "_" + self.file_string + ".h5")
                    os.system("rm -rf " + self.data_folder_merged + "Merged_" + wind_dir + "_" + wind_level + "_" + self.file_string + ".h5")
                data_file = h5py.File(
                    self.data_folder_merged + "Merged_" + wind_dir + "_" + wind_level + "_" + self.file_string + ".h5", 'w')
                data_file.create_dataset("lat", data=self.lat_merged)
                data_file.create_dataset("lon", data=self.lon_merged)
                data_file.create_dataset("depth", data=self.depth_merged)
                data_file.create_dataset("salinity", data=self.salinity_merged)
                t2 = time.time()
                print("Finished data creation! Time consumed: ", t2 - t1)
        else:
            print("Not enough data, no corresponding ", wind_dir, wind_level, "data is found.")
        print("Finished merging...")

    def merge_all_data_from_file(self):
        wind_dirs = ['North', 'South', 'West', 'East'] # get wind_data for all conditions
        wind_levels = ['Mild', 'Moderate', 'Heavy'] # get data for all conditions

        counter = 0
        # os.system("say Start merging all the data")
        for file in os.listdir(self.data_folder):
            if file.endswith(".mat"):
                print(self.data_folder + file)
                self.loaddata(self.data_folder + file)
                self.associate_data_with_wind()
                for wind_dir in wind_dirs:
                    for wind_level in wind_levels:
                        print("wind_dir: ", wind_dir)
                        print("wind_level: ", wind_level)
                        t1 = time.time()
                        self.merge_data_for_wind(wind_dir, wind_level)
                        t2 = time.time()
                        # os.system("say Step {:d} of {:d}, time consumed {:.1f} seconds".format(counter, len(os.listdir(self.data_folder)) * len(wind_dirs) * len(wind_levels), t2 - t1))
                        print("Step {:d} of {:d}, time consumed {:.1f} seconds".format(counter, len(os.listdir(self.data_folder)) * len(wind_dirs) * len(wind_levels), t2 - t1))
                        counter = counter + 1

    def Average_all(self):
        # new_data_path = self.data_folder + "Merged_all/"
        wind_dirs = ['North', 'South', 'West', 'East'] # get wind_data for all conditions
        wind_levels = ['Mild', 'Moderate', 'Heavy'] # get data for all conditions
        # wind_dirs = ['North'] # get wind_data for all conditions
        # wind_levels = ['Moderate'] # get data for all conditions
        counter = 0
        for wind_dir in wind_dirs:
            for wind_level in wind_levels:
                print("wind_dir: ", wind_dir)
                print("wind_level: ", wind_level)
                self.lat_merged_all = 0
                self.lon_merged_all = 0
                self.depth_merged_all = 0
                self.salinity_merged_all = 0
                exist = False
                counter_mean = 0
                for file in os.listdir(self.data_folder_merged):
                    if "Merged_" + wind_dir + "_" + wind_level in file:
                        print(file)
                        print(counter)
                        counter = counter + 1
                        counter_mean = counter_mean + 1
                        exist = True
                        data = h5py.File(self.data_folder_merged + file, 'r')
                        self.lat_merged_all = self.lat_merged_all + np.array(data.get("lat"))
                        self.lon_merged_all = self.lon_merged_all + np.array(data.get("lon"))
                        self.depth_merged_all = self.depth_merged_all + np.array(data.get("depth"))
                        self.salinity_merged_all = self.salinity_merged_all + np.array(data.get("salinity"))
                # print("lat_merged_all: ", self.lat_merged_all.shape)
                # print("lon_merged_all: ", self.lon_merged_all.shape)
                # print("depth_merged_all:, ", self.depth_merged_all.shape)
                # print("salinity_merged_all: ", self.salinity_merged_all.shape)
                if exist:
                    print("counter_mean: ", counter_mean)
                    logging.info(wind_dir + " " + wind_level + "has " + str(counter_mean) + " frames...")
                    self.lat_mean = self.lat_merged_all / counter_mean
                    self.lon_mean = self.lon_merged_all / counter_mean
                    self.depth_mean = self.depth_merged_all / counter_mean
                    self.salinity_mean = self.salinity_merged_all / counter_mean
                    data_file = h5py.File(self.data_folder + "Merged_all/" + wind_dir + "_" + wind_level + "_all" + ".h5", 'w')
                    data_file.create_dataset("lat", data=self.lat_mean)
                    data_file.create_dataset("lon", data=self.lon_mean)
                    data_file.create_dataset("depth", data=self.depth_mean)
                    data_file.create_dataset("salinity", data=self.salinity_mean)
                else:
                    print("No data found for " + wind_dir + " " + wind_level)
                    pass

    def Compress_data(self):
        self.box = np.array([[41.0622, -8.68138],
                             [41.1419, -8.68112],
                             [41.1685, -8.74536],
                             [41.1005, -8.81627],
                             [41.042, -8.81393],
                             [41.0622, -8.68138]])
        self.polygon_box = mplPath.Path(self.box)

        self.path_merged_all = self.data_folder + "Merged_all/"
        files = os.listdir(self.path_merged_all)
        for file in files:
            if file.endswith(".h5"):
                self.filename = file
                self.data = h5py.File(self.path_merged_all + file)
                self.lat = np.array(self.data.get("lat"))
                self.lon = np.array(self.data.get("lon"))
                self.depth = np.array(self.data.get("depth"))
                self.salinity = np.array(self.data.get("salinity"))

                t1 = time.time()
                for i in range(self.lat.shape[0]):
                    for j in range(self.lat.shape[1]):
                        for k in range(self.lat.shape[2]):
                            if np.isnan(self.lat[i, j, k]) or np.isnan(self.lon[i, j, k]) or np.isnan(
                                    self.depth[i, j, k]) or np.isnan(self.salinity[i, j, k]):
                                self.lat[i, j, k] = 0
                                self.lon[i, j, k] = 0
                                self.depth[i, j, k] = 0
                                self.salinity[i, j, k] = 0
                                pass
                t2 = time.time()
                print("Filtering takes: ", t2 - t1)
                ind_matrix = np.zeros_like(self.lat[:, :, 0])
                for i in range(self.lat.shape[0]):
                    for j in range(self.lon.shape[1]):
                        if self.polygon_box.contains_point((self.lat[i, j, 0], self.lon[i, j, 0])):
                            ind_matrix[i, j] = True
                self.lat_extracted = self.lat[np.nonzero(self.lat[:, :, 0] * ind_matrix)]
                self.lon_extracted = self.lon[np.nonzero(self.lon[:, :, 0] * ind_matrix)]
                self.depth_extracted = self.depth[np.nonzero(self.depth[:, :, 0] * ind_matrix)]
                self.salinity_extracted = self.salinity[np.nonzero(self.salinity[:, :, 0] * ind_matrix)]

                t1 = time.time()
                self.path_data_save = self.data_folder + "Extracted/" + self.filename
                print(self.path_data_save)
                data_hdf = h5py.File(self.path_data_save, 'w')
                print("Finished: file creation")
                data_hdf.create_dataset("lon", data=self.lon_extracted)
                print("Finished: lon dataset creation")
                data_hdf.create_dataset("lat", data=self.lat_extracted)
                print("Finished: lat dataset creation")
                # data_hdf.create_dataset("timestamp", data=self.timestamp_data)
                # print("Finished: timestamp dataset creation")
                data_hdf.create_dataset("depth", data=self.depth_extracted)
                print("Finished: depth dataset creation")
                data_hdf.create_dataset("salinity", data=self.salinity_extracted)
                print("Finished: salinity dataset creation")
                t2 = time.time()
                print("Saving data takes: ", t2 - t1, " seconds")


    def checkMerged(self):
        path_operation_area = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Adaptive_script/Porto/Onboard/Config/OperationArea.txt"
        OpArea = np.loadtxt(path_operation_area, delimiter=", ")
        files = os.listdir(self.data_folder + "Merged_all/")
        wind_dirs = ['North', 'South', 'West', 'East']  # get wind_data for all conditions
        wind_levels = ['Mild', 'Moderate', 'Heavy']  # get data for all conditions
        plt.figure(figsize=(30, 20))
        counter_plot = 0
        for wind_level in wind_levels:
            for wind_dir in wind_dirs:
                print("wind_dir: ", wind_dir)
                print("wind_level: ", wind_level)
                for i in range(len(files)):
                    if wind_dir + "_" + wind_level in files[i]:
                        break
                plt.subplot(3, 4, counter_plot + 1)
                self.data_test = h5py.File(self.data_folder + "Merged_all/" + files[i], 'r')
                self.lat_test = np.array(self.data_test.get("lat"))
                self.lon_test = np.array(self.data_test.get("lon"))
                self.depth_test = np.array(self.data_test.get("depth"))
                self.salinity_test = np.array(self.data_test.get("salinity"))
                # print("lat_test: ", self.lat_test)
                # print("lon_test: ", self.lon_test)
                # print("depth_test: ", self.depth_test)
                # print("salinity_test: ", self.salinity_test)
                plt.scatter(self.lon_test[:, :, 0], self.lat_test[:, :, 0], c=self.salinity_test[:, :, 0], vmin=15,
                            vmax=36, cmap="Paired")
                plt.plot(OpArea[:, 1], OpArea[:, 0], 'k-.')
                plt.xlabel('Lon [deg]')
                plt.ylabel("Lat [deg]")
                plt.title(files[i])
                plt.colorbar()
                counter_plot = counter_plot + 1
        figpath = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Prior/Nov_Prior/fig/"
        plt.savefig(figpath + "prior.pdf")
        plt.show()

if __name__ == "__main__":
    a = DataMerger()
    # a.loaddata(data_path = "/Users/yaoling/Downloads/salinity_dez2017_1-005.mat")
    # a.merge_all_data_from_file()
    # a.Average_all()
    # a.Compress_data()
    a.checkMerged()
    print("Finished data merging...")


#%%
# import h5py
# import matplotlib.pyplot as plt
# # path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Prior/Nov_Prior/merged_test/Merged_North_Heavy_016_sal_1.h5"
# # # path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Prior/Nov_Prior/Merged/Merged_North_Heavy_2016_sal_1.h5"
# path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Prior/Nov_Prior/Extracted/North_Mild_all.h5"
# path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Prior/Dec_Prior/salinity_dez2016_1.mat"
# path = "/Users/yaoling/Downloads/salinity_dez2017_1-005.mat"
path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Prior/Dec_Prior/Merged_all/North_Heavy_all.h5"


# from datetime import datetime
# import mat73
# data = mat73.loadmat(path)
# data = data["data"]
# lon = data["X"]
# lat = data["Y"]
# depth = data["Z"]
# Time = data['Time']
# timestamp_data = (Time - 719529) * 24 * 3600  # 719529 is how many days have passed from Jan1 0,
# # to Jan1 1970. Since 1970Jan1, is used as the starting index for datetime
# salinity = data["Val"]
# string_date = datetime.fromtimestamp(timestamp_data[0]).strftime("%Y_%m")

data = h5py.File(path, 'r')
lat = np.array(data.get("lat"))
lon = np.array(data.get("lon"))
depth = np.array(data.get("depth"))
salinity = np.array(data.get("salinity"))

box = np.array([[41.1419, -8.68112],
[41.1685, -8.74536],
[41.1005, -8.81627],
[41.042, -8.81393],
[41.0622, -8.68138]])

plt.scatter(lon[:, :, 0], lat[:, :, 0], c = salinity[:, :, 0], cmap = "Paired", vmax = 35)
# plt.scatter(lon[:, 0], lat[:, 0], c = salinity[:, 0], cmap = "Paired", vmax = 35)
plt.plot(box[:, 1], box[:, 0])
plt.colorbar()
plt.show()

#%%
# print(lat.shape)
# print(lon.shape)
# print(depth.shape)
# print(salinity.shape)

#%%
# import matplotlib.pyplot as plt
# plt.scatter(lon[:, :, 0], lat[:, :, 0], c = salinity[0, :, :, 0], cmap = "Paired", vmax = 35)
# plt.colorbar()
# plt.show()
#%%
# import plotly.graph_objects as go
# import numpy as np
# import plotly
#
# fig = go.Figure(data=[go.Scatter3d(
#     x=lon.flatten(),
#     y=lat.flatten(),
#     z=depth.flatten(),
#     mode='markers',
#     marker=dict(
#         size=12,
#         color=salinity.flatten(),                # set color to an array/list of desired values
#         colorscale='Viridis',   # choose a colorscale
#         opacity=0.8
#     )
# )])
#
# # # tight layout
# fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
# plotly.offline.plot(fig, "test.html", auto_open=True)

# fig.show()


