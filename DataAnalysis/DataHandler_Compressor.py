# ! /usr/bin/env python3
__author__ = "Yaolin Ge"
__copyright__ = "Copyright 2021, The MASCOT Project, NTNU (https://wiki.math.ntnu.no/mascot)"
__credits__ = ["Yaolin Ge"]
__license__ = "MIT"
__version__ = "1.0.1"
__maintainer__ = "Yaolin Ge"
__email__ = "yaolin.ge@ntnu.no"
__status__ = "UnderDevelopment"


import os
import numpy as np
import time
import h5py
import matplotlib.pyplot as plt
import matplotlib.path as mplPath  # used to determine whether a point is inside the grid or not

class DataCompressor:
    path_data = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Prior/Nov_Prior/Merged_all/"
    path_data_new = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Prior/Nov_Prior/Merged_all_extracted/"
    path_OpArea = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Adaptive_script/Porto/Onboard/Config/OperationArea.txt"

    def __init__(self):
        self.load_OpArea()

        self.get_bigger_box()
        self.load_merged_data()
        pass

    def load_OpArea(self):
        self.operational_area = np.loadtxt(self.path_OpArea, delimiter = ", ")

        # self.operational_area = self.operational_area[, :]
        print(self.operational_area)

    def get_bigger_box(self):
        self.box = np.array([[41.0622, -8.68138],
                             [41.1419, -8.68112],
                             [41.1685, -8.74536],
                             [41.1005, -8.81627],
                             [41.042, -8.81393],
                             [41.0622, -8.68138]])

        # self.box = np.array([[41.125, -8.7],
        #                      [41.126, -8.7],
        #                      [41.126, -8.71],
        #                      [41.125, -8.7]])
                             # [41.042, -8.81393],
                             # [41.0622, -8.68138]])
        # self.lat_min = np.nanmin(self.operational_area[:, 0])
        # self.lat_max = np.nanmax(self.operational_area[:, 0])
        # self.lon_min = np.nanmin(self.operational_area[:, 1])
        # self.lon_max = np.nanmax(self.operational_area[:, 1])
        # self.box = np.array([[self.lat_min, self.lon_min],
        #                      [self.lat_max, self.lon_min],
        #                      [self.lat_max, self.lon_max],
        #                      [self.lat_min, self.lon_max]])
        plt.plot(self.box[:, 1], self.box[:, 0])
        # plt.show()
        self.polygon_box = mplPath.Path(self.box)

    def filterNaN(self):
        for i in range(self.lat.shape[0]):
            for j in range(self.lat.shape[1]):
                for k in range(self.lat.shape[2]):
                    if np.isnan(self.lat[i, j, k]) or np.isnan(self.lon[i, j, k]) or np.isnan(self.depth[i, j, k]) or np.isnan(self.salinity[i, j, k]):
                        self.lat[i, j, k] = 0
                        self.lon[i, j, k] = 0
                        self.depth[i, j, k] = 0
                        self.salinity[i, j, k] = 0
                        pass

    def extract_data_inside_box(self):
        t1 = time.time()
        self.filterNaN()
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
        # for k in range(self.lat.shape[2]):
            # self.lat_extracted[:, k] = self.lat[np.nonzero(self.lat[:, :, k] * ind_matrix)]
            # self.lon_extracted[:, k] = self.lon[np.nonzero(self.lon[:, :, k] * ind_matrix)]
            # self.depth_extracted[:, k] = self.depth[np.nonzero(self.depth[:, :, k] * ind_matrix)]
            # self.salinity_extracted[:, k] = self.salinity[np.nonzero(self.salinity[:, :, k] * ind_matrix)]

    def save_extracted_data(self):
        t1 = time.time()
        self.path_data_save = self.path_data_new + self.filename
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


    def load_merged_data(self):
        files = os.listdir(self.path_data)
        print(files)

        for file in files:
            if file.endswith(".h5"):
                print(file)
                self.filename = file
                self.data = h5py.File(self.path_data + file)
                self.lat = np.array(self.data.get("lat"))
                self.lon = np.array(self.data.get("lon"))
                self.depth = np.array(self.data.get("depth"))
                self.salinity = np.array(self.data.get("salinity"))

                print(self.data)
                self.extract_data_inside_box()
                self.save_extracted_data()

                # break

if __name__ == "__main__":
    a = DataCompressor()


#%%
ind_depth = 8

import matplotlib.pyplot as plt
plt.scatter(a.lon_extracted[:, ind_depth], a.lat_extracted[:, ind_depth], c = a.salinity_extracted[:, ind_depth], vmin = 10, vmax = 35, cmap = "Paired")
plt.plot(a.operational_area[:, 1], a.operational_area[:, 0])
plt.colorbar()
plt.show()


