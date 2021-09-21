#! /usr/bin/env python3
__author__ = "Yaolin Ge"
__copyright__ = "Copyright 2021, The MASCOT Project, NTNU (https://wiki.math.ntnu.no/mascot)"
__credits__ = ["Yaolin Ge"]
__license__ = "MIT"
__version__ = "1.0.1"
__maintainer__ = "Yaolin Ge"
__email__ = "yaolin.ge@ntnu.no"
__status__ = "UnderDevelopment"

import time
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'font.style': 'oblique'})

'''
generate the prior based on the selected polygon and grid
'''

class DelftPrior:
    '''
    Prior1 is built based on the surface data and wind correaltion
    '''
    data_path = None
    fig_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Porto/Setup/Region/"
    path_onboard = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Adaptive_script/Porto/Onboard/"
    path_delft3d = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Prior/Sep_Prior/Merged_all/"
    debug = True

    def __init__(self, debug = False):
        self.debug = debug
        self.load_windcondition()
        self.load_grid()
        self.loaddata()
        self.select_data()

    def loaddata(self):
        print("Loading the merged data...")
        t1 = time.time()
        self.data_path = self.path_delft3d + self.wind_dir + "_" + self.wind_level + "_all.h5"
        self.data = h5py.File(self.data_path, 'r')
        self.lat = np.array(self.data.get('lat'))
        self.lon = np.array(self.data.get('lon'))
        self.depth = np.array(self.data.get('depth'))
        self.salinity = np.array(self.data.get('salinity'))
        print("Merged data is loaded correctly!")
        print("lat: ", self.lat.shape)
        print("lon: ", self.lon.shape)
        print("depth: ", self.depth.shape)
        print("salinity: ", self.salinity.shape)
        t2 = time.time()
        print("Loading data takes: ", t2 - t1)

    def load_grid(self):
        print("Loading grid...")
        self.grid_poly = np.loadtxt(self.path_onboard + "grid.txt", delimiter=", ")
        print("grid is loaded successfully, grid shape: ", self.grid_poly.shape)

    def load_windcondition(self):
        print("It will load the wind conditions...")
        f_wind = open(self.path_onboard + "wind_condition.txt", 'r')
        s = f_wind.read()
        ind_wind_dir = s.index("wind_dir=")
        ind_wind_level = s.index(", wind_level=")
        self.wind_dir = s[ind_wind_dir + 9:ind_wind_level]
        self.wind_level = s[ind_wind_level + 13:]
        print("wind_dir: ", self.wind_dir)
        print("wind_level: ", self.wind_level)

    def select_data(self):
        t1 = time.time()
        self.lat_selected = np.zeros([len(self.grid_poly), 1])
        self.lon_selected = np.zeros([len(self.grid_poly), 1])
        self.depth_selected = np.zeros([len(self.grid_poly), 1])
        self.salinity_selected = np.zeros([len(self.grid_poly), 1])
        self.depth_mean = np.nanmean(self.depth, axis = (0, 1)) # find the mean depth of each layer from delft3d
        self.grid_coord = []
        for i in range(len(self.grid_poly)):
            temp_depth = np.abs(self.depth_mean - self.grid_poly[i, 2])
            depth_ind = np.where(temp_depth == temp_depth.min())[0][0]
            lat_diff = self.lat[:, :, depth_ind] - self.grid_poly[i, 0]
            lon_diff = self.lon[:, :, depth_ind] - self.grid_poly[i, 1]
            dist_diff = lat_diff ** 2 + lon_diff ** 2
            row_ind = np.where(dist_diff == np.nanmin(dist_diff))[0]
            col_ind = np.where(dist_diff == np.nanmin(dist_diff))[1]
            self.lat_selected[i] = self.grid_poly[i, 0]
            self.lon_selected[i] = self.grid_poly[i, 1]
            self.depth_selected[i] = self.grid_poly[i, 2]
            self.salinity_selected[i] = self.salinity[row_ind, col_ind, depth_ind]

        self.dataset_prior = np.hstack((self.lat_selected.reshape(-1, 1),
                                        self.lon_selected.reshape(-1, 1),
                                        self.depth_selected.reshape(-1, 1),
                                        self.salinity_selected.reshape(-1, 1)))
        np.savetxt(self.path_onboard + "Prior_polygon.txt", self.dataset_prior, delimiter=", ")
        t2 = time.time()
        print("Data polygon selection is complete! Time consumed: ", t2 - t1)

    def checkPolyData(self):
        plt.figure(figsize = (20, 10))
        for i in range(self.lat_selected.shape[1]):
            plt.subplot(1, 3, i + 1)
            plt.scatter(self.lon_selected[:, i], self.lat_selected[:, i], c = self.salinity_selected[:, i], vmin = 26, vmax = 36, cmap = "Paired")
            plt.colorbar()
        plt.show()

if __name__ == "__main__":
    a = DelftPrior()
    # a.checkPolyData()


