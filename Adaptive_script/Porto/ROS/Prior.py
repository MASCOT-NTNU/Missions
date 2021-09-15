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


# from Adaptive_script.Porto.Grid import GridPoly
from Grid import GridPoly


class Prior(GridPoly):
    '''
    Prior1 is built based on the surface data and wind correaltion
    '''
    data_path = None
    fig_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Porto/Setup/Region/"
    debug = True
    polygon = np.array([[41.0999, -8.7283],
                        [41.1135, -8.7229],
                        [41.1143, -8.7333],
                        [41.0994, -8.7470],
                        [41.0999, -8.7283]])

    def __init__(self, debug = False):
        self.debug = debug
        GridPoly.__init__(self, polygon = self.polygon, debug = self.debug)
        self.data_path = "Prior_polygon.h5"
        # self.data_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Prior/Sep_Prior/Merged_all/North_Mild_all.h5"
        # self.prior_data_path = "Sep_Prior/"
        # self.prior_data_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Prior/Sep_Prior/"
        # self.loaddata()
        self.load_selected_data()

        # self.load_corrected_prior()
        # self.select_polygon()
        # self.extract_data_for_grid_poly()
        # self.checkPolyData()

    def loaddata(self):
        print("Loading the merged data...")
        t1 = time.time()
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

    def load_selected_data(self):
        print("Loading the merged data...")
        t1 = time.time()
        self.data = h5py.File(self.data_path, 'r')
        self.lat_selected = np.array(self.data.get('lat_selected'))
        self.lon_selected = np.array(self.data.get('lon_selected'))
        self.depth_selected = np.array(self.data.get('depth_selected'))
        self.salinity_selected = np.array(self.data.get('salinity_selected'))
        print("Merged data is loaded correctly!")
        print("lat_selected: ", self.lat_selected.shape)
        print("lon_selected: ", self.lon_selected.shape)
        print("depth_selected: ", self.depth_selected.shape)
        print("salinity_selected: ", self.salinity_selected.shape)
        t2 = time.time()
        print("Loading data takes: ", t2 - t1)

    def select_polygon(self):
        plt.figure()
        plt.scatter(self.lon[:, :, 0], self.lat[:, :, 0], c = self.salinity[:, :, 0], vmin = 15, vmax = 33, cmap = "Paired")
        plt.axvline(-8.75)
        plt.axvline(-8.70)
        plt.axhline(41.1)
        plt.axhline(41.115)
        plt.colorbar()
        plt.show()

    def extract_data_for_grid_poly(self):
        t1 = time.time()
        self.lat_selected = np.zeros([len(self.grid_poly), len(self.depth_obs)])
        self.lon_selected = np.zeros([len(self.grid_poly), len(self.depth_obs)])
        self.depth_selected = np.zeros([len(self.grid_poly), len(self.depth_obs)])
        self.salinity_selected = np.zeros([len(self.grid_poly), len(self.depth_obs)])
        self.depth_mean = np.nanmean(self.depth, axis = (0, 1)) # find the mean depth of each layer
        self.grid_coord = []
        for i in range(len(self.grid_poly)):
            for j in range(len(self.depth_obs)):
                self.grid_coord.append([self.grid_poly[i, 0], self.grid_poly[i, 1], self.depth_obs[j]])
                temp_depth = np.abs(self.depth_mean - self.depth_obs[j])
                depth_ind = np.where(temp_depth == temp_depth.min())[0][0]
                lat_diff = self.lat[:, :, depth_ind] - self.grid_poly[i, 0]
                lon_diff = self.lon[:, :, depth_ind] - self.grid_poly[i, 1]
                dist_diff = lat_diff ** 2 + lon_diff ** 2
                row_ind = np.where(dist_diff == np.nanmin(dist_diff))[0]
                col_ind = np.where(dist_diff == np.nanmin(dist_diff))[1]
                self.lat_selected[i, j] = self.grid_poly[i, 0]
                self.lon_selected[i, j] = self.grid_poly[i, 1]
                self.depth_selected[i, j] = self.depth_obs[j]
                self.salinity_selected[i, j] = self.salinity[row_ind, col_ind, depth_ind]

        print("lat_selected: ", self.lat_selected.shape)
        print("lon_selected: ", self.lon_selected.shape)
        print("depth_selected: ", self.depth_selected.shape)
        print("salinity_selected: ", self.salinity_selected.shape)
        data_file = h5py.File(self.prior_data_path + "Prior_polygon.h5", 'w')
        data_file.create_dataset("lat_selected", data = self.lat_selected)
        data_file.create_dataset("lon_selected", data = self.lon_selected)
        data_file.create_dataset("depth_selected", data = self.depth_selected)
        data_file.create_dataset("salinity_selected", data = self.salinity_selected)
        self.grid_coord = np.array(self.grid_coord)
        np.savetxt(self.prior_data_path + "grid.txt", self.grid_coord, delimiter = ", ")
        t2 = time.time()
        print("Data polygon selection is complete! Time consumed: ", t2 - t1)

    def checkPolyData(self):
        plt.figure(figsize = (20, 10))
        for i in range(self.lat_selected.shape[1]):
            plt.subplot(1, 3, i + 1)
            plt.scatter(self.lon_selected[:, i], self.lat_selected[:, i], c = self.salinity_selected[:, i], vmin = 34, vmax = 36, cmap = "Paired")
            plt.colorbar()
        plt.show()


if __name__ == "__main__":
    a = Prior()


