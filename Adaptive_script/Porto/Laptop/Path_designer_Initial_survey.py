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
This path designer will design the path for the initial survey
'''

class PathDesigner:
    '''
    Prior1 is built based on the surface data and wind correaltion
    '''
    data_path = None
    path_onboard = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Adaptive_script/Porto/Onboard/"
    debug = True

    def __init__(self, debug = False):
        self.debug = debug
        self.load_grid()
        self.load_polygon()
        self.load_data()
        self.design_path()
        self.checkPath()

    def load_grid(self):
        print("Loading grid...")
        self.grid_poly = np.loadtxt(self.path_onboard + "grid.txt", delimiter=", ")
        print("grid is loaded successfully, grid shape: ", self.grid_poly.shape)

    def load_polygon(self):
        print("Loading the polygon...")
        self.polygon = np.loadtxt(self.path_onboard + "polygon.txt", delimiter=", ")
        print("Finished polygon loading, polygon: ", self.polygon.shape)

    def load_data(self):
        print("Loading the selected data...")
        t1 = time.time()
        self.data_path = self.path_onboard + "Prior_polygon.h5"
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

    def design_path(self):
        print("Please design the path, every second node will be pop up, use the yoyo pattern")
        # ind_grid_surface = int(len(self.grid_poly) / 3)
        close_polygon = self.polygon[0, :]
        self.polygon = np.append(self.polygon, close_polygon.reshape(1, -1), axis = 0)
        plt.figure(figsize=(10, 10))
        plt.plot(self.grid_poly[:, 1], self.grid_poly[:, 0], 'k.')
        plt.plot(self.polygon[:, 1], self.polygon[:, 0], 'r-')
        plt.xlabel("Lon [deg]")
        plt.ylabel("Lat [deg]")
        path_initialsurvey = plt.ginput(n=100, timeout = 0)  # wait for the click to select the polygon
        plt.show()
        self.path_initial_survey = []
        for i in range(len(path_initialsurvey)):
            if i % 2 == 0:
                self.path_initial_survey.append([path_initialsurvey[i][1], path_initialsurvey[i][0], 0])
            else:
                self.path_initial_survey.append([path_initialsurvey[i][1], path_initialsurvey[i][0], np.amin(self.grid_poly[:, 2])])
        self.path_initial_survey = np.array(self.path_initial_survey)
        np.savetxt(self.path_onboard + "path_initial_survey.txt", self.path_initial_survey, delimiter=", ")
        print("The initial survey path is designed successfully, path_initial_survey: ", self.path_initial_survey.shape)

    def checkPath(self):
        plt.figure(figsize=(10, 10))
        plt.plot(self.grid_poly[:, 1], self.grid_poly[:, 0], 'k.')
        plt.plot(self.polygon[:, 1], self.polygon[:, 0], 'r-')
        plt.plot(self.path_initial_survey[:, 1], self.path_initial_survey[:, 0], 'b*-')
        plt.xlabel("Lon [deg]")
        plt.ylabel("Lat [deg]")
        plt.show()


if __name__ == "__main__":
    a = PathDesigner()


