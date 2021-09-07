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
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly
import plotly.express as px
import matplotlib.pyplot as plt
plotly.io.orca.config.executable = '/usr/local/bin/orca'
plotly.io.orca.config.save()
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'font.style': 'oblique'})

from Data_analysis import Mat2HDF5, DataHandler_Delft3D
from Grid import GridPoly

class DataGetter2(Mat2HDF5, DataHandler_Delft3D, GridPoly):
    '''
    Get data according to date specified and wind direction
    '''
    data_path = None
    data_path_new = None
    polygon = np.array([[41.12902, -8.69901],
                        [41.12382, -8.68799],
                        [41.12642, -8.67469],
                        [41.12071, -8.67189],
                        [41.11743, -8.68336],
                        [41.11644, -8.69869],
                        [41.12295, -8.70283],
                        [41.12902, -8.69901]])

    def __init__(self, data_path):
        GridPoly.__init__(self, polygon = DataGetter2.polygon, debug = False)
        self.plotGridonMap(self.grid_poly)
        self.data_path = data_path
        self.loaddata()
        self.select_data()
        self.save_selected_data()

    def loaddata(self):
        print("Loading the 3D data...")
        t1 = time.time()
        self.data = h5py.File(self.data_path, 'r')
        self.lat = np.array(self.data.get('lat'))
        self.lon = np.array(self.data.get('lon'))
        self.depth = np.array(self.data.get('depth'))
        self.salinity = np.array(self.data.get('salinity'))
        self.salinity_ave = np.mean(self.salinity, axis = 0)
        self.depth_ave = np.mean(self.depth, axis = 0)
        print("3D data is loaded correctly!")
        print("lat: ", self.lat.shape)
        print("lon: ", self.lon.shape)
        print("depth: ", self.depth.shape)
        print("salinity: ", self.salinity.shape)
        print("depth ave: ", self.depth_ave.shape)
        print("salinity ave: ", self.salinity_ave.shape)
        t2 = time.time()
        print("Loading data takes: ", t2 - t1)

    def select_data(self):
        self.row_ind = []
        self.col_ind = []
        t1 = time.time()
        for i in range(self.grid_poly.shape[0]):
            self.temp_latlon = (self.lat - self.grid_poly[i, 0]) ** 2 + (self.lon - self.grid_poly[i, 1]) ** 2
            self.row_ind.append(np.where(self.temp_latlon == np.nanmin(self.temp_latlon))[0][0])
            self.col_ind.append(np.where(self.temp_latlon == np.nanmin(self.temp_latlon))[1][0])
        self.lat_selected = self.lat[self.row_ind, self.col_ind, :]
        self.lon_selected = self.lon[self.row_ind, self.col_ind, :]
        self.depth_selected_ave = self.depth_ave[self.row_ind, self.col_ind, :]
        self.salinity_selected_ave = self.salinity_ave[self.row_ind, self.col_ind, :]
        self.depth_selected = self.depth[:, self.row_ind, self.col_ind, :]
        self.salinity_selected = self.salinity[:, self.row_ind, self.col_ind, :]
        t2 = time.time()
        print("lat selected: ", self.lat_selected.shape)
        print("lon selected: ", self.lon_selected.shape)
        print("depth selected: ", self.depth_selected.shape)
        print("salinity selected: ", self.salinity_selected.shape)
        print("depth selected time average: ", self.depth_selected_ave.shape)
        print("salinity selected time agerage: ", self.salinity_selected_ave.shape)
        print("time consumed: ", t2 - t1)

    def save_selected_data(self):
        t1 = time.time()
        if os.path.exists(self.data_path[:-10] + "Selected/Selected_Prior2.h5"):
            os.system("rm -rf " + self.data_path[:-10] + "Selected/Selected_Prior2.h5")
            print("File is removed: path is clean" + self.data_path[:-10] + "Selected/Selected_Prior2.h5")
        data_file = h5py.File(self.data_path[:-10] + "Selected/Selected_Prior2.h5", 'w')
        data_file.create_dataset("lat", data = self.lat_selected)
        data_file.create_dataset("lon", data = self.lon_selected)
        data_file.create_dataset("depth", data = self.depth_selected)
        data_file.create_dataset("salinity", data = self.salinity_selected)
        data_file.create_dataset("depth_ave", data=self.depth_selected_ave)
        data_file.create_dataset("salinity_ave", data=self.salinity_selected_ave)
        t2 = time.time()
        print("Finished data creation, time consumed: ", t2 - t1)

# data_path = '/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Delft3D/Delft3D.h5'
# a = DataGetter2(data_path)

class Prior2(GridPoly):
    '''
    Prior2 is build based on the 3D data forcasting, no wind data is available.
    '''
    data_path = 'Selected_Prior2.h5'
    depth_obs = [-.5, -1.25, -2.0]

    def __init__(self):
        self.loaddata()
        self.gatherData2Layers()
        pass

    def loaddata(self):
        print("Loading the 3D data...")
        t1 = time.time()
        self.data = h5py.File(self.data_path, 'r')
        self.lat = np.array(self.data.get('lat'))
        self.lon = np.array(self.data.get('lon'))
        self.depth = np.array(self.data.get('depth'))
        self.salinity = np.array(self.data.get('salinity'))
        self.salinity_ave = np.array(self.data.get('salinity_ave'))
        self.depth_ave = np.array(self.data.get('depth_ave'))
        print("3D data is loaded correctly!")
        print("lat: ", self.lat.shape)
        print("lon: ", self.lon.shape)
        print("depth: ", self.depth.shape)
        print("salinity: ", self.salinity.shape)
        print("depth ave: ", self.depth_ave.shape)
        print("salinity ave: ", self.salinity_ave.shape)
        t2 = time.time()
        print("Loading data takes: ", t2 - t1)

    def gatherData2Layers(self):
        print("Now start to gathering data...")
        self.salinity_layers_ave = np.empty((0, len(self.depth_obs)))
        self.depth_layers_ave = np.empty((0, len(self.depth_obs)))
        self.lat_layers = np.empty((0, len(self.depth_obs)))
        self.lon_layers = np.empty((0, len(self.depth_obs)))
        t1 = time.time()
        for i in range(self.lat.shape[0]):
            temp_salinity_ave = []
            temp_depth_ave = []
            temp_lat = []
            temp_lon = []
            for j in range(len(self.depth_obs)):
                ind_depth = np.abs(self.depth_ave[i, :] - self.depth_obs[j]).argmin()
                temp_salinity_ave.append(self.salinity_ave[i, ind_depth])
                temp_depth_ave.append(self.depth_ave[i, ind_depth])
                temp_lat.append(self.lat[i, ind_depth])
                temp_lon.append(self.lon[i, ind_depth])
            self.salinity_layers_ave = np.append(self.salinity_layers_ave, np.array(temp_salinity_ave).reshape(1, -1), axis = 0)
            self.depth_layers_ave = np.append(self.depth_layers_ave, np.array(temp_depth_ave).reshape(1, -1), axis = 0)
            self.lat_layers = np.append(self.lat_layers, np.array(temp_lat).reshape(1, -1), axis = 0)
            self.lon_layers = np.append(self.lon_layers, np.array(temp_lon).reshape(1, -1), axis = 0)
        t2 = time.time()
        print("Data gathered correctly, it takes ", t2 - t1)
        print("salinity: ", self.salinity_layers_ave.shape)
        print("depth: ", self.depth_layers_ave.shape)
        print("lat: ", self.lat_layers.shape)
        print("lon: ", self.lon_layers.shape)

    def getVariogram(self):
        '''
        get the coef for both lateral and vertical variogram
        '''
        from skgstat import Variogram
        x, y = self.latlon2xy(self.lat[:, 0], self.lon[:, 0], self.lat_origin, self.lon_origin)
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        self.range_coef = []
        self.sill_coef = []
        self.nugget_coef = []

        self.number_frames = 500
        if self.number_frames > self.salinity.shape[0]:
            self.number_frames = self.salinity.shape[0]
        t1 = time.time()
        for i in range(self.number_frames):
            for j in range(self.salinity_ave.shape[0]):
                self.residual = self.salinity_ave[j, :] - self.salinity[np.random.randint(0, self.salinity.shape[0]), j, :]
                V_v = Variogram(coordinates=np.hstack((np.zeros_like(self.depth_ave[j, :]).reshape(-1, 1), self.depth_ave[j, :].reshape(-1, 1))), values=self.residual, n_lags=30, maxlag=20, use_nugget=True)
                coef = V_v.cof
                self.range_coef.append(coef[0])
                self.sill_coef.append(coef[1])
                self.nugget_coef.append(coef[2])
            print(np.mean(self.range_coef), np.mean(self.sill_coef), np.mean(self.nugget_coef))
        t2 = time.time()

# a = Prior2()
# a.getVariogram()



