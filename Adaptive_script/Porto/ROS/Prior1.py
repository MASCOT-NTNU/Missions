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

from Porto.Data_analysis import Mat2HDF5, DataHandler_Delft3D
from Adaptive_script.Porto.Grid import GridPoly


class DataGetter(Mat2HDF5, DataHandler_Delft3D, GridPoly):
    '''
    Get data according to date specified and wind direction
    '''
    data_folder = None
    data_folder_new = None
    wind_path = None

    def __init__(self, data_folder, data_folder_new, wind_path):
        # GridPoly.__init__(self, debug = False)
        self.depth_layers = 5 # top five layers will be chosen for saving the data, to make sure it has enough space to operate
        self.data_folder = data_folder
        self.data_folder_new = data_folder_new
        self.wind_path = wind_path
        self.mergedata()
        self.getAllPriorData()
        pass

    def mergedata(self):
        t1 = time.time()
        self.lat = [] # used to append all the data for all the months
        self.lon = []
        self.depth = []
        self.salinity = []
        self.wind_dir = []
        self.wind_level = []
        self.file_string = []
        counter = 0
        for file in os.listdir(self.data_folder):
            if file.endswith(".h5"):
                print(file)
                datahandler = DataHandler_Delft3D(self.data_folder + file, self.wind_path, rough = True, voiceControl = False)
                self.file_string.append(file[3:7])
                self.lat.append(datahandler.lat)
                self.lon.append(datahandler.lon)
                self.depth.append(datahandler.depth)
                self.salinity.append(datahandler.salinity)
                self.wind_dir.append(datahandler.wind_dir)
                self.wind_level.append(datahandler.wind_level)
                print("lat: ", datahandler.lat.shape)
                print("lon: ", datahandler.lon.shape)
                print("depth: ", datahandler.depth.shape)
                print("salinity", datahandler.salinity.shape)
                print("wind_dir: ", np.array(self.wind_dir[counter]).shape, len(self.wind_dir))
                print("wind_level: ", np.array(self.wind_level[counter]).shape, len(self.wind_level))
                counter = counter + 1
        self.lat_merged = np.mean(self.lat, axis = 0)[:, :, :self.depth_layers] # merged lat, with one fixed dimension, 410, 260, 5
        self.lon_merged = np.mean(self.lon, axis = 0)[:, :, :self.depth_layers] # merged lon, with one fixed dimension, 410, 260, 5
        print("lat_merged: ", self.lat_merged.shape)
        print("lon_merged: ", self.lon_merged.shape)
        t2 = time.time()
        print(t2 - t1)

    def getdata4wind(self, wind_dir, wind_level):
        print("Wind direction selected: ", wind_dir)
        print("Wind level selected: ", wind_level)
        length_frames = 0
        self.salinity_merged = np.empty_like(self.lat_merged)
        self.depth_merged = np.empty_like(self.lon_merged)
        print("Before adding new axis")
        print("salinity_merged: ", self.salinity_merged.shape)
        print("depth_merged: ", self.depth_merged.shape)
        self.salinity_merged = self.salinity_merged[np.newaxis, :]
        self.depth_merged = self.depth_merged[np.newaxis, :]
        print("After adding new axis")
        print("salinity_merged: ", self.salinity_merged.shape)
        print("depth_merged: ", self.depth_merged.shape)
        for i in range(len(self.depth)):
            self.ind_selected = (np.array(self.wind_dir) == wind_dir) & (np.array(self.wind_level) == wind_level) # indices for selecting the time frames
            # self.ind_selected = np.array(self.wind_dir_merged[i]) == wind_dir # only use wind_direction, since it is hard to pick both satisfying criteria
            if np.any(self.ind_selected):
                print("Found ", wind_dir, wind_level, " {:d} timeframes are used to average".format(np.sum(self.ind_selected)))
                self.salinity_merged = np.concatenate((self.salinity_merged, self.salinity[i][self.ind_selected[i]][:, :, :, :self.depth_layers]), axis = 0)
                self.depth_merged = np.concatenate((self.depth_merged, self.depth[i][self.ind_selected[i]][:, :, :, :self.depth_layers]), axis = 0)
                length_frames = length_frames + sum(self.ind_selected)
            else:
                print("Not enough data, no corresponding ", wind_dir, wind_level, "data is found in " + self.file_string[i])

        t1 = time.time()
        if os.path.exists(self.data_folder_new + "Merged_" + wind_dir + "_" + wind_level + ".h5"):
            print("rm -rf " + self.data_folder_new + "Merged_" + wind_dir + "_" + wind_level + ".h5")
            os.system("rm -rf " + self.data_folder_new + "Merged_" + wind_dir + "_" + wind_level + ".h5")
        data_file = h5py.File(self.data_folder_new + "Merged_" + wind_dir + "_" + wind_level + ".h5", 'w')
        data_file.create_dataset("lat", data = self.lat_merged)
        data_file.create_dataset("lon", data = self.lon_merged)
        data_file.create_dataset("depth", data = self.depth_merged)
        data_file.create_dataset("salinity", data = self.salinity_merged)
        t2 = time.time()
        print("Finished data creation! Time consumed: ", t2 - t1)

    def getAllPriorData(self):
        wind_dirs = ['East', 'South', 'West', 'North'] # get wind_data for all conditions
        wind_levels = ['Mild', 'Moderate', 'Heavy'] # get data for all conditions
        # wind_dirs = ['East'] # get wind_data for all conditions
        # wind_levels = ['Mild'] # get data for all conditions
        for wind_dir in wind_dirs:
            for wind_level in wind_levels:
                self.getdata4wind(wind_dir, wind_level)



class Prior(GridPoly):
    '''
    Prior1 is built based on the surface data and wind correaltion
    '''
    data_path = None
    fig_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Porto/Setup/Region/"
    debug = True

    def __init__(self, debug = False):
        self.debug = debug
        GridPoly.__init__(self, debug = self.debug)
        self.data_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/D2_HDF/Merged/Merged_09_North_Calm.h5"
        self.loaddata()

    def loaddata(self):
        print("Loading the merged data...")
        t1 = time.time()
        self.data = h5py.File(self.data_path, 'r')
        self.lat = np.array(self.data.get('lat'))
        self.lon = np.array(self.data.get('lon'))
        self.depth = np.array(self.data.get('depth'))
        self.salinity = np.array(self.data.get('salinity'))
        self.salinity_ave = np.mean(self.salinity, axis = 0)
        self.depth_ave = np.mean(self.depth, axis = 0)
        print("Merged data is loaded correctly!")
        print("lat: ", self.lat.shape)
        print("lon: ", self.lon.shape)
        print("depth: ", self.depth.shape)
        print("salinity: ", self.salinity.shape)
        print("depth ave: ", self.depth_ave.shape)
        print("salinity ave: ", self.salinity_ave.shape)
        t2 = time.time()
        print("Loading data takes: ", t2 - t1)
        # self.filterNaN()

    def filterNaN(self):
        self.lat_filtered = np.empty((0, 1))
        self.lon_filtered = np.empty((0, 1))
        self.depth_filtered = np.empty((0, 1))
        self.salinity_filtered = np.empty((0, 1))
        print("Before filtering!")
        print("lat: ", self.lat.shape)
        print("lon: ", self.lon.shape)
        print("depth: ", self.depth.shape)
        print("salinity: ", self.salinity.shape)
        for i in range(len(self.lat)):
            if np.isnan(self.lat[i]) or np.isnan(self.lon[i]) or np.isnan(self.depth_ave[i]) or np.isnan(self.salinity_ave[i]):
                pass
            else:
                self.lat_filtered = np.append(self.lat_filtered, self.lat[i])
                self.lon_filtered = np.append(self.lon_filtered, self.lon[i])
                self.depth_filtered = np.append(self.depth_filtered, self.depth_ave[i])
                self.salinity_filtered = np.append(self.salinity_filtered, self.salinity_ave[i])
        print("Filtered correctly:")
        print("lat: ", self.lat_filtered.shape)
        print("lon: ", self.lon_filtered.shape)
        print("depth: ", self.depth_filtered.shape)
        print("salinity: ", self.salinity_filtered.shape)

    def getVariogramLateral(self):
        from skgstat import Variogram
        x, y = self.latlon2xy(self.lat, self.lon, self.lat_origin, self.lon_origin)
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        self.range_coef = []
        self.sill_coef = []
        self.nugget_coef = []
        self.number_frames = 1000
        t1 = time.time()
        for i in range(self.number_frames):
            print(i)
            residual = self.salinity_ave - self.salinity[np.random.randint(0, self.salinity.shape[0]), :]
            V_v = Variogram(coordinates=np.hstack((x, y)), values=residual, n_lags=20, maxlag=3000, use_nugget=True)
            coef = V_v.cof
            if i == 500:
                fig = V_v.plot()
                figpath = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Presentation/MASCOT/Sept6/fig/"
                fig.savefig(figpath + "variogram.pdf")
            self.range_coef.append(coef[0])
            self.sill_coef.append(coef[1])
            self.nugget_coef.append(coef[2])
        t2 = time.time()
        print(t2 - t1)
        print(sum(self.range_coef)/len(self.range_coef), sum(self.sill_coef)/len(self.sill_coef), sum(self.nugget_coef)/len(self.nugget_coef))

    def plot_select_region(self):
        # fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scene'}]])
        fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scene'}]])
        fig.add_trace(
            go.Scatter3d(
                x=self.lon_filtered.flatten(), y=self.lat_filtered.flatten(), z=np.zeros_like(self.lat_filtered.flatten()),
                mode='markers',
                marker=dict(
                    size=4,
                    color=self.salinity_filtered.flatten(),
                    colorscale=px.colors.qualitative.Light24,
                    showscale=True
                ),
            ),
            row=1, col=1,
        )
        fig.update_layout(
            scene={
                'aspectmode': 'manual',
                'xaxis_title': 'Lon [deg]',
                'yaxis_title': 'Lat [deg]',
                'zaxis_title': 'Depth [m]',
                'aspectratio': dict(x=1, y=1, z=.5),
            },
            showlegend=False,
            title="Prior"
        )
        plotly.offline.plot(fig, filename=self.fig_path + "Prior1.html",
                            auto_open=True)


data_folder = "/Volumes/Extreme SSD/2021/"
data_folder_new = "/Volumes/Extreme SSD/2021/Merged/"
wind_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Porto/Wind/wind_data.txt"

# if __name__ == "__main__":
a = DataGetter(data_folder, data_folder_new, wind_path)


