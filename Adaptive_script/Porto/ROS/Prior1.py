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
        GridPoly.__init__(self, debug = False)
        self.data_folder = data_folder
        self.data_folder_new = data_folder_new
        self.wind_path = wind_path
        pass

    def mergedata(self):
        t1 = time.time()
        lat = []
        lon = []
        depth = []
        salinity = []
        wind_dir = []
        wind_level = []
        self.file_string = []
        for file in os.listdir(self.data_folder_new):
            if file.endswith(".h5"):
                datahandler = DataHandler_Delft3D(self.data_folder_new + file, self.wind_path, rough = True, voiceControl = False)
                self.file_string.append(file[3:7])
                # datahandler.merge_data_explicit()
                lat.append(datahandler.lat)
                lon.append(datahandler.lon)
                depth.append(datahandler.depth)
                salinity.append(datahandler.salinity)
                wind_dir.append(datahandler.wind_dir)
                wind_level.append(datahandler.wind_level)
                print(datahandler.lat.shape)
                print(datahandler.lon.shape)
                print(datahandler.depth.shape)
                print(datahandler.salinity.shape)
                print("wind_dir: ", np.array(wind_dir).shape, len(wind_dir))
                print("wind_level: ", np.array(wind_level).shape, len(wind_level))
        print("Here comes the merging!")
        self.lat_merged = np.mean(np.mean(lat, axis = 0)[:-1, :-1, :], axis = 2)
        self.lon_merged = np.mean(np.mean(lon, axis = 0)[:-1, :-1, :], axis = 2)
        self.depth_merged = depth
        self.salinity_merged = salinity
        self.wind_dir_merged = wind_dir
        self.wind_level_merged = wind_level
        t2 = time.time()
        print(t2 - t1)

    def getdata4wind(self, wind_dir, wind_level):
        print("Wind direction selected: ", wind_dir)
        print("Wind level selected: ", wind_level)
        self.row_ind = []
        self.col_ind = []
        for i in range(self.grid_poly.shape[0]):
            self.temp_latlon = (self.lat_merged - self.grid_poly[i, 0]) ** 2 + (self.lon_merged - self.grid_poly[i, 1]) ** 2
            self.row_ind.append(np.where(self.temp_latlon == np.nanmin(self.temp_latlon))[0][0])
            self.col_ind.append(np.where(self.temp_latlon == np.nanmin(self.temp_latlon))[1][0])
        self.lat_selected = self.grid_poly[:, 0] # here use the grid lat, instead of the data lat
        self.lon_selected = self.grid_poly[:, 1]
        self.depth_selected = np.empty_like(self.lat_selected)
        self.salinity_selected = np.empty_like(self.lat_selected)
        self.salinity_selected = self.salinity_selected[np.newaxis, :]
        self.depth_selected = self.depth_selected[np.newaxis, :]
        print(self.depth_selected.shape)
        print(self.salinity_selected.shape)
        length_frames = 0
        for i in range(len(self.depth_merged)):
            # self.ind_selected = (np.array(self.wind_dir_merged) == wind_dir) & (np.array(self.wind_level_merged) == wind_level)
            self.ind_selected = np.array(self.wind_dir_merged[i]) == wind_dir # only use wind_direction, since it is hard to pick both satisfying criteria
            if sum(self.ind_selected) > 0:
                print("Found ", wind_dir, wind_level, "data in " + self.file_string[i] + self.date_string + ", {:d} timeframes are used to average".format(sum(self.ind_selected)))
                self.salinity_selected = np.concatenate((self.salinity_selected, self.salinity_merged[i][self.ind_selected][:, self.row_ind, self.col_ind]), axis = 0)
                self.depth_selected = np.concatenate((self.depth_selected, np.mean(self.depth_merged[i], axis = 3)[self.ind_selected][:, self.row_ind, self.col_ind]), axis = 0)
                length_frames = length_frames + sum(self.ind_selected)
            else:
                print("Not enough data, no corresponding ", wind_dir, wind_level, "data is found in " + self.file_string[i])

        if length_frames == 0:
            print("Not engouth data")
            print("The time average for the entire month including all frames is instead used! Wind condition is ignored")
            for i in range(len(self.depth_merged)):
                self.salinity_selected = np.concatenate((self.salinity_selected, self.salinity_merged[i][:, self.row_ind, self.col_ind]), axis = 0)
                self.depth_selected = np.concatenate((self.depth_selected, np.mean(self.depth_merged[0], axis = 3)[:, self.row_ind, self.col_ind]), axis = 0)
                length_frames = length_frames + len(self.salinity_merged[i].shape[0])
            print("{:d} frames are used to find the average".format(length_frames))
        t1 = time.time()
        if os.path.exists(self.data_folder_new + "Merged/Merged_" + self.date_string + "_" + wind_dir + "_" + wind_level + ".h5"):
            print("rm -rf ../Data/Porto/D2_HDF/Merged/Merged_" + self.date_string + "_" + wind_dir + "_" + wind_level + ".h5")
            os.system("rm -rf ../Data/Porto/D2_HDF/Merged/Merged_" + self.date_string + "_" + wind_dir + "_" + wind_level + ".h5")
        data_file = h5py.File(self.data_folder_new + "Merged/Merged_" + self.date_string + "_" + wind_dir + "_" + wind_level + ".h5", 'w')
        data_file.create_dataset("lat", data = self.lat_selected)
        data_file.create_dataset("lon", data = self.lon_selected)
        data_file.create_dataset("depth", data = self.depth_selected)
        data_file.create_dataset("salinity", data = self.salinity_selected)
        t2 = time.time()
        print("Finished data creation! Time consumed: ", t2 - t1)

    def getfiles(self):
        self.FOUND = False
        for file in os.listdir(self.data_folder):
            if file.endswith(".mat"):
                if len(self.date_string) == 2:
                    if self.date_string in file[7:9]:
                        print("Found file: ")
                        print(file)
                        self.FOUND = True
                        data_mat = Mat2HDF5(self.data_folder + file, self.data_folder_new + file[:-4] + ".h5")
                        data_mat.mat2hdf()
                else:
                    if self.date_string in file[3:9]:
                        print("Found file: ")
                        print(file)
                        data_mat = Mat2HDF5(self.data_folder + file, self.data_folder_new + file[:-4] + ".h5")
                        data_mat.mat2hdf()
                        self.FOUND = True

        if not self.FOUND:
            if len(self.date_string) == 2:
                print("There is no month ", self.date_string, ", file does not exist, please check!")
            else:
                print("There is no date ", self.date_string, ", file does not exist, please check!")


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


data_folder = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Sep_Prior/"
data_folder_new = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Sep_Prior/Merged/"
wind_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Porto/Wind/wind_data.txt"

