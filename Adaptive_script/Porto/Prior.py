#! /usr/bin/env python3
__author__ = "Yaolin Ge"
__copyright__ = "Copyright 2021, The MASCOT Project, NTNU (https://wiki.math.ntnu.no/mascot)"
__credits__ = ["Yaolin Ge"]
__license__ = "MIT"
__version__ = "1.0.1"
__maintainer__ = "Yaolin Ge"
__email__ = "yaolin.ge@ntnu.no"
__status__ = "UnderDevelopment"
#
# import numpy as np
# import os
# from Adaptive_script.Porto.GP import GaussianProcess

# class Prior(GaussianProcess):
#     AUVdata = None
#     mu_prior_sal = None
#     mu_prior_temp = None
#     Sigma_prior_sal = None
#     Sigma_prior_temp = None
#     beta0 = None
#     beta1 = None
#     SINMOD_Data = None
#     SINMOD_datapath = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Adaptive_script/samples_2020.05.01.nc"
#     if os.path.exists(SINMOD_datapath):
#         pass
#     else:
#         SINMOD_datapath = "/home/yaoling/MASCOT/adaframe/catkin_ws/src/adaframe_examples/scripts/samples_2020.05.01.nc"
#
#     def __init__(self):
#         GaussianProcess.__init__(self)
#         self.getAUVData()
#         self.getSINMODData()
#         self.getCoefficients()
#         self.get_mu_prior()
#         self.saveCoef()
#         self.print_Prior()
#
#     def print_Prior(self):
#         print("coord_grid: ", self.grid_coord.shape)
#         print("beta0: \n", self.beta0)
#         print("beta1: \n", self.beta1)
#         print("mu_sal_prior: ", self.mu_prior_sal.shape)
#         print("mu_temp_prior: ", self.mu_prior_temp.shape)
#         print("Prior is setup successfully!\n\n")
#
#     def getAUVData(self):
#         self.AUVdata = np.loadtxt('Adaptive_script/Porto/data.txt', delimiter = ",")
#
#     def getSINMODData(self):
#         import netCDF4
#         print(self.SINMOD_datapath)
#         self.SINMOD_Data = netCDF4.Dataset(self.SINMOD_datapath)
#
#     def getSINMODFromCoordsDepth(self, coordinates, depth):
#         salinity = np.mean(self.SINMOD_Data['salinity'][:, :, :, :], axis=0)
#         temperature = np.mean(self.SINMOD_Data['temperature'][:, :, :, :], axis=0) - 273.15
#         depth_sinmod = np.array(self.SINMOD_Data['zc'])
#         lat_sinmod = np.array(self.SINMOD_Data['gridLats'][:, :]).reshape(-1, 1)
#         lon_sinmod = np.array(self.SINMOD_Data['gridLons'][:, :]).reshape(-1, 1)
#         sal_sinmod = np.zeros([coordinates.shape[0], 1])
#         temp_sinmod = np.zeros([coordinates.shape[0], 1])
#
#         for i in range(coordinates.shape[0]):
#             lat, lon = coordinates[i]
#             print(np.where(np.array(depth_sinmod) == depth)[0][0])
#             ind_depth = np.where(np.array(depth_sinmod) == depth)[0][0]
#             idx = np.argmin((lat_sinmod - lat) ** 2 + (lon_sinmod - lon) ** 2)
#             sal_sinmod[i] = salinity[ind_depth].reshape(-1, 1)[idx]
#             temp_sinmod[i] = temperature[ind_depth].reshape(-1, 1)[idx]
#         return sal_sinmod, temp_sinmod
#
#     def getCoefficients(self):
#         # timestamp = self.AUVdata[:, 0].reshape(-1, 1)
#         lat_auv_origin = self.rad2deg(self.AUVdata[:, 1]).reshape(-1, 1)
#         lon_auv_origin = self.rad2deg(self.AUVdata[:, 2]).reshape(-1, 1)
#         xauv = self.AUVdata[:, 3].reshape(-1, 1)
#         yauv = self.AUVdata[:, 4].reshape(-1, 1)
#         # zauv = self.AUVdata[:, 5].reshape(-1, 1)
#         depth_auv = self.AUVdata[:, 6].reshape(-1, 1)
#         sal_auv = self.AUVdata[:, 7].reshape(-1, 1)
#         temp_auv = self.AUVdata[:, 8].reshape(-1, 1)
#         lat_auv = lat_auv_origin + self.rad2deg(xauv * np.pi * 2.0 / self.circumference)
#         lon_auv = lon_auv_origin + self.rad2deg(yauv * np.pi * 2.0 / (self.circumference * np.cos(self.deg2rad(lat_auv))))
#
#         depthl = np.array(self.depth_obs) - self.depth_tolerance
#         depthu = np.array(self.depth_obs) + self.depth_tolerance
#
#         self.beta0 = np.zeros([len(self.depth_obs), 2])
#         self.beta1 = np.zeros([len(self.depth_obs), 2])
#         sal_residual = []
#         temp_residual = []
#
#         for i in range(len(self.depth_obs)):
#             # sort out AUV data
#             ind_obs = (depthl[i] <= depth_auv) & (depth_auv <= depthu[i])
#             lat_obs = lat_auv[ind_obs].reshape(-1, 1)
#             lon_obs = lon_auv[ind_obs].reshape(-1, 1)
#             sal_obs = sal_auv[ind_obs].reshape(-1, 1)
#             temp_obs = temp_auv[ind_obs].reshape(-1, 1)
#             coord_obs = np.hstack((lat_obs, lon_obs))
#
#             # sort out SINMOD data
#             sal_sinmod, temp_sinmod = self.getSINMODFromCoordsDepth(coord_obs, self.depth_obs[i])
#
#             # compute the coef for salinity
#             sal_modelX = np.hstack((np.ones_like(sal_sinmod), sal_sinmod))
#             sal_modelY = sal_obs
#             Beta_sal = np.linalg.solve((sal_modelX.T @ sal_modelX), (sal_modelX.T @ sal_modelY))
#             # compute the coef for temperature
#             temp_modelX = np.hstack((np.ones_like(temp_sinmod), temp_sinmod))
#             temp_modelY = temp_obs
#             Beta_temp = np.linalg.solve((temp_modelX.T @ temp_modelX), (temp_modelX.T @ temp_modelY))
#
#             self.beta0[i, :] = np.hstack((Beta_sal[0], Beta_temp[0]))
#             self.beta1[i, :] = np.hstack((Beta_sal[1], Beta_temp[1]))
#
#             sal_residual.append(sal_obs - Beta_sal[0] - Beta_sal[1] * sal_sinmod)
#             temp_residual.append(temp_obs - Beta_temp[0] - Beta_temp[1] * temp_sinmod)
#
#     def get_mu_prior(self):
#         self.mu_prior_sal = []
#         self.mu_prior_temp = []
#         for i in range(len(self.depth_obs)):
#             sal_sinmod, temp_sinmod = self.getSINMODFromCoordsDepth(self.grid_coord, self.depth_obs[i])
#             self.mu_prior_sal.append(self.beta0[i, 0] + self.beta1[i, 0] * sal_sinmod)
#             self.mu_prior_temp.append(self.beta0[i, 1] + self.beta1[i, 1] * temp_sinmod)
#         self.mu_prior_sal = np.array(self.mu_prior_sal).reshape(-1, 1)
#         self.mu_prior_temp = np.array(self.mu_prior_temp).reshape(-1, 1)
#
#     def saveCoef(self):
#         np.savetxt("beta0.txt", self.beta0, delimiter=",")
#         np.savetxt("beta1.txt", self.beta1, delimiter=",")
#         np.savetxt("mu_prior_sal.txt", self.mu_prior_sal, delimiter=",")
#         np.savetxt("mu_prior_temp.txt", self.mu_prior_temp, delimiter=",")

# if __name__ == "__main__":
#     prior = Prior()
#     print("Ferdig med prior")


#%%
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
    date_string = None
    wind_path = None

    def __init__(self, data_folder, date_string, data_folder_new, wind_path):
        GridPoly.__init__(self, debug = False)
        self.data_folder = data_folder
        self.data_folder_new = data_folder_new
        self.date_string = date_string
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
        self.lat_selected = self.lat_merged[self.row_ind, self.col_ind]
        self.lon_selected = self.lon_merged[self.row_ind, self.col_ind]
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
                print("Found ", wind_dir, wind_level, "data in ," + self.file_string[i] + self.date_string + " {:d} timeframes are used to average".format(sum(self.ind_selected)))
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

data_folder = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/D2/"
data_folder_new = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/D2_HDF/"
wind_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Porto/Wind/wind_data.txt"


# a = DataGetter(data_folder, "09", data_folder_new, wind_path)
# # a.getfiles() # only used when the data file is not created
# a.mergedata()
# a.getdata4wind(wind_dir = "North", wind_level = "Calm")

class Prior1(GridPoly):
    '''
    Prior1 is built based on the surface data and wind correaltion
    '''
    data_path = None
    fig_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Porto/Setup/Region/"
    debug = True

    def __init__(self, data_folder, debug = True):
        self.debug = debug
        # GridPoly.__init__(self, debug = self.debug)
        self.data_path = data_folder + os.listdir(data_folder)[0]
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
        self.filterNaN()

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

# data_folder_merged = data_folder_new + "Merged/"
# a = Prior1(data_folder_merged, debug = False)
# a.getVariogramLateral()
# a.getData4Grid()
# a.plot_select_region()


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
    data_path = None
    depth_obs = None

    def __init__(self, data_path, depth_obs):
        self.data_path = data_path
        self.depth_obs = depth_obs
        self.loaddata()
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


data_path = '/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Delft3D/Selected/Selected_Prior2.h5'
depth_obs = [-.5, -1.25, -2.0]
a = Prior2(data_path, depth_obs)
a.gatherData2Layers()
# a.getVariogram()



