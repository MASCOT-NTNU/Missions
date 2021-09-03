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

from Adaptive_script.Porto.Grid import GridPoly

class Prior1(GridPoly):
    '''
    Prior1 is built based on the surface data and wind correaltion
    '''
    data_path = None
    fig_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Porto/Setup/Region/"
    debug = True

    def __init__(self, data_folder, debug = True):
        self.debug = debug
        GridPoly.__init__(self, debug = self.debug)
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
        for i in range(self.lat.shape[0]):
            for j in range(self.lat.shape[1]):
                if np.isnan(self.lat[i, j]) or np.isnan(self.lon[i, j]) or np.isnan(self.depth_ave[i, j]) or np.isnan(self.salinity_ave[i, j]):
                    pass
                else:
                    self.lat_filtered = np.append(self.lat_filtered, self.lat[i, j])
                    self.lon_filtered = np.append(self.lon_filtered, self.lon[i, j])
                    self.depth_filtered = np.append(self.depth_filtered, self.depth_ave[i, j])
                    self.salinity_filtered = np.append(self.salinity_filtered, self.salinity_ave[i, j])
        print("Filtered correctly:")
        print("lat: ", self.lat_filtered.shape)
        print("lon: ", self.lon_filtered.shape)
        print("depth: ", self.depth_filtered.shape)
        print("salinity: ", self.salinity_filtered.shape)

    def getData4Grid(self):
        self.salinity_grid = []
        self.ind_data = []
        for i in range(self.grid_poly.shape[0]):
            ind_data = ((self.lat_filtered - self.grid_poly[i, 0]) ** 2 + (self.lon_filtered - self.grid_poly[i, 1]) ** 2).argmin()
            self.salinity_grid.append(self.salinity_filtered[ind_data])
            self.ind_data.append(ind_data)
        self.salinity_grid = np.array(self.salinity_grid).reshape(-1, 1)
        self.ind_data = np.array(self.ind_data).reshape(-1, 1)

    def getVariogramLateral(self):
        from skgstat import Variogram
        x, y = self.latlon2xy(self.grid_poly[:, 0], self.grid_poly[:, 1], self.lat_origin, self.lon_origin)

        # V_v = Variogram(coordinates=np.hstack((x, y)), values=residual[ind].squeeze(), n_lags=20, maxlag=1500,
        #                 use_nugget=True)
        # # V_v.fit_method = 'trf' # moment method
        # fig = V_v.plot(hist=True)
        # fig.savefig("test1.pdf")
        # print(V_v)

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


data_folder_new = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/D2_HDF/"
data_folder_merged = data_folder_new + "Merged/"
a = Prior1(data_folder_merged, debug = False)
a.getData4Grid()

# a.plot_select_region()
#%%
# plt.contourf(a.grid_poly[:, 1], a.grid_poly[:, 0], a.salinity_grid, np.arange(0,1.01,0.01))
# plt.contourf(a.grid_poly[:, 1], a.grid_poly[:, 0], c = a.salinity_grid, vmin = 25, vmax = 35, cmap = "Paired", s = 200)
# plt.colorbar()
# plt.show()
# plt.scatter(a.lon, a.lat, c = a.salinity, cmap = "Paired")
# plt.colorbar()
# plt.show()

class Prior2:
    '''
    Prior2 is build based on the 3D data forcasting, no wind data is available.
    '''
    def __init__(self):
        pass
