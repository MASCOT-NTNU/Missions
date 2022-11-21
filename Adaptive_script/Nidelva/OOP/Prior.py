#! /usr/bin/env python3
__author__ = "Yaolin Ge"
__copyright__ = "Copyright 2021, The MASCOT Project, NTNU (https://wiki.math.ntnu.no/mascot)"
__credits__ = ["Yaolin Ge"]
__license__ = "MIT"
__version__ = "1.0.1"
__maintainer__ = "Yaolin Ge"
__email__ = "yaolin.ge@ntnu.no"
__status__ = "UnderDevelopment"

import numpy as np
import os
from Adaptive_script.Porto.GP import GaussianProcess

class Prior(GaussianProcess):
    AUVdata = None
    mu_prior_sal = None
    mu_prior_temp = None
    Sigma_prior_sal = None
    Sigma_prior_temp = None
    beta0 = None
    beta1 = None
    SINMOD_Data = None
    SINMOD_datapath = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Adaptive_script/samples_2020.05.01.nc"
    if os.path.exists(SINMOD_datapath):
        pass
    else:
        SINMOD_datapath = "/home/yaoling/MASCOT/adaframe/catkin_ws/src/adaframe_examples/scripts/samples_2020.05.01.nc"

    def __init__(self):
        GaussianProcess.__init__(self)
        self.getAUVData()
        self.getSINMODData()
        self.getCoefficients()
        self.get_mu_prior()
        self.saveCoef()
        self.print_Prior()

    def print_Prior(self):
        print("coord_grid: ", self.grid_coord.shape)
        print("beta0: \n", self.beta0)
        print("beta1: \n", self.beta1)
        print("mu_sal_prior: ", self.mu_prior_sal.shape)
        print("mu_temp_prior: ", self.mu_prior_temp.shape)
        print("Prior is setup successfully!\n\n")

    def getAUVData(self):
        self.AUVdata = np.loadtxt('Adaptive_script/Porto/data.txt', delimiter = ",")

    def getSINMODData(self):
        import netCDF4
        print(self.SINMOD_datapath)
        self.SINMOD_Data = netCDF4.Dataset(self.SINMOD_datapath)

    def getSINMODFromCoordsDepth(self, coordinates, depth):
        salinity = np.mean(self.SINMOD_Data['salinity'][:, :, :, :], axis=0)
        temperature = np.mean(self.SINMOD_Data['temperature'][:, :, :, :], axis=0) - 273.15
        depth_sinmod = np.array(self.SINMOD_Data['zc'])
        lat_sinmod = np.array(self.SINMOD_Data['gridLats'][:, :]).reshape(-1, 1)
        lon_sinmod = np.array(self.SINMOD_Data['gridLons'][:, :]).reshape(-1, 1)
        sal_sinmod = np.zeros([coordinates.shape[0], 1])
        temp_sinmod = np.zeros([coordinates.shape[0], 1])

        for i in range(coordinates.shape[0]):
            lat, lon = coordinates[i]
            print(np.where(np.array(depth_sinmod) == depth)[0][0])
            ind_depth = np.where(np.array(depth_sinmod) == depth)[0][0]
            idx = np.argmin((lat_sinmod - lat) ** 2 + (lon_sinmod - lon) ** 2)
            sal_sinmod[i] = salinity[ind_depth].reshape(-1, 1)[idx]
            temp_sinmod[i] = temperature[ind_depth].reshape(-1, 1)[idx]
        return sal_sinmod, temp_sinmod

    def getCoefficients(self):
        # timestamp = self.AUVdata[:, 0].reshape(-1, 1)
        lat_auv_origin = self.rad2deg(self.AUVdata[:, 1]).reshape(-1, 1)
        lon_auv_origin = self.rad2deg(self.AUVdata[:, 2]).reshape(-1, 1)
        xauv = self.AUVdata[:, 3].reshape(-1, 1)
        yauv = self.AUVdata[:, 4].reshape(-1, 1)
        # zauv = self.AUVdata[:, 5].reshape(-1, 1)
        depth_auv = self.AUVdata[:, 6].reshape(-1, 1)
        sal_auv = self.AUVdata[:, 7].reshape(-1, 1)
        temp_auv = self.AUVdata[:, 8].reshape(-1, 1)
        lat_auv = lat_auv_origin + self.rad2deg(xauv * np.pi * 2.0 / self.circumference)
        lon_auv = lon_auv_origin + self.rad2deg(yauv * np.pi * 2.0 / (self.circumference * np.cos(self.deg2rad(lat_auv))))

        depthl = np.array(self.depth_obs) - self.depth_tolerance
        depthu = np.array(self.depth_obs) + self.depth_tolerance

        self.beta0 = np.zeros([len(self.depth_obs), 2])
        self.beta1 = np.zeros([len(self.depth_obs), 2])
        sal_residual = []
        temp_residual = []

        for i in range(len(self.depth_obs)):
            # sort out AUV data
            ind_obs = (depthl[i] <= depth_auv) & (depth_auv <= depthu[i])
            lat_obs = lat_auv[ind_obs].reshape(-1, 1)
            lon_obs = lon_auv[ind_obs].reshape(-1, 1)
            sal_obs = sal_auv[ind_obs].reshape(-1, 1)
            temp_obs = temp_auv[ind_obs].reshape(-1, 1)
            coord_obs = np.hstack((lat_obs, lon_obs))

            # sort out SINMOD data
            sal_sinmod, temp_sinmod = self.getSINMODFromCoordsDepth(coord_obs, self.depth_obs[i])

            # compute the coef for salinity
            sal_modelX = np.hstack((np.ones_like(sal_sinmod), sal_sinmod))
            sal_modelY = sal_obs
            Beta_sal = np.linalg.solve((sal_modelX.T @ sal_modelX), (sal_modelX.T @ sal_modelY))
            # compute the coef for temperature
            temp_modelX = np.hstack((np.ones_like(temp_sinmod), temp_sinmod))
            temp_modelY = temp_obs
            Beta_temp = np.linalg.solve((temp_modelX.T @ temp_modelX), (temp_modelX.T @ temp_modelY))

            self.beta0[i, :] = np.hstack((Beta_sal[0], Beta_temp[0]))
            self.beta1[i, :] = np.hstack((Beta_sal[1], Beta_temp[1]))

            sal_residual.append(sal_obs - Beta_sal[0] - Beta_sal[1] * sal_sinmod)
            temp_residual.append(temp_obs - Beta_temp[0] - Beta_temp[1] * temp_sinmod)

    def get_mu_prior(self):
        self.mu_prior_sal = []
        self.mu_prior_temp = []
        for i in range(len(self.depth_obs)):
            sal_sinmod, temp_sinmod = self.getSINMODFromCoordsDepth(self.grid_coord, self.depth_obs[i])
            self.mu_prior_sal.append(self.beta0[i, 0] + self.beta1[i, 0] * sal_sinmod)
            self.mu_prior_temp.append(self.beta0[i, 1] + self.beta1[i, 1] * temp_sinmod)
        self.mu_prior_sal = np.array(self.mu_prior_sal).reshape(-1, 1)
        self.mu_prior_temp = np.array(self.mu_prior_temp).reshape(-1, 1)

    def saveCoef(self):
        np.savetxt("beta0.txt", self.beta0, delimiter=",")
        np.savetxt("beta1.txt", self.beta1, delimiter=",")
        np.savetxt("mu_prior_sal.txt", self.mu_prior_sal, delimiter=",")
        np.savetxt("mu_prior_temp.txt", self.mu_prior_temp, delimiter=",")

# if __name__ == "__main__":
#     prior = Prior()
#     print("Ferdig med prior")