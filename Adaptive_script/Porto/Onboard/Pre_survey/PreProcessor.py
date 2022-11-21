#!/usr/bin/env python3
__author__ = "Yaolin Ge"
__copyright__ = "Copyright 2021, The MASCOT Project, NTNU (https://wiki.math.ntnu.no/mascot)"
__credits__ = ["Yaolin Ge"]
__license__ = "MIT"
__version__ = "1.0.1"
__maintainer__ = "Yaolin Ge"
__email__ = "yaolin.ge@ntnu.no"
__status__ = "UnderDevelopment"


import time
from usr_func import *


class PreProcessor:
    '''
    Prepare the Gaussian Process
    '''
    lat_origin, lon_origin = 41.061874, -8.650977  # origin location
    sigma_sal = np.sqrt(.5) # scaling coef in matern kernel for salinity
    tau_sal = np.sqrt(.04) # iid noise
    # coef shared in common
    range_lateral = 550
    range_vertical = 2
    eta = 4.5 / range_lateral # coef in matern kernel
    ksi = range_lateral / range_vertical # scaling factor in 3D
    noise_sal = tau_sal ** 2
    R_sal = np.diagflat(noise_sal)  # diag not anymore support constructing matrix from vector

    def __init__(self):
        print("PreProcessor is initialised successfully!")
        self.load_global_path()
        self.load_grid()
        self.load_prior_corrected()
        self.extractData4Grid()
        self.getThreshold()
        self.getR()
        self.getCovMatrix()

    def load_global_path(self):
        print("Now it will load the global path.")
        self.path_global = open("path_global.txt", 'r').read()
        print("global path is set up successfully!")
        print(self.path_global)

    def load_grid(self):
        print("Loading grid...")
        self.grid = np.loadtxt(self.path_global + "/Config/grid.txt", delimiter = ", ")
        print("Grid is loaded successfully, grid: ", self.grid.shape)

    def load_prior_corrected(self):
        print("Loading corrected prior...")
        self.prior_corrected = np.loadtxt(self.path_global + "/Data/Corrected/Prior_corrected.txt", delimiter = ", ")
        self.lat_prior_corrected = self.prior_corrected[:, 0]
        self.lon_prior_corrected = self.prior_corrected[:, 1]
        self.depth_prior_corrected = self.prior_corrected[:, 2]
        self.salinity_prior_corrected = self.prior_corrected[:, -1]
        print("Corrected prior is loaded successfully! ", self.prior_corrected.shape)

    def getPriorIndAtLoc(self, loc):
        '''
        return the index in the prior data which corresponds to the location
        '''
        lat, lon, depth = loc
        distDepth = self.depth_prior_corrected - depth
        distLat = self.lat_prior_corrected - lat
        distLon = self.lon_prior_corrected - lon
        dist = np.sqrt(distLat ** 2 + distLon ** 2 + distDepth ** 2)
        ind_loc = np.where(dist == np.nanmin(dist))[0][0]
        return ind_loc

    def extractData4Grid(self):
        self.sal_grid = []
        t1 = time.time()
        for loc in self.grid:
            ind_loc = self.getPriorIndAtLoc(loc)
            self.sal_grid.append(self.salinity_prior_corrected[ind_loc])
        self.sal_grid = np.array(self.sal_grid)
        self.prior_extracted = np.hstack((self.grid[:, 0].reshape(-1, 1), self.grid[:, 1].reshape(-1, 1),
                                         self.grid[:, 2].reshape(-1, 1), self.sal_grid.reshape(-1, 1)))
        t2 = time.time()
        print("Data is extracted successfully on the grid!")
        print("Time consumed: ", t2 - t1)
        print("Salinity for grid: ", self.sal_grid.shape)
        np.savetxt(self.path_global + "/Data/Corrected/Prior_extracted.txt", self.prior_extracted, delimiter=", ")
        print("Extracted prior is saved successfully!")

    def getThreshold(self):
        self.threshold = np.nanmean(self.sal_grid).reshape(-1, 1)
        print("Threshold is set: ", self.threshold)
        np.savetxt(self.path_global + "/Config/threshold.txt", self.threshold, delimiter = ", ")

    def getR(self):
        self.R_sal = self.R_sal.reshape(-1, 1)
        print("R_sal will be saved: ", self.R_sal)
        np.savetxt(self.path_global + "/Config/R_sal.txt", self.R_sal, delimiter=", ")

    def getCovMatrix(self):
        '''
        :return: Distance matrix with scaling the depth direction
        '''
        print("computing the covariance matrix...")
        t1 = time.time()
        lat = self.grid[:, 0].reshape(-1, 1)
        lon = self.grid[:, 1].reshape(-1, 1)
        depth = self.grid[:, 2].reshape(-1, 1)
        x, y = latlon2xy(lat, lon, self.lat_origin, self.lon_origin)
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        dist_x = x @ np.ones([1, x.shape[0]]) - np.ones([x.shape[0], 1]) @ x.T
        dist_y = y @ np.ones([1, y.shape[0]]) - np.ones([y.shape[0], 1]) @ y.T
        dist_xy = dist_x ** 2 + dist_y ** 2
        dist_z = depth @ np.ones([1, depth.shape[0]]) - np.ones([depth.shape[0], 1]) @ depth.T
        self.distanceMatrix = np.sqrt(dist_xy + (self.ksi * dist_z) ** 2)
        self.Sigma_sal = self.sigma_sal ** 2 * (1 + self.eta * self.distanceMatrix) * np.exp(
            -self.eta * self.distanceMatrix)
        np.savetxt(self.path_global + "/Config/Sigma_sal.txt", self.Sigma_sal, delimiter=", ")
        t2 = time.time()
        print("Covariance matrix is saved successfully, Sigma: ", self.Sigma_sal.shape)
        print("Time consumed: ", t2 - t1)


if __name__ == "__main__":
    a = PreProcessor()





