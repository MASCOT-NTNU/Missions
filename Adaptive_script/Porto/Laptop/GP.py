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
import numpy as np

class GP_Poly:
    '''
    Gaussian Process
    '''
    path_onboard = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Adaptive_script/Porto/Onboard/"
    # coef for salinity

    sigma_sal = np.sqrt(.5) # scaling coef in matern kernel for salinity
    tau_sal = np.sqrt(.04) # iid noise
    Threshold_S = 35 # threshold for salinity

    # coef shared in common
    range_lateral = 550
    range_vertical = 2
    eta = 4.5 / range_lateral # coef in matern kernel
    ksi = range_lateral / range_vertical # scaling factor in 3D

    circumference = 40075000 # m
    lat_origin, lon_origin = 41.10251, -8.669811  # the right bottom corner coordinates
    # compute distance matrix and covariance matrix
    distanceMatrix = None
    Sigma_sal = None
    Sigma_temp = None

    noise_sal = tau_sal ** 2
    R_sal = np.diagflat(noise_sal)  # diag not anymore support constructing matrix from vector

    def __init__(self, debug = False):
        print("Here comes the Gaussian process setup!")
        t1 = time.time()
        self.load_grid()
        self.getDistanceMatrix()
        self.getMaternSigma()
        self.print_gaussianprocess()
        self.save_GP()
        t2 = time.time()
        print("Parameters are set up correctly, it takes: ", t2 - t1)

    def save_GP(self):
        self.R_sal = self.R_sal.reshape(-1, 1)
        print("R_sal will be saved: ", self.R_sal)
        np.savetxt(self.path_onboard + "R_sal.txt", self.R_sal, delimiter=", ")
        self.Threshold_S = np.array(self.Threshold_S).reshape(-1, 1)
        print("Threshold_S will be saved: ", self.Threshold_S)
        np.savetxt(self.path_onboard + "Threshold_S.txt", self.Threshold_S, delimiter=", ")
        print("GP is saved successfully.")


    def load_grid(self):
        print("loading the grid...")
        self.grid_poly = np.loadtxt(self.path_onboard + "grid.txt", delimiter=", ")
        print("grid is loaded successfully, grid: ", self.grid_poly.shape)

    def print_gaussianprocess(self):
        print("sigma_sal: ", self.sigma_sal)
        print("tau_sal: ", self.tau_sal)
        print("Threshold_S: ", self.Threshold_S)
        print("eta: ", self.eta)
        print("ksi: ", self.ksi)

    def getDistanceMatrix(self):
        '''
        :return: Distance matrix with scaling the depth direction
        '''
        lat = self.grid_poly[:, 0].reshape(-1, 1)
        lon = self.grid_poly[:, 1].reshape(-1, 1)
        depth = self.grid_poly[:, 2].reshape(-1, 1)
        x, y = self.latlon2xy(lat, lon, self.lat_origin, self.lon_origin)
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        dist_x = x @ np.ones([1, x.shape[0]]) - np.ones([x.shape[0], 1]) @ x.T
        dist_y = y @ np.ones([1, y.shape[0]]) - np.ones([y.shape[0], 1]) @ y.T
        dist_xy = dist_x ** 2 + dist_y ** 2
        dist_z = depth @ np.ones([1, depth.shape[0]]) - np.ones([depth.shape[0], 1]) @ depth.T
        self.distanceMatrix = np.sqrt(dist_xy + (self.ksi * dist_z) ** 2)

    def getMaternSigma(self):
        '''
        :return: Covariance matrix for salinity only
        '''
        print("computing the covariance matrix...")
        t1 = time.time()
        self.Sigma_sal = self.sigma_sal ** 2 * (1 + self.eta * self.distanceMatrix) * np.exp(-self.eta * self.distanceMatrix)
        np.savetxt(self.path_onboard + "Sigma_sal.txt", self.Sigma_sal, delimiter=", ")
        t2 = time.time()
        print("Covariance matrix is saved successfully, Sigma: ", self.Sigma_sal.shape)
        print("Time consumed: ", t2 - t1)

    def checkSigma(self):
        Sigma_sal = np.loadtxt(self.path_onboard + "Sigma_sal.txt", delimiter=", ")
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(Sigma_sal)
        plt.colorbar()
        plt.show()

    @staticmethod
    def deg2rad(deg):
        return deg / 180 * np.pi

    @staticmethod
    def rad2deg(rad):
        return rad / np.pi * 180

    @staticmethod
    def latlon2xy(lat, lon, lat_origin, lon_origin):
        x = GP_Poly.deg2rad((lat - lat_origin)) / 2 / np.pi * GP_Poly.circumference
        y = GP_Poly.deg2rad((lon - lon_origin)) / 2 / np.pi * GP_Poly.circumference * np.cos(GP_Poly.deg2rad(lat))
        return x, y

if __name__ == "__main__":
    a = GP_Poly()
    # a.checkSigma()

