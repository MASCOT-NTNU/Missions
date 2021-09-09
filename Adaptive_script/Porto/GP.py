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
from Adaptive_script.Porto.Prior import Prior1, Prior2

class GP_Poly(Prior2):
    '''
    Gaussian Process
    '''

    # coef for salinity
    sigma_sal = np.sqrt(.01) # scaling coef in matern kernel for salinity
    tau_sal = np.sqrt(.001) # iid noise
    Threshold_S = 34.5 # threshold for salinity

    # coef shared in common
    eta = 4.5 / 400 # coef in matern kernel
    ksi = 600 / 2 # scaling factor in 3D

    # compute distance matrix and covariance matrix
    distanceMatrix = None
    Sigma_sal = None
    Sigma_temp = None

    noise_sal = tau_sal ** 2
    R_sal = np.diagflat(noise_sal)  # diag not anymore support constructing matrix from vector

    def __init__(self, debug = False):
        Prior2.__init__(self, debug = False)
        print("Here comes the Gaussian process setup!")
        t1 = time.time()
        self.getDistanceMatrix()
        self.getMaternSigma()
        self.print_gaussianprocess()
        t2 = time.time()
        print("Parameters are set up correctly, it takes: ", t2 - t1)

    def print_gaussianprocess(self):
        print("sigma_sal: ", self.sigma_sal)
        print("tau_sal: ", self.tau_sal)
        print("Threshold_S: ", self.Threshold_S)
        print("eta: ", self.eta)
        print("ksi: ", self.ksi)

    def set_sigma_sal(self, value):
        self.sigma_sal = value

    def set_tau_sal(self, value):
        self.tau_sal = value

    def set_Threshold_S(self, value):
        self.Threshold_S = value

    def set_eta(self, value):
        self.sigma_sal = value

    def set_ksi(self, value):
        self.sigma_sal = value

    def getDistanceMatrix(self):
        '''
        :return: Distance matrix with scaling the depth direction
        '''
        lat = self.lat_selected[:, 0].reshape(-1, 1)
        lon = self.lon_selected[:, 0].reshape(-1, 1)
        # depth = self.depth.reshape(-1, 1)
        x, y = self.latlon2xy(lat, lon, self.lat_origin, self.lon_origin)
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        grid = np.hstack((y, x))
        import scipy.spatial.distance as scdist
        self.distanceMatrix = scdist.cdist(grid, grid)
        # dist_x = x @ np.ones([1, x.shape[0]]) - np.ones([x.shape[0], 1]) @ x.T
        # dist_y = y @ np.ones([1, y.shape[0]]) - np.ones([y.shape[0], 1]) @ y.T
        # dist_xy = dist_x ** 2 + dist_y ** 2
        # dist_z = depth @ np.ones([1, depth.shape[0]]) - np.ones([depth.shape[0], 1]) @ depth.T
        # self.distanceMatrix = np.sqrt(dist_xy + (self.ksi * dist_z) ** 2)

    def getMaternSigma(self):
        '''
        :return: Covariance matrix for salinity only
        '''
        self.Sigma_sal = self.sigma_sal ** 2 * (1 + self.eta * self.distanceMatrix) * np.exp(-self.eta * self.distanceMatrix)

# a = GP_Poly()





