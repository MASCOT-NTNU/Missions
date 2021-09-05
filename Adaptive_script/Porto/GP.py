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
from Adaptive_script.Porto.Grid import Grid, GridPoly

class GaussianProcess(GridPoly):
    '''
    Gaussian Process
    '''

    # coef for salinity
    sigma_sal = np.sqrt(4) # scaling coef in matern kernel for salinity
    tau_sal = np.sqrt(.3) # iid noise
    Threshold_S = 28 # threshold for salinity

    # coef for temperature
    sigma_temp = np.sqrt(.5) # scaling coef in matern kernel for temperature
    tau_temp = np.sqrt(.1) # iid noise
    Threshold_T = 10.5 # threshold for temperature

    # coef shared in common
    eta = 4.5 / 400 # coef in matern kernel
    ksi = 1000 / 24 / .5 # scaling factor in 3D

    # compute distance matrix and covariance matrix
    distanceMatrix = None
    Sigma_sal = None
    Sigma_temp = None

    noise_sal = tau_sal ** 2
    R_sal = np.diagflat(noise_sal)  # diag not anymore support constructing matrix from vector
    noise_temp = tau_temp ** 2
    R_temp = np.diagflat(noise_temp)

    def __init__(self):
        Grid.__init__(self)
        self.compute_DistanceMatrix()
        self.compute_Sigma()
        self.print_gaussianprocess()
        print("Parameters are set up correctly\n\n")

    def print_gaussianprocess(self):
        print("sigma_sal: ", self.sigma_sal)
        print("tau_sal: ", self.tau_sal)
        print("Threshold_S: ", self.Threshold_S)
        print("sigma_temp: ", self.sigma_temp)
        print("tau_temp: ", self.tau_temp)
        print("Threshold_T: ", self.Threshold_T)
        print("eta: ", self.eta)
        print("ksi: ", self.ksi)

    def set_sigma_sal(self, value):
        self.sigma_sal = value

    def set_sigma_temp(self, value):
        self.sigma_temp = value

    def set_tau_sal(self, value):
        self.tau_sal = value

    def set_tau_temp(self, value):
        self.tau_temp = value

    def set_Threshold_S(self, value):
        self.Threshold_S = value

    def set_Threshold_T(self, value):
        self.Threshold_T = value

    def set_eta(self, value):
        self.sigma_sal = value

    def set_ksi(self, value):
        self.sigma_sal = value

    def DistanceMatrix(self):
        '''
        :return: Distance matrix with scaling the depth direction
        '''
        X = self.grid[:, 0].reshape(-1, 1)
        Y = self.grid[:, 1].reshape(-1, 1)
        Z = self.grid[:, -1].reshape(-1, 1)

        distX = X @ np.ones([1, X.shape[0]]) - np.ones([X.shape[0], 1]) @ X.T
        distY = Y @ np.ones([1, Y.shape[0]]) - np.ones([Y.shape[0], 1]) @ Y.T
        distXY = distX ** 2 + distY ** 2
        distZ = Z @ np.ones([1, Z.shape[0]]) - np.ones([Z.shape[0], 1]) @ Z.T
        dist = np.sqrt(distXY + (self.ksi * distZ) ** 2)
        return dist

    def Matern_cov_sal(self):
        '''
        :return: Covariance matrix for salinity only
        '''
        return self.sigma_sal ** 2 * (1 + self.eta * self.distanceMatrix) * np.exp(-self.eta * self.distanceMatrix)

    def Matern_cov_temp(self):
        '''
        :return: Covariance matrix for temperature only
        '''
        return self.sigma_temp ** 2 * (1 + self.eta * self.distanceMatrix) * np.exp(-self.eta * self.distanceMatrix)

    def compute_DistanceMatrix(self):
        self.distanceMatrix = self.DistanceMatrix()

    def compute_Sigma(self):
        self.Sigma_sal = self.Matern_cov_sal()
        self.Sigma_temp = self.Matern_cov_temp()
