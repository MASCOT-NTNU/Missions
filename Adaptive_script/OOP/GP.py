import numpy as np
from Grid import Grid
class GaussianProcess(Grid):
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

    def __init__(self):
        Grid.__init__(self)
        self.compute_DistanceMatrix()
        self.compute_Sigma()
        self.print_var()
        print("Parameters are set up correctly\n\n")

    def print_var(self):
        print("sigma_sal: ", GaussianProcess.sigma_sal)
        print("tau_sal: ", GaussianProcess.tau_sal)
        print("Threshold_S: ", GaussianProcess.Threshold_S)
        print("sigma_temp: ", GaussianProcess.sigma_temp)
        print("tau_temp: ", GaussianProcess.tau_temp)
        print("Threshold_T: ", GaussianProcess.Threshold_T)
        print("eta: ", GaussianProcess.eta)
        print("ksi: ", GaussianProcess.ksi)

    def set_sigma_sal(self, value):
        GaussianProcess.sigma_sal = value

    def set_sigma_temp(self, value):
        GaussianProcess.sigma_temp = value

    def set_tau_sal(self, value):
        GaussianProcess.tau_sal = value

    def set_tau_temp(self, value):
        GaussianProcess.tau_temp = value

    def set_Threshold_S(self, value):
        GaussianProcess.Threshold_S = value

    def set_Threshold_T(self, value):
        GaussianProcess.Threshold_T = value

    def set_eta(self, value):
        GaussianProcess.sigma_sal = value

    def set_ksi(self, value):
        GaussianProcess.sigma_sal = value

    def DistanceMatrix(self):
        '''
        :return: Distance matrix with scaling the depth direction
        '''
        X = Grid.grid[:, 0].reshape(-1, 1)
        Y = Grid.grid[:, 1].reshape(-1, 1)
        Z = Grid.grid[:, -1].reshape(-1, 1)

        distX = X @ np.ones([1, X.shape[0]]) - np.ones([X.shape[0], 1]) @ X.T
        distY = Y @ np.ones([1, Y.shape[0]]) - np.ones([Y.shape[0], 1]) @ Y.T
        distXY = distX ** 2 + distY ** 2
        distZ = Z @ np.ones([1, Z.shape[0]]) - np.ones([Z.shape[0], 1]) @ Z.T
        dist = np.sqrt(distXY + (GaussianProcess.ksi * distZ) ** 2)
        return dist

    def Matern_cov_sal(self):
        '''
        :return: Covariance matrix for salinity only
        '''
        return GaussianProcess.sigma_sal ** 2 * (1 + GaussianProcess.eta * GaussianProcess.distanceMatrix) * \
               np.exp(-GaussianProcess.eta * GaussianProcess.distanceMatrix)

    def Matern_cov_temp(self):
        '''
        :return: Covariance matrix for temperature only
        '''
        return GaussianProcess.sigma_temp ** 2 * (1 + GaussianProcess.eta * GaussianProcess.distanceMatrix) * \
               np.exp(-GaussianProcess.eta * GaussianProcess.distanceMatrix)

    def compute_DistanceMatrix(self):
        GaussianProcess.distanceMatrix = self.DistanceMatrix()

    def compute_Sigma(self):
        GaussianProcess.Sigma_sal = self.Matern_cov_sal()
        GaussianProcess.Sigma_temp = self.Matern_cov_temp()
