#!/usr/bin/env python3
__author__ = "Yaolin Ge"
__copyright__ = "Copyright 2021, The MASCOT Project, NTNU (https://wiki.math.ntnu.no/mascot)"
__credits__ = ["Yaolin Ge"]
__license__ = "MIT"
__version__ = "1.0.1"
__maintainer__ = "Yaolin Ge"
__email__ = "yaolin.ge@ntnu.no"
__status__ = "UnderDevelopment"


from scipy.stats import norm
import time
from usr_func import *

'''

Objective: Find the Starting location based on the corrected prior data
Function: Send the starting location via SMS

'''

class Orienteer:

    def __init__(self):
        t1 = time.time()
        self.load_global_path()
        self.load_file_path_initial_survey()
        self.load_auv_loc_now()
        self.load_prior()
        self.find_starting_loc()
        t2 = time.time()
        print("Orienteer is initialised successfully! Time consumed: ", t2 - t1)

    def load_global_path(self):
        print("Now it will load the global path.")
        self.path_global = open("path_global.txt", 'r').read()
        print("global path is set up successfully!")
        print(self.path_global)

    def EP_1D(self, mu, Sigma, Threshold):
        EP = np.zeros_like(mu)
        for i in range(EP.shape[0]):
            EP[i] = norm.cdf(Threshold, mu[i], Sigma[i, i])
        return EP

    def load_prior(self):
        self.prior_extracted_path = self.path_global + "/Data/Corrected/Prior_extracted.txt"
        self.sigma_path = self.path_global + "/Config/Sigma_sal.txt"
        self.threshold_path = self.path_global + "/Config/threshold.txt"
        self.data_prior = np.loadtxt(self.prior_extracted_path, delimiter=", ")
        self.salinity_prior_corrected = self.data_prior[:, -1]
        self.lat_prior = self.data_prior[:, 0]
        self.lon_prior = self.data_prior[:, 1]
        self.depth_prior = self.data_prior[:, 2]
        self.mu_prior = self.salinity_prior_corrected
        self.Sigma_prior = np.loadtxt(self.sigma_path, delimiter=", ")
        self.Threshold_S = np.loadtxt(self.threshold_path, delimiter=", ")
        self.N = len(self.mu_prior)
        print("Prior for salinity is loaded correctly!!!")
        print("mu_prior shape is: ", self.mu_prior.shape)
        print("Sigma_prior shape is: ", self.Sigma_prior.shape)
        print("Threshold: ", self.Threshold_S)
        print("N: ", self.N)

    def load_file_path_initial_survey(self):
        print("Loading the pre survey file path...")
        self.f_pre = open(self.path_global + "/Config/filepath_initial_survey.txt", 'r')
        self.filepath_initial_survey = self.f_pre.read()
        self.f_pre.close()
        print("Finished pre survey file path, filepath: ", self.filepath_initial_survey)

    def load_auv_loc_now(self):
        print("Loading the current AUV location now...")
        self.auv_loc = np.loadtxt(self.filepath_initial_survey + "/data_path.txt", delimiter=", ")
        self.lat_auv_now, self.lon_auv_now, self.depth_auv_now = self.auv_loc[-1, :]
        print("Current AUV location is loaded successfully: ", self.lat_auv_now, self.lon_auv_now, self.depth_auv_now)

    def get_normalised_distance_vector(self):
        dx, dy = latlon2xy(self.lat_prior, self.lon_prior, self.lat_auv_now, self.lon_auv_now)
        dist = np.sqrt(dx ** 2 + dy ** 2)
        dist_normalised = dist / np.amax(dist)
        return dist_normalised

    def find_starting_loc(self):
        dist_norm = self.get_normalised_distance_vector()
        EP_Prior = self.EP_1D(self.mu_prior, self.Sigma_prior, self.Threshold_S)
        ep_criterion = 0.5  # excursion probability close to 0.5
        self.ind_start = (np.abs(EP_Prior - ep_criterion + dist_norm)).argmin()
        if self.ind_start == 0:
            self.ind_start = np.random.randint(self.N)
        print("ind_start: ", self.ind_start)
        np.savetxt(self.path_global + "/Config/ind_start.txt", self.ind_start.reshape(-1, 1), delimiter=", ")
        print("Starting index is saved successfully!")

if __name__ == "__main__":
    a = Orienteer()
