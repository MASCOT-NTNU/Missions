#! /usr/bin/env python3
__author__ = "Yaolin Ge"
__copyright__ = "Copyright 2021, The MASCOT Project, NTNU (https://wiki.math.ntnu.no/mascot)"
__credits__ = ["Yaolin Ge"]
__license__ = "MIT"
__version__ = "1.0.1"
__maintainer__ = "Yaolin Ge"
__email__ = "yaolin.ge@ntnu.no"
__status__ = "UnderDevelopment"

from usr_func import *
from pathlib import Path
import os

'''

Objective: Calibrate the prior field based on the in-situ data
Function: Get in-situ data and find the corresponding delft3D data and then adjust delft3D

'''

class PostProcessor:

    def __init__(self):
        self.load_global_path()
        self.load_file_path_initial_survey() # filepath to which it has the data from the presurvey
        self.data_path_mission = self.filepath_initial_survey
        print("filepath_initial_survey: ", self.filepath_initial_survey)
        print("data_path_mission: ", self.data_path_mission)
        self.path = np.loadtxt(self.data_path_mission + "/data_path.txt", delimiter=", ")
        self.salinity_auv = np.loadtxt(self.data_path_mission + "/data_salinity.txt", delimiter=", ")
        self.load_prior()
        self.prior_calibrator()

    def load_global_path(self):
        print("Now it will load the global path.")
        self.path_global = open("path_global.txt", 'r').read()
        print("global path is set up successfully!")
        print(self.path_global)

    def load_file_path_initial_survey(self):
        print("Loading the pre survey file path...")
        self.f_pre = open(self.path_global + "/Config/filepath_initial_survey.txt", 'r')
        self.filepath_initial_survey = self.f_pre.read()
        self.f_pre.close()
        print("Finished pre survey file path, filepath: ", self.filepath_initial_survey)

    def load_windcondition(self):
        print("It will load the wind conditions...")
        f_wind = open(self.path_global + "/Config/wind_condition.txt", 'r')
        s = f_wind.read()
        ind_wind_dir = s.index("wind_dir=")
        ind_wind_level = s.index(", wind_level=")
        self.wind_dir = s[ind_wind_dir + 9:ind_wind_level]
        self.wind_level = s[ind_wind_level + 13:]
        print("wind_dir: ", self.wind_dir)
        print("wind_level: ", self.wind_level)

    def load_prior(self):
        self.load_windcondition()
        self.prior_data = np.loadtxt(self.path_global + "/Data/Prior/Delft3D_" + self.wind_dir + "_" + self.wind_level + ".txt", delimiter=", ")
        self.lat_prior = self.prior_data[:, 0]
        self.lon_prior = self.prior_data[:, 1]
        self.depth_prior = self.prior_data[:, 2]
        self.salinity_prior = self.prior_data[:, -1]
        print("Loading prior successfully.")
        print("lat_prior: ", self.lat_prior.shape)
        print("lon_prior: ", self.lon_prior.shape)
        print("depth_prior: ", self.depth_prior.shape)
        print("salinity_prior: ", self.salinity_prior.shape)

    def getPriorIndAtLoc(self, loc):
        '''
        return the index in the prior data which corresponds to the location
        '''
        lat, lon, depth = loc
        distDepth = self.depth_prior - depth
        distLat = self.lat_prior - lat
        distLon = self.lon_prior - lon
        dist = np.sqrt(distLat ** 2 + distLon ** 2 + distDepth ** 2)
        ind_loc = np.where(dist == np.nanmin(dist))[0][0]
        return ind_loc

    def prior_calibrator(self):
        '''
        calibrate the prior, return the data matrix
        '''
        self.lat_auv = self.path[:, 0]
        self.lon_auv = self.path[:, 1]
        self.depth_auv = self.path[:, 2]
        self.salinity_prior_reg = []
        for i in range(len(self.lat_auv)):
            ind_loc = self.getPriorIndAtLoc([self.lat_auv[i], self.lon_auv[i], self.depth_auv[i]])
            self.salinity_prior_reg.append(self.salinity_prior[ind_loc])
        self.salinity_prior_reg = np.array(self.salinity_prior_reg).reshape(-1, 1)
        X = np.hstack((np.ones_like(self.salinity_prior_reg), self.salinity_prior_reg))
        Y = self.salinity_auv
        self.beta = np.linalg.solve(X.T @ X, (X.T @ Y))
        print("Prior is calibrated, beta: ", self.beta)
        np.savetxt(self.path_global + "/Config/beta.txt", self.beta, delimiter=", ")
        self.salinity_prior_corrected = self.beta[0] + self.beta[1] * self.salinity_prior
        self.data_prior_corrected = np.hstack((self.lat_prior.reshape(-1, 1),
                                          self.lon_prior.reshape(-1, 1),
                                          self.depth_prior.reshape(-1, 1),
                                          self.salinity_prior_corrected.reshape(-1, 1)))
        self.checkPath(self.path_global + "/Data/Corrected/")
        np.savetxt(self.path_global + "/Data/Corrected/Prior_corrected.txt", self.data_prior_corrected, delimiter=", ")
        print("corrected prior is saved correctly, it is saved in prior_corrected.txt")

    def checkPath(self, path):
        if not os.path.exists(path):
            print("New data path is created: ", path)
            path = Path(path)
            path.mkdir(parents = True, exist_ok=True)
        else:
            print("Folder is already existing, no need to create! ")

if __name__ == "__main__":
    a = PostProcessor()
