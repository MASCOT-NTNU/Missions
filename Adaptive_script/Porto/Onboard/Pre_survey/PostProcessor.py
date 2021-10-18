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
import rospy
from auv_handler import AuvHandler
import imc_ros_interface
from imc_ros_interface.msg import Temperature, Salinity, EstimatedState, Sms
import time
import os
from pathlib import Path
from datetime import datetime
from AUV import AUV

class Prior(Pre_surveyor):
    path = None
    def __init__(self):
        Pre_surveyor.__init__(self)
        self.load_file_path_initial_survey() # filepath to which it has the data from the presurvey
        self.data_path_mission = self.file_path_initial_survey
        print("filepath_initial_survey: ", self.file_path_initial_survey)
        print("data_path_mission: ", self.data_path_mission)
        self.path = np.loadtxt(self.data_path_mission + "/data_path.txt", delimiter=", ")
        self.salinity_auv = np.loadtxt(self.data_path_mission + "/data_salinity.txt", delimiter=", ")
        self.load_prior()
        self.prior_calibrator()

    def load_file_path_initial_survey(self):
        print("Loading the pre survey file path...")
        self.f_pre = open("filepath_initial_survey.txt", 'r')
        self.file_path_initial_survey = self.f_pre.read()
        self.f_pre.close()
        print("Finished pre survey file path, filepath: ", self.file_path_initial_survey)

    def load_prior(self):
        self.prior_data = np.loadtxt("Prior_polygon.txt", delimiter=", ")
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
        ind_loc = np.where(dist == dist.min())[0][0]
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
        np.savetxt("beta.txt", self.beta, delimiter=", ")
        self.salinity_prior_corrected = self.beta[0] + self.beta[1] * self.salinity_prior
        self.data_prior_corrected = np.hstack((self.lat_prior.reshape(-1, 1),
                                          self.lon_prior.reshape(-1, 1),
                                          self.depth_prior.reshape(-1, 1),
                                          self.salinity_prior_corrected.reshape(-1, 1)))
        np.savetxt("prior_corrected.txt", self.data_prior_corrected, delimiter=", ")
        print("corrected prior is saved correctly, it is saved in prior_corrected.txt")