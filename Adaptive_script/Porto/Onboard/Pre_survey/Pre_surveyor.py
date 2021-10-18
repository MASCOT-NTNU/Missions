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


class Pre_surveyor(DataAssimilator):
    '''
    Calibrate the prior based on the linear regression
    '''

    def __init__(self, debug = False):
        DataAssimilator.__init__(self)
        self.load_path_initial_survey()
        self.calculateDistacne()
        self.Pre_surveyor()

    def calculateDistacne(self):
        self.lat_travel = self.path_initial_survey[:, 0]
        self.lon_travel = self.path_initial_survey[:, 1]
        self.depth_travel = self.path_initial_survey[:, 2]
        self.lat_pre = self.lat_travel[0]
        self.lon_pre = self.lon_travel[0]
        self.depth_pre = self.depth_travel[0]
        dist = 0
        for i in range(len(self.lat_travel)):
            x_temp, y_temp = self.latlon2xy(self.lat_travel[i], self.lon_travel[i], self.lat_pre, self.lon_pre)
            distZ = self.depth_travel[i] - self.depth_pre
            dist = dist + np.sqrt(x_temp ** 2 + y_temp ** 2 + distZ ** 2)
            self.lat_pre = self.lat_travel[i]
            self.lon_pre = self.lon_travel[i]
            self.depth_pre = self.depth_travel[i]
        print("Total distance needs to be travelled: ", dist)
        print("Time estimated: ", dist / self.speed)

    def load_path_initial_survey(self):
        print("Loading the initial survey path...")
        self.path_initial_survey = np.loadtxt(self.path_onboard + "path_initial_survey.txt", delimiter=", ")
        print("Initial survey path is loaded successfully, path_initial_survey: ", self.path_initial_survey)

    def send_SMS(self):
        print("Message has been sent to: ", self.phone_number)
        SMS = Sms()
        SMS.number.data = self.phone_number
        SMS.timeout.data = 60
        x_auv = self.vehicle_pos[0]
        y_auv = self.vehicle_pos[1]
        lat_auv, lon_auv = self.vehpos2latlon(x_auv, y_auv, self.lat_origin, self.lon_origin)
        SMS.contents.data = "LAUV-Xplore-1 location: " + str(lat_auv) + ", " + str(lon_auv)
        self.sms_pub_.publish(SMS)

    def send_SMS_mission_complete(self):
        print("Mission complete Message has been sent to: ", self.phone_number)
        SMS = Sms()
        SMS.number.data = self.phone_number
        SMS.timeout.data = 60
        x_auv = self.vehicle_pos[0]
        y_auv = self.vehicle_pos[1]
        lat_auv, lon_auv = self.vehpos2latlon(x_auv, y_auv, self.lat_origin, self.lon_origin)
        SMS.contents.data = "Congrats, Mission complete. LAUV-Xplore-1 location: " + str(lat_auv) + ", " + str(lon_auv)
        self.sms_pub_.publish(SMS)

    def surfacing(self, time_length):
        for i in range(time_length):
            if (i + 1) % 15 == 0:
                self.send_SMS()
            self.append_mission_data()
            self.save_mission_data()
            print("Sleep {:d} seconds".format(i))
            print("Now is: ", self.waypoint_lat_now, self.waypoint_lon_now, self.waypoin_depth_now)
            self.auv_handler.setWaypoint(self.waypoint_lat_now, self.waypoint_lon_now, self.waypoin_depth_now)
            self.auv_handler.spin()  # publishes the reference, stay on the surface
            self.rate.sleep()  #

    def save_mission_data(self):
        if np.around((self.t2 - self.t1), 0) % 10 == 0:
            print("Data is saved: ", self.counter_data_saved, " times")
            self.save_data()
            self.counter_data_saved = self.counter_data_saved + 1

    def append_mission_data(self):
        self.append_salinity(self.currentSalinity)
        self.append_temperature(self.currentTemperature)
        lat_temp, lon_temp = self.vehpos2latlon(self.vehicle_pos[0], self.vehicle_pos[1], self.lat_origin,
                                                self.lon_origin)
        self.append_path([lat_temp, lon_temp, self.vehicle_pos[2]])
        self.append_timestamp(datetime.now().timestamp())

    def send_next_waypoint(self):
        if self.path_initial_survey[self.counter_waypoint, 2] == 0:
            if (self.t2 - self.t1) / 600 >= 1 and (self.t2 - self.t1) % 600 >= 0:
                print("Longer than 10 mins, need a long break")
                self.surfacing(90)  # surfacing 90 seconds after 10 mins of travelling
                self.t1 = self.t2
            else:
                print("Less than 10 mins, need a shorter break")
                self.surfacing(30) # surfacing 30 seconds

        # Move to the next waypoint
        self.counter_waypoint = self.counter_waypoint + 1 # should not be changed to the other order, since it can damage the reference
        self.waypoint_lat_now = self.deg2rad(self.path_initial_survey[self.counter_waypoint, 0])
        self.waypoint_lon_now = self.deg2rad(self.path_initial_survey[self.counter_waypoint, 1])
        self.waypoin_depth_now = -self.path_initial_survey[self.counter_waypoint, 2]
        print("Now is: " ,self.waypoint_lat_now, self.waypoint_lon_now, self.waypoin_depth_now)
        self.auv_handler.setWaypoint(self.waypoint_lat_now, self.waypoint_lon_now, self.waypoin_depth_now)
        print("next waypoint", self.deg2rad(self.path_initial_survey[self.counter_waypoint, 0]),
              self.deg2rad(self.path_initial_survey[self.counter_waypoint, 1]),
              self.path_initial_survey[self.counter_waypoint, 2])

    def Pre_surveyor(self):
        self.createDataPath()
        print("Now it will move to the starting location...")
        self.t1 = time.time()
        self.counter_waypoint = 0
        self.counter_data_saved = 0
        self.waypoint_lat_now = self.deg2rad(self.path_initial_survey[self.counter_waypoint, 0])
        self.waypoint_lon_now = self.deg2rad(self.path_initial_survey[self.counter_waypoint, 1])
        self.waypoin_depth_now = - self.path_initial_survey[self.counter_waypoint, 2]# reason for that is that setwaypoint does not accept negative values for depth
        self.auv_handler.setWaypoint(self.waypoint_lat_now, self.waypoint_lon_now, self.waypoin_depth_now)
        while not rospy.is_shutdown():
            if self.init:
                self.t2 = time.time()
                print("Elapsed time: ", self.t2 - self.t1)
                self.append_mission_data()
                self.save_mission_data()
                print(self.auv_handler.getState())
                if self.auv_handler.getState() == "waiting" and self.last_state != "waiting":
                    print("Arrived the current location")
                    if self.counter_waypoint + 1 >= len(self.path_initial_survey):
                        self.send_SMS_mission_complete()
                        rospy.signal_shutdown("Mission completed!!!")
                        break
                    self.send_next_waypoint()
                self.last_state = self.auv_handler.getState()
                self.auv_handler.spin()
            self.rate.sleep()

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

if __name__ == "__main__":
    b = Prior()




