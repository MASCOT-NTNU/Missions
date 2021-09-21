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

class AUV:
    circumference = 40075000  # circumference of the earth, [m]
    lat_origin, lon_origin = 41.061874, -8.650977  # origin location
    distance_poly = 100  # [m], distance between two neighbouring points
    depth_obs = [-.5, -1.25, -2] # [m], distance in depth, depth to be explored
    def __init__(self):
        self.node_name = 'MASCOT'
        rospy.init_node(self.node_name, anonymous=True)
        self.rate = rospy.Rate(1)  # 1Hz
        self.auv_handler = AuvHandler(self.node_name, "MASCOT")

        rospy.Subscriber("/Vehicle/Out/Temperature_filtered", Temperature, self.TemperatureCB)
        rospy.Subscriber("/Vehicle/Out/Salinity_filtered", Salinity, self.SalinityCB)
        rospy.Subscriber("/Vehicle/Out/EstimatedState_filtered", EstimatedState, self.EstimatedStateCB)

        self.speed = 1.2  # m/s
        self.depth = 0.0  # meters
        self.last_state = "unavailable"
        self.rate.sleep()
        self.init = True
        self.currentTemperature = 0.0
        self.currentSalinity = 0.0
        self.vehicle_pos = [0, 0, 0]

        self.sms_pub_ = rospy.Publisher("/IMC/In/Sms", Sms, queue_size = 10)
        self.phone_number = "+4792526858"

    def TemperatureCB(self, msg):
        self.currentTemperature = msg.value.data

    def SalinityCB(self, msg):
        self.currentSalinity = msg.value.data

    def EstimatedStateCB(self, msg):
        offset_north = msg.lat.data - AUV.deg2rad(self.lat_origin)
        offset_east = msg.lon.data - AUV.deg2rad(self.lon_origin)
        N = offset_north * self.circumference / (2.0 * np.pi)
        E = offset_east * self.circumference * np.cos(AUV.deg2rad(self.lat_origin)) / (2.0 * np.pi)
        D = msg.z.data
        self.vehicle_pos = [N, E, D]

    @staticmethod
    def deg2rad(deg):
        return deg / 180 * np.pi

    @staticmethod
    def rad2deg(rad):
        return rad / np.pi * 180

    @staticmethod
    def latlon2xy(lat, lon, lat_origin, lon_origin):
        x = AUV.deg2rad((lat - lat_origin)) / 2 / np.pi * AUV.circumference
        y = AUV.deg2rad((lon - lon_origin)) / 2 / np.pi * AUV.circumference * np.cos(AUV.deg2rad(lat))
        return x, y

    @staticmethod
    def xy2latlon(x, y, lat_origin, lon_origin):
        lat = lat_origin + AUV.rad2deg(x * np.pi * 2.0 / AUV.circumference)
        lon = lon_origin + AUV.rad2deg(y * np.pi * 2.0 / (AUV.circumference * np.cos(AUV.deg2rad(lat))))
        return lat, lon

class DataAssimilator(AUV):
    data_salinity = []
    data_temperature = []
    data_path_waypoints = []
    data_timestamp = []
    path_onboard = ''

    def __init__(self):
        AUV.__init__(self)
        print("Data collector is initialised correctly")

    def append_salinity(self, value):
        DataAssimilator.data_salinity.append(value)

    def append_temperature(self, value):
        DataAssimilator.data_temperature.append(value)

    def append_path(self, value):
        DataAssimilator.data_path_waypoints.append(value)

    def append_timestamp(self, value):
        DataAssimilator.data_timestamp.append(value)

    def createDataPath(self):
        self.date_string = datetime.now().strftime("%Y_%m%d_%H%M")
        self.data_path_mission = os.getcwd() + "/Data/Pre_survey_data_on_" + self.date_string
        f_pre = open("filepath_initial_survey.txt", 'w')
        f_pre.write(self.data_path_mission)
        f_pre.close()
        if not os.path.exists(self.data_path_mission):
            print("New data path is created: ", self.data_path_mission)
            path = Path(self.data_path_mission)
            path.mkdir(parents = True, exist_ok=True)
        else:
            print("Folder is already existing, no need to create! ")

    def save_data(self):
        self.data_salinity_saved = np.array(self.data_salinity).reshape(-1, 1)
        self.data_temperature_saved = np.array(self.data_temperature).reshape(-1, 1)
        self.data_path_waypoints_saved = np.array(self.data_path_waypoints).reshape(-1, 3)
        self.data_timestamp_saved = np.array(self.data_timestamp).reshape(-1, 1)
        np.savetxt(self.data_path_mission + "/data_salinity.txt", self.data_salinity_saved, delimiter=", ")
        np.savetxt(self.data_path_mission + "/data_temperature.txt", self.data_temperature_saved, delimiter=", ")
        np.savetxt(self.data_path_mission + "/data_path.txt", self.data_path_waypoints_saved, delimiter=", ")
        np.savetxt(self.data_path_mission + "/data_timestamp.txt", self.data_timestamp_saved, delimiter=", ")

    def vehpos2latlon(self, x, y, lat_origin, lon_origin):
        if lat_origin <= 10:
            lat_origin = AUV.rad2deg(lat_origin)
            lon_origin = AUV.rad2deg(lon_origin)
        lat = lat_origin + AUV.rad2deg(x * np.pi * 2.0 / self.circumference)
        lon = lon_origin + AUV.rad2deg(y * np.pi * 2.0 / (self.circumference * np.cos(AUV.deg2rad(lat))))
        return lat, lon

class Pre_surveyor(DataAssimilator):
    '''
    Calibrate the prior based on the linear regression
    '''

    def __init__(self, debug = False):
        DataAssimilator.__init__(self)
        self.load_path_initial_survey()
        self.Pre_surveyor()

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

    def surfacing(self, time_length):
        for i in range(time_length):
            if i % 30 == 0:
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
            if np.around((self.t2 - self.t1), 0) % 600 == 0:
                print("Longer than 10 mins, need a long break")
                self.surfacing(90)  # surfacing 90 seconds after 10 mins of travelling
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


