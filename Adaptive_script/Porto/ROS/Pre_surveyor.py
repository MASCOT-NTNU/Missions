#! /usr/bin/env python3
__author__ = "Yaolin Ge"
__copyright__ = "Copyright 2021, The MASCOT Project, NTNU (https://wiki.math.ntnu.no/mascot)"
__credits__ = ["Yaolin Ge"]
__license__ = "MIT"
__version__ = "1.0.1"
__maintainer__ = "Yaolin Ge"
__email__ = "yaolin.ge@ntnu.no"
__status__ = "UnderDevelopment"

import h5py
import numpy as np
import rospy
from auv_handler import AuvHandler
import imc_ros_interface
from imc_ros_interface.msg import Temperature, Salinity, EstimatedState
from Grid import GridPoly
from GP import GP_Poly
import time
import os
from datetime import datetime

class AUV(GridPoly):
    def __init__(self):
        self.node_name = 'MASCOT'
        rospy.init_node(self.node_name, anonymous=True)
        self.rate = rospy.Rate(1)  # 1Hz
        self.auv_handler = AuvHandler(self.node_name, "MASCOT")

        rospy.Subscriber("/Vehicle/Out/Temperature_filtered", Temperature, self.TemperatureCB)
        rospy.Subscriber("/Vehicle/Out/Salinity_filtered", Salinity, self.SalinityCB)
        rospy.Subscriber("/Vehicle/Out/EstimatedState_filtered", EstimatedState, self.EstimatedStateCB)

        self.speed = 1.6  # m/s
        self.depth = 0.0  # meters
        self.last_state = "unavailable"
        self.rate.sleep()
        self.init = True
        self.currentTemperature = 0.0
        self.currentSalinity = 0.0
        self.vehicle_pos = [0, 0, 0]
        self.surfacing = False
        self.surfacing_time = 25 # surface time, [sec]

    def TemperatureCB(self, msg):
        self.currentTemperature = msg.value.data

    def SalinityCB(self, msg):
        self.currentSalinity = msg.value.data

    def EstimatedStateCB(self, msg):
        offset_north = msg.lat.data - GridPoly.deg2rad(GridPoly.lat_origin)
        offset_east = msg.lon.data - GridPoly.deg2rad(GridPoly.lon_origin)
        N = offset_north * GridPoly.circumference / (2.0 * np.pi)
        E = offset_east * GridPoly.circumference * np.cos(GridPoly.deg2rad(GridPoly.lat_origin)) / (2.0 * np.pi)
        D = msg.z.data
        self.vehicle_pos = [N, E, D]

class DataAssimilator(GridPoly):
    data_salinity = []
    data_temperature = []
    data_path_waypoints = []
    data_timestamp = []
    def __init__(self):
        self.createDataPath()
        print("Mission data folder is created: ", self.data_path_mission)
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
        if not os.path.exists(self.data_path_mission):
            print("New data path is created: ", self.data_path_mission)
            os.mkdir(self.data_path_mission)
        else:
            print("Folder is already existing, no need to create! ")

    def save_data(self):

        self.data_salinity = np.array(self.data_salinity).reshape(-1, 1)
        self.data_temperature = np.array(self.data_temperature).reshape(-1, 1)
        self.data_path_waypoints = np.array(self.data_path_waypoints).reshape(-1, 3)
        self.data_timestamp = np.array(self.data_timestamp).reshape(-1, 1)

        np.savetxt(self.data_path_mission + "/data_salinity.txt", self.data_salinity, delimiter=", ")
        np.savetxt(self.data_path_mission + "/data_temperature.txt", self.data_temperature, delimiter=", ")
        np.savetxt(self.data_path_mission + "/data_path.txt", self.data_path_waypoints, delimiter=", ")
        np.savetxt(self.data_path_mission + "/data_timestamp.txt", self.data_timestamp, delimiter=", ")

    def vehpos2latlon(self, x, y, lat_origin, lon_origin):
        if lat_origin <= 10:
            lat_origin = GridPoly.rad2deg(lat_origin)
            lon_origin = GridPoly.rad2deg(lon_origin)
        lat = lat_origin + GridPoly.rad2deg(x * np.pi * 2.0 / GridPoly.circumference)
        lon = lon_origin + GridPoly.rad2deg(y * np.pi * 2.0 / (GridPoly.circumference * np.cos(GridPoly.deg2rad(lat))))
        return lat, lon

class Pre_surveyor(AUV, DataAssimilator):
    '''
    Calibrate the prior based on the linear regression
    '''
    # data_path = "Prior_polygon.h5"
    grid_path = "grid.txt"

    waypoints = np.array([[41.1, -8.735],
                          [41.106, -8.745],
                          [41.106, -8.730],
                          [41.112, -8.730]])

    def __init__(self, debug = False):
        AUV.__init__(self)
        DataAssimilator.__init__(self)
        self.Pre_surveyor()
        print("Hi, I am prior mate, what's up")

    def Pre_surveyor(self):
        counter_waypoint = 0
        self.auv_handler.setWaypoint(self.deg2rad(self.waypoints[counter_waypoint, 0]),
                                     self.deg2rad(self.waypoints[counter_waypoint, 1]))
        while not rospy.is_shutdown():
            if self.init:
                self.append_salinity(self.currentSalinity)
                self.append_temperature(self.currentTemperature)
                lat_temp, lon_temp = self.vehpos2latlon(self.vehicle_pos[0], self.vehicle_pos[1], self.lat_origin,
                                                        self.lon_origin)
                self.append_path([lat_temp, lon_temp, self.vehicle_pos[2]])
                print(self.auv_handler.getState())
                if self.auv_handler.getState() == "waiting" and self.last_state != "waiting":
                    print("Arrived the current location")
                    counter_waypoint = counter_waypoint + 1
                    if counter_waypoint >= len(self.waypoints):
                        rospy.signal_shutdown("Mission completed!!!")
                        break
                    # Move to the next waypoint
                    self.auv_handler.setWaypoint(self.deg2rad(self.waypoints[counter_waypoint, 0]),
                                                 self.deg2rad(self.waypoints[counter_waypoint, 1]))

                    self.save_data()

                self.last_state = self.auv_handler.getState()
                self.auv_handler.spin()
            self.rate.sleep()

class Prior_mate(Pre_surveyor):
    path = None
    def __init__(self):
        self.data_prior_path = "Prior_polygon.h5"
        self.data_path_mission = "Data/Pre_survey_data_on_2021_0915_1520"
        self.path = np.loadtxt(self.data_path_mission + "/data_path.txt", delimiter=", ")
        self.salinity_auv = np.loadtxt(self.data_path_mission + "/data_salinity.txt", delimiter=", ")
        self.load_prior()
        self.prior_calibrator()

    def load_prior(self):
        self.prior_data = h5py.File(self.data_prior_path, 'r')
        self.lat_prior = self.prior_data.get("lat_selected")
        self.lon_prior = self.prior_data.get("lon_selected")
        self.depth_prior = self.prior_data.get("depth_selected")
        self.salinity_prior = self.prior_data.get("salinity_selected")

    def getPriorIndAtLoc(self, loc):
        lat, lon, depth = loc
        distDepth = np.abs(self.depth_obs - depth)
        ind_depth = np.where(distDepth == distDepth.min())[0][0]
        distLat = self.lat_prior[:, ind_depth] - lat
        distLon = self.lon_prior[:, ind_depth] - lon
        dist = np.sqrt(distLat ** 2 + distLon ** 2)
        ind_loc = np.where(dist == dist.min())[0][0]
        return [ind_loc, ind_depth]

    def prior_calibrator(self):
        self.lat_auv = self.path[:, 0]
        self.lon_auv = self.path[:, 1]
        self.depth_auv = self.path[:, 2]

        self.salinity_prior_reg = []
        for i in range(len(self.lat_auv)):
            ind_loc, ind_depth = self.getPriorIndAtLoc([self.lat_auv[i], self.lon_auv[i], self.depth_auv[i]])
            self.salinity_prior_reg.append(self.salinity_prior[ind_loc, ind_depth])
        self.salinity_prior_reg = np.array(self.salinity_prior_reg).reshape(-1, 1)
        X = np.hstack((np.ones_like(self.salinity_prior_reg), self.salinity_prior_reg))
        Y = self.salinity_auv
        self.beta = np.linalg.solve(X.T @ X, (X.T @ Y))

        self.salinity_prior_corrected = self.beta[0] + self.beta[1] * self.salinity_prior
        np.savetxt(self.data_path_mission + "/prior_corrected.txt", self.salinity_prior_corrected, delimiter=", ")


if __name__ == "__main__":
    # a = Pre_surveyor()
    b = Prior_mate()



