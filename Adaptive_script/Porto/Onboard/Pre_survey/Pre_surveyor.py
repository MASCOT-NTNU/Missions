#!/usr/bin/env python3
__author__ = "Yaolin Ge"
__copyright__ = "Copyright 2021, The MASCOT Project, NTNU (https://wiki.math.ntnu.no/mascot)"
__credits__ = ["Yaolin Ge"]
__license__ = "MIT"
__version__ = "1.0.1"
__maintainer__ = "Yaolin Ge"
__email__ = "yaolin.ge@ntnu.no"
__status__ = "UnderDevelopment"

import os
import rospy
from auv_handler import AuvHandler
from imc_ros_interface.msg import Temperature, Salinity, EstimatedState, Sms
import time
from datetime import datetime
from AUV import AUV
from DataHandler import DataHandler
from usr_func import *

'''
The following modules are used for postprocessing before the adaptive mission
'''

class PreSurveyor(AUV, DataHandler):

    def __init__(self):
        AUV.__init__(self)
        self.load_global_path()
        self.load_path_initial_survey()
        self.Run()

    def load_global_path(self):
        print("Now it will load the global path.")
        self.path_global = open("path_global.txt", 'r').read()
        print("global path is set up successfully!")
        print(self.path_global)

    def load_path_initial_survey(self):
        print("Loading the initial survey path...")
        self.path_initial_survey = np.loadtxt(self.path_global + "/Config/path_initial_survey.txt", delimiter=", ")
        print("Initial survey path is loaded successfully, path_initial_survey: ", self.path_initial_survey)

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

    def send_SMS_mission_complete(self, lat, lon):
        print("Mission complete Message has been sent to: ", self.phone_number)
        SMS = Sms()
        SMS.number.data = self.phone_number
        SMS.timeout.data = 60
        SMS.contents.data = "Congrats, Pre Survey Mission complete. LAUV-Xplore-1 location: " + str(lat) + ", " + str(lon)
        self.sms_pub_.publish(SMS)

    def getVehiclePos(self):
        x_auv = self.vehicle_pos[0]  # x distance from the origin
        y_auv = self.vehicle_pos[1]  # y distance from the origin
        lat_auv, lon_auv = self.vehpos2latlon(x_auv, y_auv, self.lat_origin, self.lon_origin)
        return lat_auv, lon_auv, self.vehicle_pos[2]

    def move_to_next_waypoint(self):
        # Move to the next waypoint
        self.counter_waypoint = self.counter_waypoint + 1 # should not be changed to the other order, since it can damage the reference
        self.update_waypoint()
        print("Now is: ", self.waypoint_lat_now, self.waypoint_lon_now, self.waypoint_depth_now)
        print("speed: ", self.speed)
        self.auv_handler.setWaypoint(self.waypoint_lat_now, self.waypoint_lon_now, self.waypoint_depth_now, speed = self.speed)

    def update_waypoint(self):
        self.waypoint_lat_now = deg2rad(self.path_initial_survey[self.counter_waypoint, 0]) # convert to rad so it works
        self.waypoint_lon_now = deg2rad(self.path_initial_survey[self.counter_waypoint, 1])
        self.waypoint_depth_now = self.path_initial_survey[self.counter_waypoint, 2]

    def Run(self):
        self.createDataPath(self.path_global)
        print("Now it will move to the starting location...")
        self.t1 = time.time()
        self.counter_waypoint = 0
        self.counter_data_saved = 0
        self.update_waypoint()
        self.move_to_starting_location = True
        self.auv_handler.setWaypoint(self.waypoint_lat_now, self.waypoint_lon_now, self.waypoint_depth_now,speed=self.speed)
        # self.auv_handler.setWaypoint(self.waypoint_lat_now, self.waypoint_lon_now, self.waypoint_depth_now) # continue with the current waypoint

        while not rospy.is_shutdown():

            if self.init:
                self.t2 = time.time()
                print("Elapsed time: ", self.t2 - self.t1)
                self.append_mission_data()
                self.save_mission_data()
                print(self.auv_handler.getState())

                if (self.t2 - self.t1) / 600 >= 1 and (self.t2 - self.t1) % 600 >= 0:
                    print("Longer than 10 mins, need a long break")
                    self.auv_handler.PopUp(sms=True, iridium=True, popup_duration=90,
                                           phone_number=self.phone_number,
                                           iridium_dest=self.iridium_destination)  # self.ada_state = "surfacing"
                    self.t1 = time.time() # restart the counter for time
                    self.t2 = time.time()

                # if self.auv_handler.getState() == "waiting":
                if self.auv_handler.getState() == "waiting" and self.last_state != "waiting":
                    print("Arrived the current location")
                    self.move_to_starting_location = False
                    if self.counter_waypoint + 1 >= len(self.path_initial_survey):
                        x_auv = self.vehicle_pos[0]  # x distance from the origin
                        y_auv = self.vehicle_pos[1]  # y distance from the origin
                        lat_auv, lon_auv = self.vehpos2latlon(x_auv, y_auv, self.lat_origin, self.lon_origin)
                        self.auv_handler.PopUp(sms=True, iridium=True, popup_duration=30,
                                               phone_number=self.phone_number,
                                               iridium_dest=self.iridium_destination)  # self.ada_state = "surfacing"
                        self.auv_handler.setWaypoint(self.waypoint_lat_now, self.waypoint_lon_now, 0, speed = self.speed)
                        self.send_SMS_mission_complete(lat_auv, lon_auv)
                        rospy.signal_shutdown("Mission completed!!!")
                        break
                    else:
                        if not self.move_to_starting_location:
                            self.move_to_next_waypoint()
                self.last_state = self.auv_handler.getState()
                self.auv_handler.spin()
            self.rate.sleep()

if __name__ == "__main__":
    a = PreSurveyor()
    pass




