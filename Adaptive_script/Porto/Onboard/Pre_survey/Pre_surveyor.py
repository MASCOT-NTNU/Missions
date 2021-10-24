#! /usr/bin/env python3
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
import time
from datetime import datetime
from AUV import AUV
from DataHandler import DataHandler
from MessageHandler import MessageHandler
from usr_func import *


class PreSurveyor(AUV, DataHandler, MessageHandler):

    def __init__(self):
        AUV.__init__(self)
        self.load_global_path()
        self.load_path_initial_survey()
        self.check_pause()

        self.Run()

    def load_global_path(self):
        print("Now it will load the global path.")
        self.path_global = open("path_global.txt", 'r').read()
        print("global path is set up successfully!")
        print(self.path_global)



    def check_pause(self):
        self.get_resume_state()
        print(self.resume)

    def get_resume_state(self):
        print("Loading resume state...")
        self.resume = open(self.path_global + "/Config/ResumeState.txt", 'r').read()
        print("Loading resume state successfully! Resume state: ", self.resume)

    def save_resume_state(self, resume_state = 'False'):
        print("Saving resume state...")
        resume_state = open(self.path_global + "/Config/ResumeState.txt", 'w')
        resume_state.write(resume_state)
        resume_state.close()
        print("Resume state is saved successfully!" + resume_state)

    def save_counter_waypoint(self):
        print("Saving counter waypoint...")
        np.savetxt(self.path_global + "/Config/counter_waypoint.txt", self.counter_waypoint, delimiter=", ")
        print("Counter waypoint is saved! ", self.counter_waypoint)

    def load_counter_waypoint(self):
        print("Loading counter waypoint...")
        conterwaypoint = np.loadtxt(self.path_global + "/Config/counter_waypoint.txt", delimiter=", ")
        print("counter waypoint is loaded successfully! ", self.counter_waypoint)
        return conterwaypoint

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

    def surfacing(self, time_length):
        for i in range(time_length):
            lat_auv, lon_auv, depth_auv = self.getVehiclePos()
            if (i + 1) % 15 == 0:
                self.send_SMS(lat_auv, lon_auv) # send the location
                print("SMS has been sent: ", lat_auv, lon_auv)
            self.append_mission_data()
            self.save_mission_data()
            print("Sleep {:d} seconds".format(i))
            print("Now AUV is at: ", lat_auv, lon_auv, depth_auv)
            print("Desired loc: ", self.waypoint_lat_now, self.waypoint_lon_now, 0)
            self.auv_handler.setWaypoint(self.waypoint_lat_now, self.waypoint_lon_now, 0)
            self.auv_handler.spin()  # publishes the reference, stay on the surface
            self.rate.sleep()  #

    def getVehiclePos(self):
        x_auv = self.vehicle_pos[0]  # x distance from the origin
        y_auv = self.vehicle_pos[1]  # y distance from the origin
        lat_auv, lon_auv = self.vehpos2latlon(x_auv, y_auv, self.lat_origin, self.lon_origin)
        return lat_auv, lon_auv, self.vehicle_pos[2]

    def move_to_next_waypoint(self):
        # Move to the next waypoint
        self.counter_waypoint = self.counter_waypoint + 1 # should not be changed to the other order, since it can damage the reference
        self.update_waypoint()
        print("Now is: " ,self.waypoint_lat_now, self.waypoint_lon_now, self.waypoin_depth_now)
        self.auv_handler.setWaypoint(self.waypoint_lat_now, self.waypoint_lon_now, self.waypoin_depth_now)

    def update_waypoint(self):
        self.waypoint_lat_now = deg2rad(self.path_initial_survey[self.counter_waypoint, 0]) # convert to rad so it works
        self.waypoint_lon_now = deg2rad(self.path_initial_survey[self.counter_waypoint, 1])
        self.waypoin_depth_now = self.path_initial_survey[self.counter_waypoint, 2]

    def Run(self):
        self.createDataPath(self.path_global)
        print("Now it will move to the starting location...")
        self.t1 = time.time()
        self.counter_waypoint = self.load_counter_waypoint() # initialise the run
        self.counter_data_saved = 0
        self.update_waypoint()
        self.auv_handler.setWaypoint(self.waypoint_lat_now, self.waypoint_lon_now, self.waypoin_depth_now)

        while not rospy.is_shutdown():

            if self.init:
                self.t2 = time.time()
                print("Elapsed time: ", self.t2 - self.t1)
                self.append_mission_data()
                self.save_mission_data()
                print(self.auv_handler.getState())

                if (self.t2 - self.t1) / 600 >= 1 and (self.t2 - self.t1) % 600 >= 0:
                    print("Longer than 10 mins, need a long break")
                    self.surfacing(90)  # surfacing 90 seconds after 10 mins of travelling
                    self.t1 = self.t2
                    self.auv_handler.setWaypoint(self.waypoint_lat_now, self.waypoint_lon_now, self.waypoin_depth_now)

                if self.auv_handler.getState() == "waiting" and self.last_state != "waiting":
                    print("Arrived the current location")
                    if self.counter_waypoint + 1 >= len(self.path_initial_survey):
                        x_auv = self.vehicle_pos[0]  # x distance from the origin
                        y_auv = self.vehicle_pos[1]  # y distance from the origin
                        lat_auv, lon_auv = self.vehpos2latlon(x_auv, y_auv, self.lat_origin, self.lon_origin)
                        self.send_SMS_mission_complete(lat_auv, lon_auv)
                        rospy.signal_shutdown("Mission completed!!!")
                        break
                    self.move_to_next_waypoint()
                self.last_state = self.auv_handler.getState()
                self.auv_handler.spin()
            self.rate.sleep()

if __name__ == "__main__":
    a = PreSurveyor()
    pass




