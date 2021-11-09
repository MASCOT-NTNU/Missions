#!/usr/bin/env python3
__author__ = "Yaolin Ge"
__copyright__ = "Copyright 2021, The MASCOT Project, NTNU (https://wiki.math.ntnu.no/mascot)"
__credits__ = ["Yaolin Ge"]
__license__ = "MIT"
__version__ = "1.0.1"
__maintainer__ = "Yaolin Ge"
__email__ = "yaolin.ge@ntnu.no"
__status__ = "UnderDevelopment"


'''
Objective: Processing the data collected from the pre_survey and conduct the preprocessing before the adaptive mission
'''
import rospy
from auv_handler import AuvHandler
from imc_ros_interface.msg import Temperature, Salinity, EstimatedState, Sms
from usr_func import *

from PostProcessor import PostProcessor
from PolygonHandler import PolygonCircle
from GridHandler import GridPoly
from PreProcessor import PreProcessor
from AUV import AUV
from Orienteer import Orienteer

class MissionComplete(AUV):
    def __init__(self):
        AUV.__init__(self)
        self.send_SMS_mission_complete()

    def load_global_path(self):
        print("Now it will load the global path.")
        self.path_global = open("path_global.txt", 'r').read()
        print("global path is set up successfully!")
        print(self.path_global)

    def get_starting_loc(self):
        self.prior_extracted_path = self.path_global + "/Data/Corrected/Prior_extracted.txt"
        self.data_prior = np.loadtxt(self.prior_extracted_path, delimiter=", ")
        self.lat_loc = self.data_prior[:, 0]
        self.lon_loc = self.data_prior[:, 1]
        self.depth_loc = self.data_prior[:, 2]

        self.ind_start = np.loadtxt(self.path_global + "/Config/ind_start.txt", delimiter=", ")
        print("ind_start: ", self.ind_start)
        self.lat_start = self.lat_loc[self.ind_start]
        self.lon_start = self.lon_loc[self.ind_start]
        self.depth_start = self.depth_loc[self.ind_start]
        print("Starting locaiton: ", self.lat_start, self.lon_start, self.depth_start)

    def send_SMS_starting_loc(self):
        print("Starting location has been sent to: ", self.phone_number)
        SMS = Sms()
        SMS.number.data = self.phone_number
        SMS.timeout.data = 60
        SMS.contents.data = ""
        self.sms_pub_.publish(SMS)

    def send_SMS_mission_complete(self):
        print("Mission complete Message has been sent to: ", self.phone_number)
        SMS = Sms()
        SMS.number.data = self.phone_number
        SMS.timeout.data = 60
        SMS.contents.data = "Congrats, PostProcessing and PreProcessing Mission complete."
        self.sms_pub_.publish(SMS)

if __name__ == "__main__":
    a = PostProcessor()
    b = PolygonCircle()
    c = GridPoly()
    d = PreProcessor()
    e = Orienteer()
    f = MissionComplete()




