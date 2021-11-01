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

class MissionComplete(AUV):
    def __init__(self):
        AUV.__init__(self)
        self.send_SMS_mission_complete()

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
    e = MissionComplete()



