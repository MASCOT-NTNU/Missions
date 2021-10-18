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

class MessageHandler:
    phone_number = "+351969459285"
    # phone_number = "+4792526858"
    sms_pub_ = rospy.Publisher("/IMC/In/Sms", Sms, queue_size=10)

    def send_SMS(self, lat, lon):
        print("Message has been sent to: ", self.phone_number)
        SMS = Sms()
        SMS.number.data = self.phone_number
        SMS.timeout.data = 60
        SMS.contents.data = "LAUV-Xplore-1 location: " + str(lat) + ", " + str(lon)
        self.sms_pub_.publish(SMS)


    def send_SMS_mission_complete(self, lat, lon):
        print("Mission complete Message has been sent to: ", self.phone_number)
        SMS = Sms()
        SMS.number.data = self.phone_number
        SMS.timeout.data = 60
        SMS.contents.data = "Congrats, Mission complete. LAUV-Xplore-1 location: " + str(lat) + ", " + str(lon)
        self.sms_pub_.publish(SMS)


