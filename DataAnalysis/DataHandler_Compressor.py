# ! /usr/bin/env python3
__author__ = "Yaolin Ge"
__copyright__ = "Copyright 2021, The MASCOT Project, NTNU (https://wiki.math.ntnu.no/mascot)"
__credits__ = ["Yaolin Ge"]
__license__ = "MIT"
__version__ = "1.0.1"
__maintainer__ = "Yaolin Ge"
__email__ = "yaolin.ge@ntnu.no"
__status__ = "UnderDevelopment"


import os
import numpy as np
import time
import h5py



class DataCompressor:
    path_data = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Prior/Nov_Prior/Merged_all/"
    path_OpArea = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Adaptive_script/Porto/Onboard/Config/OperationArea.txt"

    def __init__(self):
        self.load_OpArea()
        self.load_merged_data()
        pass

    def load_OpArea(self):
        self.operational_area = np.loadtxt(self.path_OpArea, delimiter = ", ")

        # self.operational_area = self.operational_area[, :]
        print(self.operational_area)

    def create_bigger_box(self):
        self.box = np.array([[]])

    def load_merged_data(self):
        files = os.listdir(self.path_data)
        print(files)
        for file in files:
            if file.endswith(".h5"):
                self.data = h5py.File(self.path_data + file)
                self.lat = np.array(self.data.get("lat"))
                self.lon = np.array(self.data.get("lon"))
                self.depth = np.array(self.data.get("depth"))
                self.salinity = np.array(self.data.get("salinity"))

                print(self.data)

if __name__ == "__main__":
    a = DataCompressor()


import matplotlib.pyplot as plt
plt.scatter(a.lon[:, :, 0], a.lat[:, :, 0], c = a.salinity[:, :, 0])
plt.plot(a.operational_area[:, 1], a.operational_area[:, 0])
plt.show()
