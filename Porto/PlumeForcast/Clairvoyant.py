#!/usr/bin/env python3
__author__ = "Yaolin Ge"
__copyright__ = "Copyright 2021, The MASCOT Project, NTNU (https://wiki.math.ntnu.no/mascot)"
__credits__ = ["Yaolin Ge"]
__license__ = "MIT"
__version__ = "1.0.1"
__maintainer__ = "Yaolin Ge"
__email__ = "yaolin.ge@ntnu.no"
__status__ = "UnderDevelopment"

import numpy as np
import time
import h5py
import os
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'font.style': 'oblique'})


class Clairvoyant:
    path_maretec = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Prior/Maretec/Douro/"

    def __init__(self):
        pass

    def load_maretec_data(self, file):
        print("Now it will load the Maretec data...")
        t1 = time.time()
        self.data = h5py.File(self.path_maretec + file + "/WaterProperties.hdf5", 'r')
        self.grid = self.data.get('Grid')
        self.lat = np.array(self.grid.get("Latitude"))[:-1, :-1]
        self.lon = np.array(self.grid.get("Longitude"))[:-1, :-1]
        self.depth = []
        self.salinity = []
        for i in range(1, 26):
            string_z = "Vertical_{:05d}".format(i)
            string_sal = "salinity_{:05d}".format(i)
            self.depth.append(np.mean(np.array(self.grid.get("VerticalZ").get(string_z)), axis = 0))
            self.salinity.append(np.mean(np.array(self.data.get("Results").get("salinity").get(string_sal)), axis = 0))
        self.depth = np.array(self.depth)
        self.salinity = np.array(self.salinity)
        t2 = time.time()
        print("Maretec data is loaded correctly, time consumed: ", t2 - t1)

    def visualiseForcast(self):
        files = os.listdir(self.path_maretec)
        for file in files:
            if file.startswith("2021"):
                self.load_maretec_data(file)
                fig = plt.figure(figsize = (10, 10))
                ind_row = np.where(self.lon <= -8.6)[0]
                ind_col = np.where(self.lon <= -8.6)[1]
                print(ind_row)
                print(ind_col)
                break

class MaretecDataHandler:
    data_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Prior/Maretec/Douro/"
    figpath = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Porto/PlumeForcast/fig/"
    def __init__(self):

        # self.loaddata()
        # self.loadDelft3DSurface()
        # self.loadSatellite()
        # self.getsimilarDelft()
        # self.plotSurfaceData()
        pass

    def loaddata(self, file):
        print("Now it will load the Maretec data...")
        t1 = time.time()
        self.data = h5py.File(self.data_path + file +"/WaterProperties.hdf5", 'r')
        self.grid = self.data.get('Grid')
        self.lat = np.array(self.grid.get("Latitude"))[:-1, :-1]
        self.lon = np.array(self.grid.get("Longitude"))[:-1, :-1]
        self.depth = []
        self.salinity = []
        for i in range(1, 26):
            string_z = "Vertical_{:05d}".format(i)
            string_sal = "salinity_{:05d}".format(i)
            self.depth.append(np.mean(np.array(self.grid.get("VerticalZ").get(string_z)), axis = 0))
            self.salinity.append(np.mean(np.array(self.data.get("Results").get("salinity").get(string_sal)), axis = 0))
        self.depth = np.array(self.depth)
        self.salinity = np.array(self.salinity)
        t2 = time.time()
        print("Data is loaded correctly, time consumed: ", t2 - t1)

    def checkFolder(self):
        i = 0
        while os.path.exists(self.figpath + "P%s" % i):
            i += 1
        self.figpath = self.figpath + "P%s" % i
        if not os.path.exists(self.figpath):
            print(self.figpath + " is created")
            os.mkdir(self.figpath)
        else:
            print(self.figpath + " is already existed")

    def visualiseData(self):
        self.checkFolder()
        print("Here it comes the plotting for the updated results from Maretec.")
        files = os.listdir(self.data_path)
        files.sort()
        counter = 0
        for i in range(len(files)):
            if files[i] != ".DS_Store":
                print(files[i])
                self.loaddata(files[i])
                for j in range(self.salinity.shape[0]):
                    print(j)
                    plt.figure(figsize=(10, 10))
                    plt.scatter(self.lon[:self.lon.shape[1], :], self.lat[:self.lon.shape[1], :],
                                c=self.salinity[j, :self.lon.shape[1], :], vmin=26, vmax=36, cmap="Paired")
                    plt.colorbar()
                    plt.title("Surface salinity estimation from Maretec at time: {:02d}:00 during ".format(
                        j) + files[i])
                    plt.xlabel("Lon [deg]")
                    plt.ylabel("Lat [deg]")
                    plt.savefig(self.figpath + "/P_{:04d}.png".format(counter))
                    plt.close("all")
                    counter = counter + 1
                # plt.show()

    @staticmethod
    def deg2rad(deg):
        return deg / 180 * np.pi

if __name__ == "__main__":
    a = MaretecDataHandler()
    a.visualiseData()
    # a = Clairvoyant()
    # a.visualiseForcast()


