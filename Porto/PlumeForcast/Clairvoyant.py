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
import urllib.request
import matplotlib.pyplot as plt
from dateutil import rrule
from datetime import datetime
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'font.style': 'oblique'})

server = True
if not server:
    path_data_new = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Prior/Maretec/Data/"
    figpath = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Porto/PlumeForcast/fig/"
else:
    path_data_new = "/home/ahomea/y/yaoling/MASCOT/Porto_Data_Processing/MOHID/Data/"
    figpath = "/home/ahomea/y/yaoling/MASCOT/Porto_Data_Processing/MOHID/fig/"


class DataFetcher:
    string_start = '20210915'
    string_end = '20211120' # this needs to be two days more than the current day since it needs to populate more and cut off one day

    def __init__(self):
        pass

    def get_all_date(self):
        '''
        get all the available date
        '''
        from dateutil import rrule
        from datetime import datetime
        self.string_dates = []
        for dt in rrule.rrule(rrule.DAILY,dtstart=datetime.strptime(self.string_start, '%Y%m%d'),
                              until=datetime.strptime(self.string_end, '%Y%m%d')):
            self.string_dates.append(dt.strftime('%Y%m%d'))

    def fetch_data(self):
        self.get_all_date()
        print(self.string_dates)
        for i in range(len(self.string_dates)):
            if i < len(self.string_dates) - 1:
                string_date1 = self.string_dates[i]
                year1 = string_date1[:4]
                month1 = string_date1[4:6]
                day1 = string_date1[6:]
                string_date2 = self.string_dates[i+1]
                year2 = string_date2[:4]
                month2 = string_date2[4:6]
                day2 = string_date2[6:]
                print('ftp://Renato:REP_Modelling@ftp.mohid.com/CoLAB_Atlantic/LSTS/Douro/'+year1+'-'+month1+'-'+day1+'_'+year2+'-'+month2+'-'+day2+'/WaterProperties.hdf5',
                                           path_data_new + 'WaterProperties_'+string_date1+'.hdf5')
                try:
                    urllib.request.urlretrieve('ftp://Renato:REP_Modelling@ftp.mohid.com/CoLAB_Atlantic/LSTS/Douro/'+year1+'-'+month1+'-'+day1+'_'+year2+'-'+month2+'-'+day2+'/WaterProperties.hdf5',
                                               path_data_new + 'WaterProperties_'+string_date1+'.hdf5')
                except:
                    print("Something wrong, but I ignored")
                    pass
                print(i)
        print("data is fetched successfully!")


class Clairvoyant:

    def __init__(self):
        pass

    def loaddata(self, file):
        print("Now it will load the Maretec data...")
        t1 = time.time()
        self.data = h5py.File(path_data_new + file, 'r')
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
        while os.path.exists(figpath + "P%s" % i):
            i += 1
        self.figpath = figpath + "P%s" % i
        if not os.path.exists(self.figpath):
            print(self.figpath + " is created")
            os.mkdir(self.figpath)
        else:
            print(self.figpath + " is already existed")

    def visualiseData(self):
        self.checkFolder()
        print("Here it comes the plotting for the updated results from Maretec.")
        files = os.listdir(path_data_new)
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

if __name__ == "__main__":
    a = DataFetcher()
    a.fetch_data()

    b = Clairvoyant()
    b.visualiseData()


