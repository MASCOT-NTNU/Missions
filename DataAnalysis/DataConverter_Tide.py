# ! /usr/bin/env python3
__author__ = "Yaolin Ge"
__copyright__ = "Copyright 2021, The MASCOT Project, NTNU (https://wiki.math.ntnu.no/mascot)"
__credits__ = ["Yaolin Ge"]
__license__ = "MIT"
__version__ = "1.0.1"
__maintainer__ = "Yaolin Ge"
__email__ = "yaolin.ge@ntnu.no"
__status__ = "UnderDevelopment"

from usr_func import *
import os
import pandas as pd
from datetime import datetime


class DataConverter_Tide:
    tide_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Tide/Data/"
    month_selected = 12  # December
                         # "Preia - Mar" --> high tide
                         # "Baixa - Mar" --> low tide

    def __init__(self):
        print("Tide handler is initialised successfully.")
        self.loaddata()
        self.select_data_with_month()
        self.extractEbb()

    def loaddata(self):
        self.year = []
        self.month = []
        self.day = []
        self.hour = []
        self.min = []
        self.tide_height = np.empty([0, 1])
        self.tide_type_numerical = np.empty([0, 1])
        for tide_file in os.listdir(self.tide_path):
            print(tide_file)
            self.temp = pd.read_csv(self.tide_path + tide_file, skiprows = 12, sep = "\t", header = None)
            self.temp = pd.DataFrame(self.temp[0].str.split('  ').tolist())
            self.year_month_day = np.array(self.temp.iloc[:, 0])
            self.hour_min = np.array(self.temp.iloc[:, 1])
            for i in range(len(self.hour_min)):
                ind_year = self.year_month_day[i].index('-')
                self.year.append(int(self.year_month_day[i][:ind_year]))
                ind_month = self.year_month_day[i][ind_year + 1:].index('-')
                self.month.append(int(self.year_month_day[i][ind_year + 1:][:ind_month]))
                self.day.append(int(self.year_month_day[i][ind_year + 1:][ind_month + 1:]))
                ind_hour = self.hour_min[i].index(":")
                self.hour.append(int(self.hour_min[i][:ind_hour]))
                self.min.append(int(self.hour_min[i][ind_hour + 1:]))
            self.tide_height = np.concatenate((self.tide_height, np.array(self.temp.iloc[:, 2]).astype(float).reshape(-1, 1)), axis = 0)
            self.tide_type = self.temp.iloc[:, 3] # tide type
            self.tide_type_numerical = np.concatenate((self.tide_type_numerical, np.array(self.tide_type == "Preia-Mar").astype(int).reshape(-1, 1)), axis = 0)

        self.year = np.array(self.year).reshape(-1, 1)
        self.month = np.array(self.month).reshape(-1, 1)
        self.day = np.array(self.day).reshape(-1, 1)
        self.hour = np.array(self.hour).reshape(-1, 1)
        self.min = np.array(self.min).reshape(-1, 1)

    def select_data_with_month(self):
        self.ind_selected = self.month == self.month_selected # tide in November months in the historical data
        self.year_selected = self.year[self.ind_selected].reshape(-1, 1)
        self.month_selected = self.month[self.ind_selected].reshape(-1, 1)
        self.day_selected = self.day[self.ind_selected].reshape(-1, 1)
        self.hour_selected = self.hour[self.ind_selected].reshape(-1, 1)
        self.min_selected = self.min[self.ind_selected].reshape(-1, 1)
        self.tide_height_selected = self.tide_height[self.ind_selected].reshape(-1, 1)
        self.tide_type_numerical_selected = self.tide_type_numerical[self.ind_selected].reshape(-1, 1)
        self.tide_timestamp = []
        for i in range(len(self.year_selected)):
            self.tide_timestamp.append(datetime(self.year_selected[i, 0], self.month_selected[i, 0], self.day_selected[i, 0],
                                                self.hour_selected[i, 0], self.min_selected[i, 0]).timestamp())
        self.tide_timestamp = np.array(self.tide_timestamp).reshape(-1, 1)
        self.data = np.hstack((self.tide_timestamp, self.tide_height_selected, self.tide_type_numerical_selected))
        np.savetxt(self.tide_path[:-5] + "tide.txt", self.data, delimiter=", ")

    def extractEbb(self):
        self.ebb_start = []
        self.ebb_end = []
        for i in range(len(self.tide_type_numerical_selected)):
            if self.tide_type_numerical_selected[i] == 1:
                if i < len(self.tide_type_numerical_selected) - 1:
                    if self.tide_type_numerical_selected[i + 1] == 0: # filter undesired state
                        self.ebb_start.append(self.tide_timestamp[i, 0])
                        self.ebb_end.append(self.tide_timestamp[i + 1, 0])
                    else:
                        pass
                else:
                    pass
        self.ebb_start = np.array(self.ebb_start).reshape(-1, 1)
        self.ebb_end = np.array(self.ebb_end).reshape(-1, 1)
        self.ebb = np.hstack((self.ebb_start, self.ebb_end))
        np.savetxt(self.tide_path[:-5] + "ebb_dec.txt", self.ebb, delimiter = ", ")

if __name__ == "__main__":
    a = DataConverter_Tide()
