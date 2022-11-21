import pandas as pd
from datetime import datetime
import numpy as np
import os


class WaterDischargeHandler:
    path_water_discharge = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/WaterDischarge/Data/douro_discharge_2015_2021.csv"
    path_new = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/WaterDischarge/"

    def __init__(self):
        print("Hello")
        self.load_water_discharge_data()
        self.export_csv2txt()

    def load_water_discharge_data(self):
        my_cols = ['time', 'water_discharge', 'flag', 'water_discharge2', 'flag2', 'none']
        self.data_water_discharge = pd.read_csv(self.path_water_discharge,
                                           sep=",",
                                           names=my_cols,
                                           header=None,
                                           engine="python",
                                           skiprows=4)
        self.data_water_discharge = self.data_water_discharge.iloc[:-2]  # remove last two rows
        self.data_water_discharge = self.data_water_discharge.iloc[:, 0:2]  # remove last columns

    def export_csv2txt(self):
        self.timestamp_water_discharge = []  # save timestamp
        self.water_discharge = []  # save data values
        for i in range(len(self.data_water_discharge)):
            print(i)
            self.timestamp_water_discharge.append(
                datetime.strptime(self.data_water_discharge.iloc[i, 0], '%d/%m/%Y %H:%M').timestamp())
            self.water_discharge.append(self.data_water_discharge.iloc[i, 1])
        self.timestamp_water_discharge = np.array(self.timestamp_water_discharge).reshape(-1, 1)
        self.water_discharge = np.array(self.water_discharge).reshape(-1, 1)

        data_wds = np.hstack((self.timestamp_water_discharge, self.water_discharge))
        np.savetxt(self.path_new + "data_water_discharge.txt", data_wds, delimiter=", ")
        print("data is saved sucessfully!")
        os.system("say Congrats!")

if __name__ == "__main__":
    a = WaterDischargeHandler()

