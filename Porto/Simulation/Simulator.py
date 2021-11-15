#!/usr/bin/env python3
__author__ = "Yaolin Ge"
__copyright__ = "Copyright 2021, The MASCOT Project, NTNU (https://wiki.math.ntnu.no/mascot)"
__credits__ = ["Yaolin Ge"]
__license__ = "MIT"
__version__ = "1.0.1"
__maintainer__ = "Yaolin Ge"
__email__ = "yaolin.ge@ntnu.no"
__status__ = "UnderDevelopment"


import time
import os
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import plotly

class Simulator:
    path_data_global = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Porto/Simulation/Data/"
    path_adaptive = path_data_global + "Adaptive/"
    path_presurvey = path_data_global + "PreSurvey/"
    path_config = path_data_global + "Config/"
    path_data = path_data_global + "Data/"
    figpath = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Porto/Simulation/fig/"

    def __init__(self):
        print("Hello")
        self.load_prior()
        # self.load_adaptive_data()

    def load_adaptive_data(self, file):
        self.data_salinity = np.loadtxt(self.path_adaptive + file + "/data_salinity.txt", delimiter = ", ")
        self.data_waypoints = np.loadtxt(self.path_adaptive + file + "/data_path.txt", delimiter=", ")
        self.data_timestamp = np.loadtxt(self.path_adaptive + file + "/data_timestamp.txt", delimiter=", ")
        print("Adaptive mission data is loaded successfully!")
        pass

    def load_presurvey_data(self, file):
        print("PreSurvey data is loaded successfully!")
        pass

    def load_prior(self):
        self.prior_corrected = np.loadtxt(self.path_data + "Prior_corrected.txt", delimiter=", ")
        self.prior_extracted = np.loadtxt(self.path_data + "Prior_extracted.txt", delimiter=", ")
        print("Prior is loaded successfully!")
        pass

    def visualiseAdaptive(self):
        files = sorted(os.listdir(self.path_adaptive))
        file = files[-1]
        print(file)
        self.load_adaptive_data(file)
        # for file in files:
        #     if file.startswith("MissionData_on_"):
        #         self.load_adaptive_data(file)
        #
        #     break

    def visualisePrior(self):
        X = self.prior_corrected[:, 1]
        Y = self.prior_corrected[:, 0]
        Z = self.prior_corrected[:, 2]
        values = self.prior_corrected[:, 3]

        fig = go.Figure(data=go.Scatter3d(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            mode = "markers",
            marker=dict(
                size=12,
                color=values.flatten(),  # set color to an array/list of desired values
                colorscale='Viridis',  # choose a colorscale
                opacity=0.8
            )
        ))
        plotly.offline.plot(fig, filename=self.figpath + "Prior.html", auto_open=True)


if __name__ == "__main__":
    a = Simulator()
    a.visualiseAdaptive()
    a.visualisePrior()




