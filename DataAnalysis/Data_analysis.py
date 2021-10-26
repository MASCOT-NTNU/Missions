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

import numpy as np
from matplotlib.gridspec import GridSpec
import plotly.graph_objects as go
import plotly
plotly.io.orca.config.executable = '/usr/local/bin/orca'
plotly.io.orca.config.save()
from plotly.subplots import make_subplots
from datetime import datetime
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'font.style': 'oblique'})
import pandas as pd
from DataAnalysis.usr_func import *

class AUVData:
    path_data = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Adaptive_script/Porto/Onboard/Data/"
    path_global = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Adaptive_script/Porto/Onboard/"
    figpath = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Porto/Simulation/fig/"

    def __init__(self):
        print("hello;-)")
        # self.load_pre_survey()
        self.load_config()
        self.load_adaptive_mission()
        # self.plot_data_in3D(self.waypoint_adaptive, self.salinity_adaptive)
        # self.plot_data_in3D(self.waypoint_adaptive, self.salinity_adaptive)

    def load_config(self):
        path_config = self.path_global + "Config/"
        self.grid = np.loadtxt(path_config + "grid.txt", delimiter=", ")
        self.beta = np.loadtxt(path_config + "beta.txt", delimiter=", ")
        self.counter_waypoint = np.loadtxt(path_config + "counter_waypoint_Mission.txt", delimiter=", ")
        self.OpArea = np.loadtxt(path_config + "OperationArea.txt", delimiter=", ")
        self.polygon_c = np.loadtxt(path_config + "polygon_centre.txt", delimiter=", ")
        self.polygon = np.loadtxt(path_config + "polygon.txt", delimiter=", ")
        self.threshold = np.loadtxt(path_config + "threshold.txt", delimiter=", ")
        self.mu_cond = np.loadtxt(path_config + "mu_cond.txt", delimiter=", ")
        self.Sigma_cond = np.loadtxt(path_config + "Sigma_cond.txt", delimiter=", ")
        print("finished with config")






    def load_adaptive_mission(self):
        datapath_adaptive = self.path_data + "MASCOT/"
        files = os.listdir(datapath_adaptive)
        # for file in files:
        #     if file == ".DS_Store":
        #         pass
        #     else:
        #         print(file)
        file = files[-3]
        print(file)
        datapath_adaptive = datapath_adaptive + file + "/"
        path_salinity = datapath_adaptive + "data_salinity.txt"
        path_timestamp = datapath_adaptive + "data_timestamp.txt"
        path_waypoint = datapath_adaptive + "data_path.txt"
        self.salinity_adaptive= np.loadtxt(path_salinity, delimiter=", ")
        self.waypoint_adaptive = np.loadtxt(path_waypoint, delimiter=", ")
        self.timestamp_adaptive = np.loadtxt(path_timestamp, delimiter=", ")
        ind_selected = np.where(self.salinity_adaptive >= 20)[0]
        self.salinity_adaptive = self.salinity_adaptive[ind_selected]
        self.waypoint_adaptive = self.waypoint_adaptive[ind_selected, :]
        self.timestamp_adaptive = self.timestamp_adaptive[ind_selected]
        print("finsihed")
        self.plot_data_in3D(self.waypoint_adaptive, self.salinity_adaptive, file)

    def load_pre_survey(self):
        datapath_pre_survey = self.path_data + "Pre_survey/"
        files = os.listdir(datapath_pre_survey)
        file = files[-1]
        datapath_pre_survey = datapath_pre_survey + file + "/"
        path_salinity = datapath_pre_survey + "data_salinity.txt"
        path_timestamp = datapath_pre_survey + "data_timestamp.txt"
        path_waypoint = datapath_pre_survey + "data_path.txt"
        self.salinity_presurvey = np.loadtxt(path_salinity, delimiter=", ")
        self.waypoint_presurvey = np.loadtxt(path_waypoint, delimiter=", ")
        self.timestamp_presurvey = np.loadtxt(path_timestamp, delimiter=", ")
        ind_selected = np.where(self.salinity_presurvey >= 20)[0]
        self.salinity_presurvey = self.salinity_presurvey[ind_selected]
        self.waypoint_presurvey = self.waypoint_presurvey[ind_selected, :]
        self.timestamp_presurvey = self.timestamp_presurvey[ind_selected]


    def plot_data_in3D(self, waypoint, salinity, file_string):
        import plotly.express as px
        lat = waypoint[:, 0]
        lon = waypoint[:, 1]
        depth = waypoint[:, 2]
        sal = salinity
        # Make 3D plot # #
        fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scene'}]])
        fig.add_trace(
            go.Scatter3d(
                x=lon.squeeze(), y=lat.squeeze(), z=depth.squeeze(),
                # mode='markers',
                marker=dict(
                    size=4,
                    color=sal.squeeze(),
                    colorscale=px.colors.qualitative.Light24,  # to have quantitified colorbars and colorscales
                    showscale=True
                ),
            ),
            row=1, col=1,
        )
        grid_lat = self.polygon[:, 0]
        grid_lon = self.polygon[:, 1]
        grid_depth = np.zeros_like(grid_lon)
        fig.add_trace(
            go.Scatter3d(
                x=grid_lon.squeeze(), y=grid_lat.squeeze(), z=grid_depth.squeeze(),
                # mode='markers',
                # marker=dict(
                    # size=4,
                    # color=sal.squeeze(),
                    # colorscale=px.colors.qualitative.Light24,  # to have quantitified colorbars and colorscales
                    # showscale=True
                # ),
            ),
            row=1, col=1,
        )
        grid_lat = self.grid[:, 0]
        grid_lon = self.grid[:, 1]
        grid_depth = self.grid[:, 2]
        fig.add_trace(
            go.Scatter3d(
                x=grid_lon.squeeze(), y=grid_lat.squeeze(), z=grid_depth.squeeze(),
                mode = "markers",
                marker=dict(
                size=2,
                # color=sal.squeeze(),
                # colorscale=px.colors.qualitative.Light24,  # to have quantitified colorbars and colorscales
                # showscale=True
                ),
            ),
            row=1, col=1,
        )

        fig.update_layout(
            scene={
                'aspectmode': 'manual',
                'xaxis_title': 'Lon [deg]',
                'yaxis_title': 'Lat [deg]',
                'zaxis_title': 'Depth [m]',
                'aspectratio': dict(x=1, y=1, z=.5),
            },
            showlegend=True,
            title="Simulation"
            # scene_camera_eye=camera,
        )
        plotly.offline.plot(fig, filename=self.figpath + "Data" + file_string + ".html", auto_open=True)
        # fig.write_image(self.figpath + "Scatter3D/S_{:04}.png".format(frame), width=1980, height=1080, engine = "orca")


if __name__ == "__main__":
    a = AUVData()
    t = a.salinity_adaptive
    t1 = a.timestamp_adaptive
    t2 = a.waypoint_adaptive


