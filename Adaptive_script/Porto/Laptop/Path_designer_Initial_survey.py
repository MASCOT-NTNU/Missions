#! /usr/bin/env python3
__author__ = "Yaolin Ge"
__copyright__ = "Copyright 2021, The MASCOT Project, NTNU (https://wiki.math.ntnu.no/mascot)"
__credits__ = ["Yaolin Ge"]
__license__ = "MIT"
__version__ = "1.0.1"
__maintainer__ = "Yaolin Ge"
__email__ = "yaolin.ge@ntnu.no"
__status__ = "UnderDevelopment"

import simplekml
import time
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'font.style': 'oblique'})

'''
This path designer will design the path for the initial survey
'''

class PathDesigner:
    '''
    Prior1 is built based on the surface data and wind correaltion
    '''
    data_path = None
    path_onboard = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Adaptive_script/Porto/Onboard/"
    debug = True

    def __init__(self, debug = False):
        self.debug = debug
        self.load_grid()
        self.load_polygon()
        self.plot_3d_prior()
        self.design_path()
        self.checkPath()
        self.saveKML()

    def load_grid(self):
        print("Loading grid...")
        self.grid_poly = np.loadtxt(self.path_onboard + "grid.txt", delimiter=", ")
        print("grid is loaded successfully, grid shape: ", self.grid_poly.shape)

    def load_polygon(self):
        print("Loading the polygon...")
        self.polygon = np.loadtxt(self.path_onboard + "polygon.txt", delimiter=", ")
        print("Finished polygon loading, polygon: ", self.polygon.shape)

    def plot_3d_prior(self):
        import plotly.graph_objects as go
        import plotly
        prior = self.path_onboard + "Prior_polygon.txt"
        data_prior = np.loadtxt(prior, delimiter=", ")
        depth_prior = data_prior[:, 2]
        lat_prior = data_prior[:, 0]
        lon_prior = data_prior[:, 1]
        salinity_prior = data_prior[:, -1]
        fig = go.Figure(data=[go.Scatter3d(
            x=lon_prior.squeeze(),
            y=lat_prior.squeeze(),
            z=depth_prior.squeeze(),
            mode='markers',
            marker=dict(
                size=12,
                color=salinity_prior.squeeze(),  # set color to an array/list of desired values
                showscale=True,
                coloraxis="coloraxis"
            )
        )])
        fig.update_coloraxes(colorscale="jet")
        figpath = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Porto/Setup/"
        plotly.offline.plot(fig, filename=figpath + "Prior.html", auto_open=True)

    def design_path(self):
        print("Please design the path, every second node will be pop up, use the yoyo pattern")
        # ind_grid_surface = int(len(self.grid_poly) / 3)
        close_polygon = self.polygon[0, :]
        self.polygon = np.append(self.polygon, close_polygon.reshape(1, -1), axis = 0)
        plt.figure(figsize=(10, 10))
        plt.plot(self.grid_poly[:, 1], self.grid_poly[:, 0], 'k.')
        plt.plot(self.polygon[:, 1], self.polygon[:, 0], 'r-')
        plt.xlabel("Lon [deg]")
        plt.ylabel("Lat [deg]")
        path_initialsurvey = plt.ginput(n=100, timeout = 0)  # wait for the click to select the polygon
        plt.show()
        self.path_initial_survey = []
        for i in range(len(path_initialsurvey)):
            if i % 2 == 0:
                self.path_initial_survey.append([path_initialsurvey[i][1], path_initialsurvey[i][0], 0])
            else:
                self.path_initial_survey.append([path_initialsurvey[i][1], path_initialsurvey[i][0], np.amin(self.grid_poly[:, 2])])
        self.path_initial_survey = np.array(self.path_initial_survey)
        np.savetxt(self.path_onboard + "path_initial_survey.txt", self.path_initial_survey, delimiter=", ")
        print("The initial survey path is designed successfully, path_initial_survey: ", self.path_initial_survey.shape)

    def saveKML(self):
        print("I will create a polygon kml file for importing...")
        with open(self.path_onboard + "path_initial_survey.txt", "r") as a_file:
            points = []
            for line in a_file:
                stripped_line = line.strip()
                coordinates = stripped_line.split(",")
                points.append((coordinates[1], coordinates[0]))
        kml = simplekml.Kml()
        pol = kml.newpolygon(name='A Polygon')
        pol.outerboundaryis = points
        pol.innerboundaryis = points
        kml.save(self.path_onboard + "Path_initial_survey.kml")
        print("Path_initial_survey.kml is created successfully")

    def checkPath(self):
        plt.figure(figsize=(10, 10))
        plt.plot(self.grid_poly[:, 1], self.grid_poly[:, 0], 'k.', label = "Grid graph")
        plt.plot(self.polygon[:, 1], self.polygon[:, 0], 'r-', label = "Operaional boundary")
        plt.plot(self.path_initial_survey[:, 1], self.path_initial_survey[:, 0], 'k-', label = "Initial Survey Path")
        plt.plot(self.path_initial_survey[0:-1:2, 1], self.path_initial_survey[0:-1:2, 0], 'b*', label = "Surfacing")
        plt.plot(self.path_initial_survey[1:-1:2, 1], self.path_initial_survey[1:-1:2, 0], 'g*', label = "Diving")
        plt.title("Initial path visualisation")
        plt.xlabel("Lon [deg]")
        plt.ylabel("Lat [deg]")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    a = PathDesigner()


