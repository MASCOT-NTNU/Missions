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
import PySimpleGUI as sg
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
    circumference = 40075000
    debug = True

    def __init__(self, debug = False):
        self.debug = debug
        self.design_path()
        self.checkPath()
        self.saveKML()
        self.load_path_initial_survey()
        self.calculateDistacne()

    def GUI_query(self):
        layout = [[sg.Text('Please enter Mission date, ebb phase time duration, wind direction and wind speed')],
                  [sg.Text('Mission date', size=(10, 1)), sg.InputText(key='-Mission_date-')],
                  [sg.Text('Ebb phase duration', size=(10, 1)), sg.InputText(key='-Ebb-')],
                  [sg.Text('Wind direction', size=(10, 1)), sg.InputText(key='-Wind_dir-')],
                  [sg.Text('Wind level', size=(10, 1)), sg.InputText(key='-Wind_level-')],
                  [sg.Button('Submit'), sg.Button('Cancel')]]

        window = sg.Window('Simple Data Entry Window', layout)
        event, values = window.read(close=True)

        if event == 'Submit':
            print('The events was ', event, 'You input', values['-NAME-'], values['-ADDRESS-'], values['-PHONE-'])
        else:
            print('User cancelled')

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

    @staticmethod
    def deg2rad(deg):
        return deg / 180 * np.pi

    @staticmethod
    def rad2deg(rad):
        return rad / np.pi * 180

    @staticmethod
    def latlon2xy(lat, lon, lat_origin, lon_origin):
        x = PathDesigner.deg2rad((lat - lat_origin)) / 2 / np.pi * PathDesigner.circumference
        y = PathDesigner.deg2rad((lon - lon_origin)) / 2 / np.pi * PathDesigner.circumference * np.cos(PathDesigner.deg2rad(lat))
        return x, y

    def calculateDistacne(self):
        self.speed = 1.2
        self.lat_travel = self.path_initial_survey[:, 0]
        self.lon_travel = self.path_initial_survey[:, 1]
        self.depth_travel = self.path_initial_survey[:, 2]
        self.lat_pre = self.lat_travel[0]
        self.lon_pre = self.lon_travel[0]
        self.depth_pre = self.depth_travel[0]
        dist = 0
        for i in range(len(self.lat_travel)):
            x_temp, y_temp = self.latlon2xy(self.lat_travel[i], self.lon_travel[i], self.lat_pre, self.lon_pre)
            distZ = self.depth_travel[i] - self.depth_pre
            dist = dist + np.sqrt(x_temp ** 2 + y_temp ** 2 + distZ ** 2)
            self.lat_pre = self.lat_travel[i]
            self.lon_pre = self.lon_travel[i]
            self.depth_pre = self.depth_travel[i]
        print("Total distance needs to be travelled: ", dist)
        print("Time estimated: ", dist / self.speed)

    def load_path_initial_survey(self):
        print("Loading the initial survey path...")
        self.path_initial_survey = np.loadtxt(self.path_onboard + "path_initial_survey.txt", delimiter=", ")
        print("Initial survey path is loaded successfully, path_initial_survey: ", self.path_initial_survey)

if __name__ == "__main__":
    a = PathDesigner()


