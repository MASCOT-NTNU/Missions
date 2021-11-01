#! /usr/bin/env python3
__author__ = "Yaolin Ge"
__copyright__ = "Copyright 2021, The MASCOT Project, NTNU (https://wiki.math.ntnu.no/mascot)"
__credits__ = ["Yaolin Ge"]
__license__ = "MIT"
__version__ = "1.0.1"
__maintainer__ = "Yaolin Ge"
__email__ = "yaolin.ge@ntnu.no"
__status__ = "UnderDevelopment"

from Adaptive_script.Porto.Laptop.usr_func import *

import datetime
import simplekml
import PySimpleGUI as sg
import time
import os, sys
import h5py
import numpy as np
import plotly.graph_objects as go
import plotly
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
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
    data_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Prior/Maretec/Exemplo_Douro/2021-09-14_2021-09-15/WaterProperties.hdf5"
    # delft_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Prior/Sep_Prior/Merged_all/"
    # delft_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Prior/Nov_Prior/Merged_all/"
    delft_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Prior/Nov_Prior/Merged_all_extracted/"
    figpath = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Porto/Setup/Pre_survey/fig/"
    path_laptop = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Adaptive_script/Porto/Laptop/"
    path_onboard = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Adaptive_script/Porto/Onboard/"
    path_operational_area = path_onboard + "Config/OperationArea.txt"
    circumference = 40075000
    lat_river_mouth, lon_river_mouth = 41.139024, -8.680089
    # pitch_angle_AUV = 15 # [deg] pitch angle for AUV, used for setting YoYo
    distance_waypoint = 100 # [m], distance between each yoyo waypoint, no need to specify the angle
    distance_total = 4000 # [m], total distance for yoyo
    popup_interval = 10 # [min], interval between pop ups
    depth_bottom = 5 # [m], bottom depth layer for yoyo
    depth_top = 0.5 # [m], top depth layer for yoyo
    speed = 1.2 # [m/s], speed for AUV

    def __init__(self, debug = False):
        self.debug = debug
        if self.debug:
            self.DebugMode()
        else:
            self.GUI_query()
        self.load_all_data() # load all the essential data
        self.compute_gradient() # compute the gradient along
        self.get_optimal_transect_line() # get the optimal transect line
        self.design_path_initial() # design the optimal path
        self.plot_gradient_along_lines() # plot the gradient along designed lines
        self.save_all()
        # self.checkPath() # check those path to plot them on scatter 3d plot

    def DebugMode(self):
        self.string_date = "2021-09-23_2021-09-24"
        self.string_hour = "05_12"
        self.wind_dir = "North" # [North, East, West, South]
        self.wind_level = "Moderate" # [Mild, Moderate, Heavy]
        self.data_path = self.data_path[:81] + self.string_date + "/WaterProperties.hdf5"
        print("Mission date: ", self.string_date)
        print("Mission hour: ", self.string_hour)
        print("Mission wind conditions: ", self.wind_dir + " " + self.wind_level)

    def GUI_query(self):
        layout = [[sg.Text('Please enter Mission date, ebb phase time duration, wind direction and wind speed')],
                  [sg.Text('Mission month', size=(20, 1)),
                   sg.Listbox(values=["{:d}".format(i + 1) for i in range(12)], select_mode="extended",
                              key="mission_month", size=(5, 10)),
                   sg.Text("Mission day", size=(20, 1)),
                   sg.Listbox(values=["{:d}".format(i + 1) for i in range(31)], select_mode="extended",
                              key="mission_day", size=(5, 10))],
                  [sg.Text('Ebb phase start', size=(20, 1)),
                   sg.Listbox(values=["{:d}".format(i + 1) for i in range(24)], select_mode="extended",
                              key="ebb_phase_start", size=(5, 10)),
                   sg.Text('Ebb phase end', size=(20, 1)),
                   sg.Listbox(values=["{:d}".format(i + 1) for i in range(24)], select_mode="extended",
                              key="ebb_phase_end", size=(5, 10))],
                  [sg.Text('Wind direction', size=(20, 1)),
                   sg.Listbox(values=["North", "East", "South", "West"], select_mode="extended", key="wind_dir",
                              size=(10, 5))],
                  [sg.Text('Wind level', size=(20, 1)),
                   sg.Listbox(values=["Mild", "Moderate", "Heavy"], select_mode="extended", key="wind_level",
                              size=(10, 5))],
                  [sg.Button('Submit'), sg.Button('Cancel')]]

        window = sg.Window('Mission set up window', layout)
        event, values = window.read(close=True)

        if event == 'Submit':
            self.mission_month = values['mission_month'][0]
            self.mission_day = values['mission_day'][0]
            self.ebb_phase_start = values['ebb_phase_start'][0]
            self.ebb_phase_end = values['ebb_phase_end'][0]
            self.wind_dir = values['wind_dir'][0]
            self.wind_level = values['wind_level'][0]
            print("Mission date: ", self.mission_day + "/" + self.mission_month)
            print("Mission ebb phase: ", self.ebb_phase_start + ":" + self.ebb_phase_end)
            print("Mission wind condition: ", self.wind_dir + " " + self.wind_level)
            sg.popup("Confirmation\n" + "Mission date: {:s}/{:s}\nMission ebb phase: {:s}:{:s}\nMission wind condition: ".
                     format(self.mission_day, self.mission_month, self.ebb_phase_start, self.ebb_phase_end) + self.wind_dir +
                     " " + self.wind_level)
            self.string_date = "2021-{:02d}-{:02d}_2021-{:02d}-{:02d}".format(int(self.mission_month), int(self.mission_day),
                                                                              int(self.mission_month), int(self.mission_day) + 1)
            self.string_hour = "{:02d}_{:02d}".format(int(self.ebb_phase_start), int(self.ebb_phase_end))
            self.data_path = self.data_path[:81] + self.string_date + "/WaterProperties.hdf5"
        else:
            print('User cancelled')
            sys.exit()


    def load_all_data(self):
        self.load_operational_area()
        self.loaddata_maretec()
        self.loadDelft3D()

    def save_all(self):
        self.save_wind_condition()
        self.saveKML()

    def load_operational_area(self):
        print("Now I will load the operational area. ")
        self.OpArea = np.loadtxt(self.path_operational_area, delimiter=", ")
        print("Loading operational area successfully! ")

    def loaddata_maretec(self):
        print("Now it will load the Maretec data...")
        t1 = time.time()
        self.data = h5py.File(self.data_path, 'r')
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

    def loadDelft3D(self):
        print(self.delft_path)
        delft3dpath = self.delft_path + self.wind_dir + "_" + self.wind_level + "_all.h5"
        delft3d = h5py.File(delft3dpath, 'r')
        self.lat_delft3d = np.array(delft3d.get("lat"))[:, 0]
        self.lon_delft3d = np.array(delft3d.get("lon"))[:, 0]
        self.salinity_delft3d = np.array(delft3d.get("salinity"))[:, 0]
        # self.lat_delft3d = np.array(delft3d.get("lat"))[:, :, 0]
        # self.lon_delft3d = np.array(delft3d.get("lon"))[:, :, 0]
        # self.salinity_delft3d = np.array(delft3d.get("salinity"))[:, :, 0]
        print("Delft3D data is loaded successfully. ")
        print("lat_delft: ", self.lat_delft3d.shape)
        print("lon_delft: ", self.lon_delft3d.shape)
        print("salinity_delft: ", self.salinity_delft3d.shape)
        self.export_h5data2txt()

    def export_h5data2txt(self):
        print("Now I will export h5 file to txt.")
        t1 = time.time()
        delft3dpath = self.delft_path + self.wind_dir + "_" + self.wind_level + "_all.h5"
        delft3d = h5py.File(delft3dpath, 'r')
        self.lat_export = np.array(delft3d.get("lat")).reshape(-1, 1)
        self.lon_export = np.array(delft3d.get("lon")).reshape(-1, 1)
        self.depth_export = np.array(delft3d.get("depth")).reshape(-1, 1)
        self.salinity_export = np.array(delft3d.get("salinity")).reshape(-1, 1)
        self.dataset_export = np.hstack((self.lat_export, self.lon_export, self.depth_export, self.salinity_export))
        np.savetxt(self.path_onboard + "Data/Prior/" + "Delft3D_" + self.wind_dir + "_" + self.wind_level + ".txt", self.dataset_export, delimiter = ", ")
        t2 = time.time()
        print("Onboard Delft3D Data is created successfully! Time consumed: ", t2 - t1)

    def get_transect_lines(self):
        self.angles = np.arange(50, 75, 3) + 180
        r = 10000
        npoints = 100
        x = r * np.sin(deg2rad(self.angles))
        y = r * np.cos(deg2rad(self.angles))
        lat_end, lon_end = xy2latlon(x, y, self.lat_river_mouth, self.lon_river_mouth)
        lat_line = np.linspace(self.lat_river_mouth, lat_end, npoints).T
        lon_line = np.linspace(self.lon_river_mouth, lon_end, npoints).T
        return lat_line, lon_line

    def get_gradient_along_transect_line(self, lat_line, lon_line):
        sal_transect = np.empty_like(lat_line)
        for i in range(lat_line.shape[0]):
            ind = []
            for j in range(lat_line.shape[1]):
                ind.append(self.getDataIndAtLoc([lat_line[i, j], lon_line[i, j]]))
            sal_transect[i, :] = self.salinity_delft3d.reshape(-1, 1)[ind].squeeze()
        self.sal_transect = sal_transect

    def get_optimal_transect_line(self):
        sum_gradient = []
        for i in range(self.sal_transect.shape[0]):
            sum_gradient.append(np.sum(np.gradient(self.sal_transect[i, :])))
        # print(sum_gradient)
        self.sum_gradient = sum_gradient
        print("The optimal line is ", np.where(self.sum_gradient == np.nanmin(self.sum_gradient))[0][0])
        self.ind_optimal = np.where(self.sum_gradient == np.nanmax(self.sum_gradient))[0][0]

    def plot_gradient_along_lines(self):
        # grid = np.loadtxt("test.txt", delimiter=", ")

        fig = plt.figure(figsize=(20, 5))
        gs = GridSpec(ncols = 3, nrows = 1, figure = fig)
        ax = fig.add_subplot(gs[0])
        im = ax.scatter(self.lon_delft3d, self.lat_delft3d, c=self.salinity_delft3d, vmin=10, vmax=36, cmap="Paired")
        plt.colorbar(im)
        for i in range(self.lat_line.shape[0]):
            ax.plot(self.lon_line[i, :], self.lat_line[i, :], label = str(i))
        ax.plot(self.OpArea[:, 1], self.OpArea[:, 0], 'k-.', label = "Operational Region")
        # ax.plot(grid[:, 1], grid[:, 0], 'k.')
        ax.plot(self.lon_line[self.ind_optimal, -1], self.lat_line[self.ind_optimal, -1], 'b*', markersize = 20, label = "Desired path")
        ax.plot(self.lon_interval, self.lat_interval, 'k.')
        ax.set_title("Salinity field for " + self.wind_dir + " " + self.wind_level)
        ax.set(xlabel = "Lon [deg]", ylabel = "Lat [deg]")
        ax = fig.add_subplot(gs[1])
        for i in range(self.lat_line.shape[0]):
            ax.plot(self.sal_transect[i, :], label = str(i))
        ax.set_title("Salinity along the transect line for " + self.wind_dir + " " + self.wind_level)
        ax.set(xlabel = "Distance along transect line", ylabel = "Salinity")
        ax = fig.add_subplot(gs[2])
        for i in range(self.lon_line.shape[0]):
            ax.plot(np.gradient(self.sal_transect[i, :]), label = str(i))
        ax.set_title("Salinity gradient along the transect line for " + self.wind_dir + " " + self.wind_level)
        ax.set(xlabel="Distance along transect line", ylabel="Gradient")
        lgd = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(self.figpath + "TL_" + self.wind_dir + "_" + self.wind_level + "_" + str(self.lat_line.shape[0]) + ".png", bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.close("all")
        plt.show()

    def compute_gradient(self):
        self.lat_line, self.lon_line = self.get_transect_lines()
        self.get_gradient_along_transect_line(self.lat_line, self.lon_line)

    def getDataIndAtLoc(self, loc):
        lat_loc, lon_loc = loc
        distLat = self.lat_delft3d.reshape(-1, 1) - lat_loc
        distLon = self.lon_delft3d.reshape(-1, 1) - lon_loc
        dist = np.sqrt(distLat ** 2 + distLon ** 2)
        ind_loc = np.where(dist == np.nanmin(dist))[0][0]
        return ind_loc

    def design_path_initial(self):
        print("Now I will generate the initial YoYo path for preliminary survey")
        self.lat_end, self.lon_end = self.lat_line[self.ind_optimal, -1], self.lon_line[self.ind_optimal, -1]
        print("The end location is at: ", self.lat_end, self.lon_end)
        self.angle_optimal = self.angles[self.ind_optimal] # convert it back to the Cartisen frame
        x_temp = self.distance_total * np.sin(deg2rad(self.angle_optimal - 180))
        y_temp = self.distance_total * np.cos(deg2rad(self.angle_optimal - 180))
        self.lat_start, self.lon_start = xy2latlon(x_temp, y_temp, self.lat_end, self.lon_end) # starting location for yoyo
        self.interval = np.arange(0, self.distance_total, self.distance_waypoint)
        self.x_interval = self.interval * np.sin(deg2rad(self.angle_optimal))
        self.y_interval = self.interval * np.cos(deg2rad(self.angle_optimal))
        self.lat_interval, self.lon_interval = xy2latlon(self.x_interval, self.y_interval, self.lat_start, self.lon_start)

        self.path = []
        self.duration = self.popup_interval * 60 * self.speed
        self.path.append([self.lat_interval[0], self.lon_interval[0], 0])
        for i in range(len(self.lat_interval)):
            # if self.interval[i] % self.duration < 100:
            #     self.path.append([self.lat_interval[i], self.lon_interval[i], 0])
            # else:
            if i % 2 == 0:
                self.path.append([self.lat_interval[i], self.lon_interval[i], self.depth_top])
            else:
                self.path.append([self.lat_interval[i], self.lon_interval[i], self.depth_bottom])
        lat_finish, lon_finish = xy2latlon(-100, -100, self.lat_interval[-1], self.lon_interval[-1])
        self.path.append([lat_finish, lon_finish, 0])
        self.path_initial_survey = np.array(self.path)
        np.savetxt(self.path_onboard + "Config/path_initial_survey.txt", self.path_initial_survey, delimiter=", ")
        print("The initial survey path is designed successfully, path_initial_survey: ", self.path_initial_survey.shape)
        self.calculateDistacne()

    def calculateDistacne(self):
        self.lat_travel = self.path_initial_survey[:, 0]
        self.lon_travel = self.path_initial_survey[:, 1]
        self.depth_travel = self.path_initial_survey[:, 2]
        self.lat_pre = self.lat_travel[0]
        self.lon_pre = self.lon_travel[0]
        self.depth_pre = self.depth_travel[0]
        dist = 0
        for i in range(len(self.lat_travel)):
            x_temp, y_temp = latlon2xy(self.lat_travel[i], self.lon_travel[i], self.lat_pre, self.lon_pre)
            distZ = self.depth_travel[i] - self.depth_pre
            dist = dist + np.sqrt(x_temp ** 2 + y_temp ** 2 + distZ ** 2)
            self.lat_pre = self.lat_travel[i]
            self.lon_pre = self.lon_travel[i]
            self.depth_pre = self.depth_travel[i]
        print("Total distance needs to be travelled: ", dist)
        print("Time estimated: ", str(datetime.timedelta(seconds = dist / self.speed)))

    def saveKML(self):
        print("I will create a polygon kml file for importing...")
        with open(self.path_onboard + "Config/path_initial_survey.txt", "r") as a_file:
            points = []
            for line in a_file:
                stripped_line = line.strip()
                coordinates = stripped_line.split(",")
                points.append((coordinates[1], coordinates[0]))
        kml = simplekml.Kml()
        ls = kml.newlinestring(name='Path_initial_survey')
        ls.coords = points
        ls.extrude = 1
        ls.altitudemode = simplekml.AltitudeMode.relativetoground
        kml.save(self.path_onboard + "Import/Path_initial_survey.kml")
        print("Path_initial_survey.kml is created successfully")

    def save_wind_condition(self):
        f_wind = open(self.path_onboard + "Config/wind_condition.txt", 'w')
        f_wind.write("wind_dir=" + self.wind_dir + ", wind_level=" + self.wind_level)
        f_wind.close()
        print("wind_condition is saved successfully!")

    def checkPath(self):
        fig = go.Figure(data=[go.Scatter3d(x=self.path_initial_survey[:, 1], y=self.path_initial_survey[:, 0],
                                       z=-self.path_initial_survey[:, 2],
                                       marker=dict(size=12, color = "black"),line=dict(color='darkblue',width=2),)])
        fig.update_layout(
            scene={
                'aspectmode': 'manual',
                'xaxis_title': 'Lon [deg]',
                'yaxis_title': 'Lat [deg]',
                'zaxis_title': 'Depth [m]',
                'aspectratio': dict(x=1, y=1, z=.5),
            },
            showlegend=True,
            title="Initial survey path visualisation",
        )
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
        plotly.offline.plot(fig, filename=self.figpath + "Path.html", auto_open=True)


if __name__ == "__main__":
    a = PathDesigner(debug = True)
    # wind_dirs = ['North', 'South', 'West', 'East']  # get wind_data for all conditions
    # wind_levels = ['Mild', 'Moderate', 'Heavy']  # get data for all conditions
    # for wind_dir in wind_dirs:
    #     for wind_level in wind_levels:
    #         a.wind_dir = wind_dir
    #         a.wind_level = wind_level
    #         a.load_all_data()  # load all the essential data
    #         a.compute_gradient()  # compute the gradient along
    #         a.get_optimal_transect_line()  # get the optimal transect line
    #         a.design_path_initial()  # design the optimal path
    #         a.plot_gradient_along_lines()  # plot the gradient along designed lines

#%%
# import matplotlib.pyplot as plt
# import numpy as np
#
# path_laptop = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Adaptive_script/Porto/Laptop/"
# path_onboard = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Adaptive_script/Porto/Onboard/"
# path_operational_area = path_onboard + "Config/OperationArea.txt"
# path_presurvey = np.loadtxt("/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Adaptive_script/Porto/Onboard/Config/path_initial_survey.txt", delimiter = ", ")
# opa = np.loadtxt(path_operational_area, delimiter = ", ")
#
# plt.figure()
# plt.plot(opa[:, 1], opa[:, 0], 'k.')
# plt.plot(path_presurvey[:, 1], path_presurvey[:, 0], 'r-')
# plt.show()




# fig = go.Figure(data=[go.Scatter3d(x=path_presurvey[:, 1], y=path_presurvey[:, 0],
#                                    z=-path_presurvey[:, 2],
#                                    marker=dict(size=12, color="black"), line=dict(color='darkblue', width=2), )])
# fig.update_layout(
#     scene={
#         'aspectmode': 'manual',
#         'xaxis_title': 'Lon [deg]',
#         'yaxis_title': 'Lat [deg]',
#         'zaxis_title': 'Depth [m]',
#         'aspectratio': dict(x=1, y=1, z=.5),
#     },
#     showlegend=True,
#     title="Initial survey path visualisation",
# )
# fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
# figpath = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Porto/Setup/Pre_survey/fig/"
# plotly.offline.plot(fig, filename=figpath + "Path.html", auto_open=True)
#
# from gmplot import GoogleMapPlotter
# from matplotlib.colors import Normalize
# from matplotlib.cm import ScalarMappable
#
# def color_scatter(gmap, lats, lngs, values=None, colormap='coolwarm',
#                   size=None, marker=False, s=None, **kwargs):
#     def rgb2hex(rgb):
#         """ Convert RGBA or RGB to #RRGGBB """
#         rgb = list(rgb[0:3])  # remove alpha if present
#         rgb = [int(c * 255) for c in rgb]
#         hexcolor = '#%02x%02x%02x' % tuple(rgb)
#         return hexcolor
#
#     if values is None:
#         colors = [None for _ in lats]
#     else:
#         cmap = plt.get_cmap(colormap)
#         norm = Normalize(vmin=min(values), vmax=max(values))
#         scalar_map = ScalarMappable(norm=norm, cmap=cmap)
#         colors = [rgb2hex(scalar_map.to_rgba(value)) for value in values]
#     for lat, lon, c in zip(lats, lngs, colors):
#         gmap.scatter(lats=[lat], lngs=[lon], c=c, size=size, marker=marker, s=s, **kwargs)
#
#
# initial_zoom = 12
# apikey = 'AIzaSyAZ_VZXoJULTFQ9KSPg1ClzHEFjyPbJUro'
# gmap = GoogleMapPlotter(path_presurvey[0, 0], path_presurvey[0, 1], initial_zoom, apikey=apikey)
# color_scatter(gmap, path_presurvey[:, 0], path_presurvey[:, 1], np.zeros_like(path_presurvey[:, 0]), size=20, colormap='hsv')
# gmap.draw("/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/MapPlot/map.html")

