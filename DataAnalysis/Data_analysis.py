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
    # path_data = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Adaptive_script/Porto/Onboard/Data/"
    path_data = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Porto/Simulation/Data/"
    path_global = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Adaptive_script/Porto/Onboard/"
    figpath = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Porto/Simulation/fig/"

    def __init__(self):
        print("hello;-)")
        # self.load_path_designed()

        self.load_config()
        self.load_prior_extracted()
        # self.load_prior_corrected()
        # self.plot_on_map()
        # self.load_pre_survey()
        self.load_adaptive_mission()
        self.update_the_field()
        self.plot_data()
        # self.plot_data_in3D(self.waypoint_adaptive, self.salinity_adaptive, "test")

        # self.plot_data_in3D(self.waypoint_adaptive, self.salinity_adaptive)
        # self.plot_data_in3D(self.waypoint_adaptive, self.salinity_adaptive)

    def load_config(self):
        path_config = self.path_global + "Config/"
        self.grid = np.loadtxt(path_config + "grid.txt", delimiter=", ")
        self.beta = np.loadtxt(path_config + "beta.txt", delimiter=", ")
        # self.counter_waypoint = np.loadtxt(path_config + "counter_waypoint_Mission.txt", delimiter=", ")
        self.OpArea = np.loadtxt(path_config + "OperationArea.txt", delimiter=", ")
        self.polygon_c = np.loadtxt(path_config + "polygon_centre.txt", delimiter=", ")
        self.polygon = np.loadtxt(path_config + "polygon.txt", delimiter=", ")
        self.threshold = np.loadtxt(path_config + "threshold.txt", delimiter=", ")
        self.path_designed = np.loadtxt(self.path_global + "Config/path_initial_survey.txt", delimiter=", ")
        # self.mu_cond = np.loadtxt(path_config + "mu_cond.txt", delimiter=", ")
        # self.Sigma_cond = np.loadtxt(path_config + "Sigma_cond.txt", delimiter=", ")
        print("finished with config")

    def load_prior_extracted(self):
        self.prior_extracted = np.loadtxt(self.path_data + "Prior_extracted.txt", delimiter=", ")
        print(self.prior_extracted.shape)

    def load_prior_corrected(self):
        self.prior_corrected = np.loadtxt(self.path_data + "Prior_corrected.txt", delimiter=", ")
        print(self.prior_corrected.shape)

    def load_adaptive_mission(self):
        datapath_adaptive = self.path_data
        files = os.listdir(datapath_adaptive)
        # for file in files:
        #     if file == ".DS_Store":
        #         pass
        #     else:
        #         print(file)
        # file = files[-3]
        # print(file)
        # datapath_adaptive = datapath_adaptive + file + "/"
        path_salinity = datapath_adaptive + "data_salinity.txt"
        path_timestamp = datapath_adaptive + "data_timestamp.txt"
        path_waypoint = datapath_adaptive + "data_path.txt"

        self.salinity_adaptive= np.loadtxt(path_salinity, delimiter=", ")
        self.waypoint_adaptive = np.loadtxt(path_waypoint, delimiter=", ")
        self.timestamp_adaptive = np.loadtxt(path_timestamp, delimiter=", ")
        ind_selected = np.where(self.salinity_adaptive >= 20)[0]
        self.salinity_adaptive = self.salinity_adaptive[ind_selected]
        self.waypoint_adaptive = self.waypoint_adaptive[ind_selected, :]
        self.waypoint_adaptive[:, 2] = self.myround(self.waypoint_adaptive[:, 2] + .5) - .5

        self.timestamp_adaptive = self.timestamp_adaptive[ind_selected]
        print("finsihed")

    def myround(self, value, base = .75):
        return base * np.round(value / base)

    def update_the_field(self):
        print("salinity: ", self.salinity_adaptive.shape)
        self.lat_auv = self.waypoint_adaptive[:, 0].reshape(-1, 1)
        self.lon_auv = self.waypoint_adaptive[:, 1].reshape(-1, 1)
        self.depth_auv = self.waypoint_adaptive[:, 2].reshape(-1, 1)
        self.salinity_auv = self.salinity_adaptive.reshape(-1, 1)
        self.x_auv, self.y_auv = latlon2xy(self.lat_auv, self.lon_auv, self.lat_auv[0], self.lon_auv[0])

        self.lat_prior = self.prior_extracted[:, 0].reshape(-1, 1)
        self.lon_prior = self.prior_extracted[:, 1].reshape(-1, 1)
        self.depth_prior = self.prior_extracted[:, 2].reshape(-1, 1)
        self.salinity_prior = self.prior_extracted[:, 3].reshape(-1, 1)
        self.x_prior, self.y_prior = latlon2xy(self.lat_prior, self.lon_prior, self.lat_auv[0], self.lon_auv[0])

        obs = np.hstack((self.x_auv, self.y_auv, self.depth_auv))
        range_lateral = 550
        range_vertical = 2
        ksi = range_lateral / range_vertical
        sigma = np.sqrt(.5)
        eta = 4.5 / range_lateral
        tau = np.sqrt(.04)
        H_obs = compute_H(obs, obs, ksi)
        Sigma_obs = Matern_cov(sigma, eta, H_obs) + tau ** 2 * np.identity(H_obs.shape[0])

        sal_est = []
        for i in range(len(self.salinity_auv)):
            ind = self.getPriorIndAtLoc([self.lat_auv[i], self.lon_auv[i], self.depth_auv[i]])
            sal_est.append(self.salinity_prior[ind])
        sal_est = np.array(sal_est).reshape(-1, 1)

        grid = np.hstack((self.x_prior, self.y_prior, self.depth_prior))
        H_grid = compute_H(grid, grid, ksi)
        Sigma_grid = Matern_cov(sigma, eta, H_grid)

        H_grid_obs = compute_H(grid, obs, ksi)
        Sigma_grid_obs = Matern_cov(sigma, eta, H_grid_obs)

        self.mu_cond = self.salinity_prior + Sigma_grid_obs @ np.linalg.solve(Sigma_obs, (self.salinity_auv - sal_est))
        self.Sigma_cond = Sigma_grid - Sigma_grid_obs @ np.linalg.solve(Sigma_obs, Sigma_grid_obs.T)
        self.perr = np.diag(self.Sigma_cond).reshape(-1, 1)

        self.ep = EP_1D(self.mu_cond, self.Sigma_cond, self.threshold)
        self.ep_prior = EP_1D(self.salinity_prior, Sigma_grid, self.threshold)

        pass

    def getPriorIndAtLoc(self, loc):
        '''
        return the index in the prior data which corresponds to the location
        '''
        lat, lon, depth = loc
        distDepth = self.depth_prior - depth
        distLat = self.lat_prior - lat
        distLon = self.lon_prior - lon
        dist = np.sqrt(distLat ** 2 + distLon ** 2 + distDepth ** 2)
        ind_loc = np.where(dist == np.nanmin(dist))[0][0]
        return ind_loc

    def load_pre_survey(self):
        # datapath_pre_survey = self.path_data + "Pre_survey/"
        datapath_pre_survey = self.path_data
        # files = os.listdir(datapath_pre_survey)
        # file = files[-1]
        datapath_pre_survey = datapath_pre_survey
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
        self.plot_data_in3D(self.waypoint_presurvey, self.salinity_presurvey, "test")


    def plot_on_map(self):

        from gmplot import GoogleMapPlotter
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable

        def color_scatter(gmap, lats, lngs, values=None, colormap='coolwarm',
                          size=None, marker=False, s=None, **kwargs):
            def rgb2hex(rgb):
                """ Convert RGBA or RGB to #RRGGBB """
                rgb = list(rgb[0:3])  # remove alpha if present
                rgb = [int(c * 255) for c in rgb]
                hexcolor = '#%02x%02x%02x' % tuple(rgb)
                return hexcolor

            if values is None:
                colors = [None for _ in lats]
            else:
                cmap = plt.get_cmap(colormap)
                norm = Normalize(vmin=min(values), vmax=max(values))
                scalar_map = ScalarMappable(norm=norm, cmap=cmap)
                colors = [rgb2hex(scalar_map.to_rgba(value)) for value in values]
            for lat, lon, c in zip(lats, lngs, colors):
                gmap.scatter(lats=[lat], lngs=[lon], c=c, size=size, marker=marker, s=s, **kwargs)

        initial_zoom = 12
        apikey = 'AIzaSyAZ_VZXoJULTFQ9KSPg1ClzHEFjyPbJUro'
        gmap = GoogleMapPlotter(self.grid[0, 0], self.grid[0, 1], initial_zoom, apikey=apikey)
        color_scatter(gmap, self.grid[:, 0], self.grid[:, 1], np.zeros_like(self.grid[:, 0]), size=20,
                      colormap='hsv')
        color_scatter(gmap, self.polygon[:, 0], self.polygon[:, 1], np.zeros_like(self.polygon[:, 0]), size=20,
                      colormap='hsv')
        color_scatter(gmap, self.path_designed[:, 0], self.path_designed[:, 1], np.zeros_like(self.path_designed[:, 0]), size=20,
                      colormap='hsv')
        gmap.polygon(self.OpArea[:, 0], self.OpArea[:, 1])
        # color_scatter(gmap, self.OpArea[:, 0], self.OpArea[:, 1], np.zeros_like(self.OpArea[:, 0]), size=20,
        #               colormap='hsv')
        gmap.draw("/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/MapPlot/map.html")

    def plot_data(self):
        fig = plt.figure(figsize=(25, 10))
        gs = GridSpec(ncols=3, nrows=1, figure=fig)
        for i in range(len(np.unique(self.depth_prior))):
            ind_auv = (self.waypoint_adaptive[:, 2] == )
            ind = (self.depth_prior == np.unique(self.depth_prior)[i])
            ax = fig.add_subplot(gs[i])
            im = ax.scatter(self.lon_prior[ind], self.lat_prior[ind], s=300, c=self.ep[ind], cmap="RdBu", vmin=0,
                            vmax=1)
            plt.colorbar(im, fraction=0.08, pad=0.04)
            ax.set_xlabel("Lon [deg]")
            ax.set_ylabel("Lat [deg]")
            ax.set_title("Depth: " + str(np.unique(self.depth_prior)[i]))
        plt.show()


    def plot_data_in3D(self, waypoint, salinity, file_string):
        import plotly.express as px
        lat = waypoint[:, 0]
        lon = waypoint[:, 1]
        depth = waypoint[:, 2]
        sal = salinity

        # Make 3D plot # #
        fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scene'}]])
        grid_lat, grid_lon = latlon2xy(self.lat_prior, self.lon_prior, 0, 0)
        grid_lat = self.lat_prior
        grid_lon = self.lon_prior
        grid_depth = self.depth_prior
        sal = self.mu_cond
        # sal = self.ep_prior

        fig.add_trace(
            go.Volume(
                x=grid_lon.flatten(), y=grid_lat.flatten(), z=grid_depth.flatten(),
                value=sal.flatten(),
                # isomin=isomin,
                # isomax=isomax,
                # opacity=opacity,
                # surface_count=surface_count,
                # colorbar=colorbar,
            ),
            row=1, col=1
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
    # t = a.salinity_adaptive
    # t1 = a.timestamp_adaptive
    # t2 = a.waypoint_adaptive
#%%

def plot_data(self):
    fig = plt.figure(figsize=(25, 10))
    gs = GridSpec(ncols=3, nrows=1, figure=fig)
    for i in range(len(np.unique(self.depth_prior))):
        ind = (self.depth_prior == np.unique(self.depth_prior)[i])
        ax = fig.add_subplot(gs[i])
        im = ax.scatter(self.lon_prior[ind], self.lat_prior[ind], s = 300, c = self.ep[ind], cmap = "RdBu", vmin = 0, vmax = 1)
        plt.colorbar(im, fraction=0.08, pad=0.04)
        ax.set_xlabel("Lon [deg]")
        ax.set_ylabel("Lat [deg]")
        ax.set_title("Depth: " + str(np.unique(self.depth_prior)[i]))
    plt.show()

plot_data(a)

