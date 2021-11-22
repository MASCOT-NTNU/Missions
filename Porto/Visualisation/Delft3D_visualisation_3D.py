import mat73
import time
import matplotlib.pyplot as plt
import os
from datetime import datetime

import pandas as pd

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'font.style': 'oblique'})
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly
from scipy.ndimage import gaussian_filter

# server = False
IDUN = False
server = True
if server:
    from usr_func import *
else:
    from usr_func import *


class Delft3D:

    server = False
    lat_river_mouth, lon_river_mouth = 41.139024, -8.680089

    def __init__(self, server = False):
        print("hello ")
        self.server = server
        self.which_path()
        self.load_tide()
        self.load_wind()
        self.load_water_discharge()
        self.ini_wind_classifier()

        self.make_png()

    def which_path(self):

        if self.server:
            print("Server mode is activated")
            if IDUN:
                self.path = "/cluster/work/yaoling/mascot/delft3d_data_plot/data/"
                self.path_wind = "/cluster/work/yaoling/mascot/delft3d_data_plot/data/wind_data.txt"
                self.path_tide = "/cluster/work/yaoling/mascot/delft3d_data_plot/data/tide.txt"
                self.path_water_discharge = "/cluster/work/yaoling/mascot/delft3d_data_plot/data/data_water_discharge.txt"
                self.figpath = "/cluster/work/yaoling/mascot/delft3d_data_plot/fig/"
            else:
                self.path = "/home/ahomea/y/yaoling/MASCOT/Porto_Data_Processing/Data/"
                self.path_wind = "/home/ahomea/y/yaoling/MASCOT/Porto_Data_Processing/Data/wind_data.txt"
                self.path_tide = "/home/ahomea/y/yaoling/MASCOT/Porto_Data_Processing/Data/tide.txt"
                self.path_water_discharge = "/home/ahomea/y/yaoling/MASCOT/Porto_Data_Processing/Data/data_water_discharge.txt"
                self.figpath = "/home/ahomea/y/yaoling/MASCOT/Porto_Data_Processing/fig/"
        else:
            print("Local mode is activated")
            self.path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Prior/Nov_Prior/"
            self.path_wind = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Wind/wind_data.txt"
            self.path_tide = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Tide/tide.txt"
            self.path_water_discharge = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/WaterDischarge/data_water_discharge.txt"
            self.figpath = None

    def load_wind(self):
        self.wind = np.loadtxt(self.path_wind, delimiter=",")
        print("Wind is loaded successfully!")

    def load_tide(self):
        self.tide = np.loadtxt(self.path_tide, delimiter=", ")
        print("Tide is loaded successfully!")

    def load_water_discharge(self):
        self.water_discharge = np.loadtxt(self.path_water_discharge, delimiter=", ")
        print("Water discharge is loaded successfully!")

    def ini_wind_classifier(self):
        self.angles = np.arange(4) * 90 + 45
        self.directions = ['East', 'South', 'West', 'North']
        self.speeds = np.array([0, 2.5, 6])
        self.levels = ['Mild', 'Moderate', 'Heavy']
        print("Wind classifier is initialised successfully!")

    def load_mat_data(self, file):
        t1 = time.time()
        self.data = mat73.loadmat(self.path + file)
        self.data = self.data["data"]
        self.lon = self.data["X"]
        self.lat = self.data["Y"]
        self.depth = self.data["Z"]
        self.Time = self.data['Time']
        self.timestamp_data = (self.Time - 719529) * 24 * 3600  # 719529 is how many days have passed from Jan1 0,
        # to Jan1 1970. Since 1970Jan1, is used as the starting index for datetime
        self.sal_data = self.data["Val"]
        self.string_date = datetime.fromtimestamp(self.timestamp_data[0]).strftime("%Y_%m")
        t2 = time.time()
        print("loading takes: ", t2 - t1, " seconds")

    def get_rotational_matrix(self, alpha):
        R = np.array([[np.cos(deg2rad(alpha)), np.sin(deg2rad(alpha))],
                      [-np.sin(deg2rad(alpha)), np.cos(deg2rad(alpha))]])
        return R

    def get_value_at_loc(self, loc, k_neighbour):
        lat_loc, lon_loc, depth_loc = loc
        x_dist, y_dist = latlon2xy(self.lat_f, self.lon_f, lat_loc, lon_loc)
        depth_dist = self.depth_f - depth_loc
        dist = np.sqrt(x_dist ** 2 + y_dist ** 2 + depth_dist ** 2)  # cannot use lat lon since deg/rad will mix metrics
        ind_neighbour = np.argsort(dist)[:k_neighbour]  # use nearest k neighbours to compute the average
        value = np.nanmean(self.salinity_f[ind_neighbour])
        return value

    def gaussian_smoothing_on_usr_defined_grid(self):
        lat_lc, lon_lc = 41.045, -8.819  # left bottom corner coordinate
        nlat = 20 # number of points along lat direction
        nlon = 25 # number of points along lon direction
        ndepth = 4 # number of layers in the depth direction
        max_distance_lat = 18000 # distance of the box along lat
        max_distance_lon = 14000 # distance of the box along lon
        max_depth = -5 # max depth
        alpha = 12
        k_n = 1 # k neighbours will be used for averaging
        Rm = self.get_rotational_matrix(alpha) # rotational matrix used to find the new grid

        X = np.linspace(0, max_distance_lat, nlat)
        Y = np.linspace(0, max_distance_lon, nlon)
        depth_domain = np.linspace(0, max_depth, ndepth)

        self.values_3d = np.zeros([nlat, nlon, ndepth])
        self.grid_3d = []

        for i in range(nlat):
            print("nlat: ", i)
            for j in range(nlon):
                tmp = Rm @ np.array([X[i], Y[j]])
                xnew, ynew = tmp
                lat_loc, lon_loc = xy2latlon(xnew, ynew, lat_lc, lon_lc)
                for k in range(ndepth):
                    self.values_3d[i, j, k] = self.get_value_at_loc([lat_loc, lon_loc, depth_domain[k]], k_n)
                    self.grid_3d.append([X[i], Y[j], depth_domain[k]])
        self.grid_3d = np.array(self.grid_3d)
        self.values_gussian_filtered = gaussian_filter(self.values_3d, 1)

    def make_png(self):
        counter = 0
        files = os.listdir(self.path)
        files = sorted(files)
        for file in files:
            # if file.endswith("9_sal_3.mat"):
            if file.endswith(".mat"):
                self.load_mat_data(file)

                for i in range(self.sal_data.shape[0]):
                # for i in [0]:
                    # Here comes the flattened version of the frame at the specific time
                    self.lat_f = self.lat.flatten()
                    self.lon_f = self.lon.flatten()
                    self.depth_f = self.depth[i, :, :, :].flatten()
                    self.salinity_f = self.sal_data[i, :, :, :].flatten()
                    self.gaussian_smoothing_on_usr_defined_grid()

                    dist_wind = np.abs(self.timestamp_data[i] - self.wind[:, 0])
                    ind_wind = np.where(dist_wind == np.nanmin(dist_wind))[0][0]

                    dist_tide = np.abs(self.timestamp_data[i] - self.tide[:, 0])
                    ind_tide = np.where(dist_tide == np.nanmin(dist_tide))[0][0]

                    dist_water_discharge = np.abs(self.timestamp_data[i] - self.water_discharge[:, 0])
                    ind_water_discharge = np.where(dist_water_discharge == np.nanmin(dist_water_discharge))[0][0]

                    id_wind_dir = len(self.angles[self.angles < self.wind[ind_wind, 2]]) - 1
                    id_wind_level = len(self.speeds[self.speeds < self.wind[ind_wind, 1]]) - 1
                    wind_dir = self.directions[id_wind_dir]
                    wind_level = self.levels[id_wind_level]

                    u, v = s2uv(self.wind[ind_wind, 1], self.wind[ind_wind, 2])

                    fig = make_subplots(rows=3, cols=3,
                                        specs=[[{"type": "scene", "colspan": 2, "rowspan": 3}, None, {"type": "scene"}],
                                               [None, None, {"type": "scene", "rowspan": 2}],
                                               [None, None, None]],
                                        subplot_titles=("Salinity field under " + wind_dir + " " + wind_level + " wind",
                                                        "Tide condition", "Water discharge [m3/s]"))

                    # The regular grid needs to be used for plotting, image it is a square but the data is extracted at the certain locations
                    fig.add_trace(go.Volume(
                        x=self.grid_3d[:, 1].flatten(),
                        y=self.grid_3d[:, 0].flatten(),
                        z=self.grid_3d[:, 2].flatten(),
                        value=self.values_gussian_filtered.flatten(),
                        isomin=24,
                        isomax=33.3,
                        opacity=.1,
                        surface_count=30,
                        coloraxis="coloraxis",
                        caps=dict(x_show=False, y_show=False, z_show=False),
                    ),
                        row=1, col=1)

                    # add wind direction
                    lat_lc, lon_lc = 41.045, -8.819  # left bottom corner coordinate
                    loc_x, loc_y = latlon2xy(self.lat_river_mouth, self.lon_river_mouth, lat_lc, lon_lc)

                    fig.add_trace(go.Cone(
                        x=[loc_y],
                        y=[loc_x],
                        z=[5],
                        u=[u],
                        v=[v],
                        w=[0],
                        showscale=False,
                        sizemode="absolute",
                        sizeref=self.wind[ind_wind, 1] * 500),
                        row=1, col=1)

                    if self.tide[ind_tide, 2] == 1:
                        if self.timestamp_data[i] >= self.tide[ind_tide, 0]:
                            fig.add_trace(
                                go.Cone(x=[0], y=[0], z=[0], u=[-1], v=[0], w=[0], colorscale='Blues', showscale=False,
                                        sizemode="absolute",
                                        sizeref=1000),
                                row=1, col=3)
                            tide_string = "Ebb (River to Ocean)"
                        else:
                            fig.add_trace(
                                go.Cone(x=[0], y=[0], z=[0], u=[1], v=[0], w=[0], colorscale='Blues', showscale=False,
                                        sizemode="absolute",
                                        sizeref=1000),
                                row=1, col=3)
                            tide_string = "Flood (Ocean to River)"
                    else:
                        if self.timestamp_data[i] >= self.tide[ind_tide, 0]:
                            fig.add_trace(
                                go.Cone(x=[0], y=[0], z=[0], u=[1], v=[0], w=[0], colorscale='Blues', showscale=False,
                                        sizemode="absolute",
                                        sizeref=1000),
                                row=1, col=3)
                            tide_string = "Flood (Ocean to River)"
                            fig.add_trace(row=1, col=3)
                        else:
                            fig.add_trace(
                                go.Cone(x=[0], y=[0], z=[0], u=[-1], v=[0], w=[0], colorscale='Blues', showscale=False,
                                        sizemode="absolute",
                                        sizeref=1000),
                                row=1, col=3)
                            tide_string = "Ebb (River to Ocean)"

                    if self.water_discharge[ind_water_discharge, 1] <= 400:
                        fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0],
                                                   mode="markers+text",
                                                   marker=dict(size=self.water_discharge[ind_water_discharge, 1] / 10,
                                                               color="green"),
                                                   text=str(self.water_discharge[ind_water_discharge, 1]),
                                                   textfont=dict(color="green", size=20),
                                                   textposition="top right"),
                                      row=2, col=3)
                    elif self.water_discharge[ind_water_discharge, 1] <= 800:
                        fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0],
                                                   mode="markers+text",
                                                   marker=dict(size=self.water_discharge[ind_water_discharge, 1] / 10,
                                                               color="orange"),
                                                   text=str(self.water_discharge[ind_water_discharge, 1]),
                                                   textfont=dict(color="orange", size=20),
                                                   textposition="top right"),
                                      row=2, col=3)
                    else:
                        fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0],
                                                   mode="markers+text",
                                                   marker=dict(size=self.water_discharge[ind_water_discharge, 1] / 10,
                                                               color="red"),
                                                   text=str(self.water_discharge[ind_water_discharge, 1]),
                                                   textfont=dict(color="red", size=20),
                                                   textposition="top right"),
                                      row=2, col=3)

                    camera1 = dict(
                        up=dict(x=0, y=0, z=1),
                        center=dict(x=0, y=0, z=0),
                        eye=dict(x=1, y=-1, z=1)
                    )
                    fig.update_coloraxes(colorscale="rainbow")
                    fig.update_layout(
                        scene=dict(
                            zaxis=dict(nticks=4, range=[-5, 6], ),
                            xaxis_title='Distance along Lon direction [m]',
                            yaxis_title='Distance along Lat direction [m]',
                            zaxis_title='Depth [m]',
                        ),
                        scene2=dict(
                            annotations=[
                                dict(showarrow=False, x=0, y=0, z=500, text=tide_string,
                                     font=dict(family="Times New Roman", size=16, color="#000000"), )]
                        ),
                        scene3=dict(xaxis=dict(range=[-10000, 10000]),
                                    yaxis=dict(range=[-10000, 10000]),
                                    zaxis=dict(range=[-10000, 10000]), ),

                        scene_camera=camera1,
                        title=datetime.fromtimestamp(self.timestamp_data[i]).strftime("%Y%m%d - %H:%M"),
                        scene_aspectmode='manual',
                        scene_aspectratio=dict(x=1, y=1, z=.4),
                        coloraxis_colorbar_x=-0.05,
                        scene2_camera_eye=dict(x=0, y=-1.25, z=0),
                        scene3_camera_eye=dict(x=0, y=-1.25, z=0)
                    )
                    fig.update_scenes(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False)
                    if not server:
                        plotly.offline.plot(fig,
                                        filename="/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Porto/Delft3D/fig/Sal_delft3d.html",
                                        auto_open=True)
                    else:
                        fig.write_image(self.figpath + "I_{:05d}.png".format(counter), width=1980, height=1080)
                    counter = counter + 1
                    print(counter)

if __name__ == "__main__":
    a = Delft3D(server = server)


