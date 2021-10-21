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
from datetime import datetime
import time
import os
import h5py
import pandas as pd
from matplotlib.gridspec import GridSpec
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly
from pathlib import Path


class DataHandler_Delft3D:
    data_path = None
    data = None
    wind_path = None
    wind_data = None
    figpath = None
    ROUGH = True
    voiceControl = True

    def __init__(self, datapath, windpath, rough = True, voiceControl = True):
        self.ROUGH = rough
        self.voiceControl = voiceControl
        if self.ROUGH:
            if self.voiceControl:
                os.system("say Rough mode is activated")
            print("Rough mode is activated!")
        else:
            if self.voiceControl:
                os.system("say Fine mode is activated")
            print("Fine mode is activated!")
        self.data_path = datapath
        self.wind_path = windpath
        self.load_data()
        self.load_wind()
        self.merge_data()

    def set_figpath(self, figpath):
        self.figpath = figpath
        print("figpath is set correctly, ", self.figpath)

    def load_data(self):
        t1 = time.time()
        self.data = h5py.File(self.data_path, 'r')
        self.lat = np.array(self.data.get("lat"))
        self.lon = np.array(self.data.get("lon"))
        self.depth = np.array(self.data.get("depth"))
        self.salinity = np.array(self.data.get("salinity"))
        self.timestamp_data = np.array(self.data.get("timestamp"))
        self.string_date = datetime.fromtimestamp(self.timestamp_data[0]).strftime("%Y-%m")
        t2 = time.time()
        print("Time consumed: ", t2 - t1, " seconds")
        if self.voiceControl:
            os.system("say loading data correctly, it takes {:.1f} seconds".format(t2 - t1))
        print("Lat: ", self.lat.shape)
        print("Lon: ", self.lon.shape)
        print("Depth: ", self.depth.shape)
        print("Salinity: ", self.salinity.shape)
        print("Date: ", self.string_date)
        print("Time: ", self.timestamp_data.shape)

    def load_wind(self):
        t1 = time.time()
        if "wind_times_serie_porto_obs_2015_2020.txt" in self.wind_path:
            print("Wind data: wind_times_serie_porto_obs_2015_2020.txt")
            self.wind_data = np.array(pd.read_csv(self.wind_path, sep="\t", engine='python'))
            self.wind_data = self.wind_data[:-3, :5]
            yr_wind = self.wind_data[:, 0]
            hr_wind = self.wind_data[:, 1]
            self.timestamp_wind = []
            for i in range(len(yr_wind)):
                year = int(yr_wind[i][6:])
                month = int(yr_wind[i][3:5])
                day = int(yr_wind[i][:2])
                hour = int(hr_wind[i][:2])
                self.timestamp_wind.append(datetime(year, month, day, hour).timestamp())
            self.wind_speed = self.wind_data[:, 3]
            self.wind_maxspeed = self.wind_data[:, 4]
            self.wind_angle = self.wind_data[:, 2]
        else:
            print("Wind data: ", self.wind_path[-13:])
            self.wind_data = np.loadtxt(self.wind_path, delimiter=",")
            self.timestamp_wind = self.wind_data[:, 0]
            self.wind_speed = self.wind_data[:, 1]
            self.wind_angle = self.wind_data[:, -1]
        print("Wind data is loaded correctly!")
        print("wind_data: ", self.wind_data.shape)
        print("wind speed: ", self.wind_speed.shape)
        print("wind angle: ", self.wind_angle.shape)
        print("wind timestamp: ", self.timestamp_wind.shape)
        t2 = time.time()
        print("Time consumed: ", t2 - t1, " seconds")
        if self.voiceControl:
            os.system("say Importing wind data takes {:.1f} seconds".format(t2 - t1))

    def windangle2direction(self, wind_angle):
        angles = np.arange(8) * 45 + 22.5
        self.directions = ['NorthEast', 'East', 'SouthEast', 'South',
                           'SouthWest', 'West', 'NorthWest', 'North']
        id = len(angles[angles < wind_angle]) - 1
        return self.directions[id]

    def windspeed2level(self, wind_speed):
        speeds = np.array([0, 2.5, 10])
        self.levels = ['Mild', 'Moderate', 'Heavy']
        id = len(speeds[speeds < wind_speed]) - 1
        return self.levels[id]

    def windangle2directionRough(self, wind_angle):
        angles = np.arange(4) * 90 + 45
        self.directions = ['East', 'South', 'West', 'North']
        id = len(angles[angles < wind_angle]) - 1
        return self.directions[id]

    def windspeed2levelRough(self, wind_speed):
        speeds = np.array([0, 2.5, 6])
        self.levels = ['Mild', 'Moderate', 'Heavy']
        id = len(speeds[speeds < wind_speed]) - 1
        return self.levels[id]

    def deg2rad(self, deg):
        return deg / 180 * np.pi

    def rad2deg(self, rad):
        return rad / np.pi * 180

    def angle2angle(self, nautical_angle):
        '''
        convert nautical angle to plot angle
        '''
        return self.deg2rad(270 - nautical_angle)

    def wind2uv(self, wind_speed, wind_angle): # convert wind to uv coord
        wind_angle = self.angle2angle(wind_angle)
        u = wind_speed * np.cos(wind_angle)
        v = wind_speed * np.sin(wind_angle)
        return u, v

    def uv2wind(self, u, v): # convert uv coord to wind again
        wind_speed = np.sqrt(u ** 2 + v ** 2)
        wind_angle = self.angle2angle(self.rad2deg(np.arctan2(v, u))) # v is the speed component in y, u is the speed component in x, cartisian normal
        return wind_speed, wind_angle

    def merge_data(self):
        self.wind_v = []
        self.wind_dir = []
        self.wind_level = []
        for i in range(len(self.timestamp_data)):
            id_wind = (np.abs(self.timestamp_wind - self.timestamp_data[i])).argmin()
            self.wind_v.append(self.wind_speed[id_wind])
            if self.ROUGH:
                self.wind_dir.append(self.windangle2directionRough(self.wind_angle[id_wind])) # here one can choose whether
                self.wind_level.append(self.windspeed2levelRough(self.wind_speed[id_wind])) # to use rough or not
            else:
                self.wind_dir.append(self.windangle2direction(self.wind_angle[id_wind]))
                self.wind_level.append(self.windspeed2level(self.wind_speed[id_wind]))
        print("Data is merged correctly!!")
        print("wind levels: ", len(np.unique(self.wind_level)), np.unique(self.wind_level))
        print("wind directions: ", len(np.unique(self.wind_dir)), np.unique(self.wind_dir))

    def refill_unmatched_data(self, sal_ave):
        fill_row = []
        fill_column = []
        for i in range(sal_ave.shape[1]):
            fill_row.append(np.nan)
        for i in range(self.lat.shape[0]):
            fill_column.append(np.nan)
        fill_column = np.array(fill_column).reshape([-1, 1])
        fill_row = np.array(fill_row).reshape([1, -1])
        sal_ave = np.concatenate((sal_ave, fill_row), axis=0)
        sal_ave = np.concatenate((sal_ave, fill_column), axis=1)
        return sal_ave

    def filterNaN(self, val):
        ncol = val.shape[1]
        temp = np.empty((0, ncol))
        for i in range(val.shape[0]):
            indicator = 0
            for j in range(val.shape[1]):
                if not np.isnan(val[i, j]):
                    indicator = indicator + 1
                if indicator == ncol:
                    temp = np.append(temp, val[i, :].reshape(1, -1), axis=0)
                else:
                    pass
        return temp

    def plot_grouppeddata(self):
        fig = plt.figure(figsize=(len(self.directions) * 10, len(self.levels) * 10))
        gs = GridSpec(ncols=len(self.directions), nrows=len(self.levels), figure=fig)
        counter = 0
        for i in range(len(self.levels)):
            idx = np.where(np.array(self.wind_level) == self.levels[i])[0]
            if len(self.salinity.shape) == 4:
                sal_temp = self.salinity[idx, :, :, 0] # only extract surface data
            else:
                sal_temp = self.salinity[idx, :, :]
            for j in range(len(self.directions)):
                idy = np.where(np.array(self.wind_dir)[idx] == self.directions[j])[0]
                ax = fig.add_subplot(gs[i, j])
                if len(idy):
                    sal_total = sal_temp[idy, :, :]
                    sal_ave = np.mean(sal_total, axis=0)
                    if sal_ave.shape[0] != self.lon.shape[0]:
                        sal_ave = self.refill_unmatched_data(sal_ave)
                    im = ax.scatter(self.lon[:, :, 0], self.lat[:, :, 0], c=sal_ave, cmap = "Paired")
                    plt.colorbar(im)
                else:
                    ax.scatter(self.lon[:, :, 0], self.lat[:, :, 0], c='w')
                ax.set_xlabel('Lon [deg]')
                ax.set_ylabel('Lat [deg]')
                ax.set_title(self.levels[i] + " " + self.directions[j])
                counter = counter + 1
                print(counter)
        if self.ROUGH:
            plt.savefig(self.figpath + "WindCondition/WindCondition_" + self.string_date + "_Rough.png")
        else:
            plt.savefig(self.figpath + "WindCondition/WindCondition_" + self.string_date + ".png")
        print(self.figpath + "WindCondition/WindCondition_" + self.string_date + ".png")
        plt.close("all")
        if self.voiceControl:
            os.system("say Finished plotting the groupped data")

    def plot_surface_timeseries(self):
        if self.voiceControl:
            os.system("say Now it plots timeseries")
        vmin = 0
        vmax = 35
        Lon = self.lon[:, :, 0].reshape(-1, 1)
        Lat = self.lat[:, :, 0].reshape(-1, 1)
        for i in range(self.salinity.shape[0]):
            if len(self.salinity.shape) == 4:
                S = self.salinity[i, :, :, 0].reshape(-1, 1)
            else:
                S = self.salinity[i, :, :].reshape(-1, 1)
            dataset = self.filterNaN(np.hstack((Lon, Lat, S)))
            fig = plt.figure(figsize=(20, 10))
            gs = GridSpec(ncols=2, nrows=1, figure=fig)
            ax = fig.add_subplot(gs[0])
            im = ax.scatter(dataset[:, 0], dataset[:, 1], c=dataset[:, 2], cmap="Paired", vmin=vmin, vmax=vmax)
            ax.set_title("Salinity on " + datetime.fromtimestamp(self.timestamp_data[i]).strftime("%d / %m / %Y - %H:%M"))
            ax.set_xlabel("Lon [deg]")
            ax.set_ylabel("Lat [deg]")
            plt.colorbar(im)

            id_wind = (np.abs(self.timestamp_wind - self.timestamp_data[i])).argmin()
            ax = fig.add_subplot(gs[1])
            cir = plt.Circle((0, 0), 2.5, color='r', fill=False)
            plt.gca().add_patch(cir)
            ws = self.wind_speed[id_wind]
            wd = self.wind_angle[id_wind]
            u = ws * np.cos(self.angle2angle(wd))
            v = ws * np.sin(self.angle2angle(wd))
            plt.quiver(0, 0, u, v, scale=20)
            ax.set_title("Wind on " + datetime.fromtimestamp(self.timestamp_wind[id_wind]).strftime("%d / %m / %Y - %H:%M"))
            plt.xlim(-10, 10)
            plt.ylim(-10, 10)
            ax.set_aspect("equal", adjustable="box")
            plt.savefig(self.figpath + "TimeSeries/D_{:04d}.png".format(i))
            plt.close("all")
        if self.voiceControl:
            os.system("say Finished plotting time series")

    def plotscatter3D(self, layers, frame = -1, camera = dict(x=-1.25, y=-1.25, z=1.25)):
        import plotly.express as px
        if self.voiceControl:
            os.system("say I am plotting the scatter data on the 3D grid now")
        Lon = self.lon[:, :, :layers].reshape(-1, 1)
        Lat = self.lat[:, :, :layers].reshape(-1, 1)
        Depth = self.depth[0, :, :, :layers].reshape(-1, 1)
        if frame == -1:
            sal_val = np.mean(self.salinity[:, :, :, :layers], axis=0).reshape(-1, 1)
        else:
            sal_val = self.salinity[frame, :, :, :layers].reshape(-1, 1)
        print(sal_val.shape)
        # Make 3D plot # #
        fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scene'}]])
        fig.add_trace(
            go.Scatter3d(
                x=Lon.squeeze(), y=Lat.squeeze(), z=Depth.squeeze(),
                mode='markers',
                marker=dict(
                    size=4,
                    color=sal_val.squeeze(),
                    colorscale = px.colors.qualitative.Light24, # to have quantitified colorbars and colorscales
                    showscale=True
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
            title="Delft 3D data visualisation on " + self.string_date,
            scene_camera_eye=camera,
        )
        if frame == -1:
            plotly.offline.plot(fig, filename=self.figpath + "Scatter3D/Data_" + self.string_date + "_ave.html", auto_open=False)
        else:
            plotly.offline.plot(fig, filename=self.figpath + "Scatter3D/Data_" + self.string_date + ".html", auto_open=False)
        if self.voiceControl:
            os.system("say Finished plotting 3D")
        # fig.write_image(self.figpath + "Scatter3D/S_{:04}.png".format(frame), width=1980, height=1080, engine = "orca")

    def plot_grid_on_data(self, grid):
        Lon = self.lon[:, :, 0].reshape(-1, 1)
        Lat = self.lat[:, :, 0].reshape(-1, 1)
        Depth = self.depth[0, :, :, 0].reshape(-1, 1)
        sal_val = np.mean(self.salinity[:, :, :, 0], axis = 0).reshape(-1, 1)

        lat_grid = grid.grid_coord[:, 0]
        lon_grid = grid.grid_coord[:, 1]
        depth_grid = np.ones_like(lat_grid) * 1
        fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scene'}]])
        fig.add_trace(
            go.Scatter3d(
                x=Lon.squeeze(), y=Lat.squeeze(), z=Depth.squeeze(),
                mode='markers',
                marker=dict(
                    size=4,
                    color=sal_val.squeeze(),
                    colorscale="RdBu",
                    showscale=False
                ),
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter3d(
                x=lon_grid.squeeze(), y=lat_grid.squeeze(), z=depth_grid.squeeze(),
                mode='markers',
                marker=dict(
                    size=2,
                    showscale=False
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
            showlegend=False,
            scene_camera_eye=dict(x=-1.25, y=-1.25, z=1.25),
        )
        plotly.offline.plot(fig, filename=self.figpath + "Grid/Data" + ".html",
                            auto_open=False)


if __name__ == "__main__":
    data_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Delft3D/Delft3D.h5"
    wind_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Porto/Wind/wind_data.txt"
    datahandler = DataHandler_Delft3D(data_path, wind_path, rough = True)
    datahandler.set_figpath("/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Porto/Delft3D/fig/")
    # datahandler.plot_grouppeddata()
    # datahandler.plot_grid_on_data(Grid())
    datahandler.plotscatter3D(layers=1, frame = -1)
    datahandler.plot3Danimation()
    datahandler.plot_surface_timeseries() # it has problems, needs to be fixed

