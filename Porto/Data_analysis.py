import numpy as np
from datetime import datetime
import time
import mat73
import os
import h5py
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly
from pathlib import Path

plotly.io.orca.config.executable = '/usr/local/bin/orca'
plotly.io.orca.config.save()
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'font.style': 'oblique'})



class Mat2HDF5:
    data_path = None
    data_path_new = None

    def __init__(self, data_path, data_path_new):
        # data_path contains the path for the mat file
        # data_path_new contains the path for the new hdf5 file
        self.data_path = data_path
        self.data_path_new = data_path_new
        # self.loaddata()

    def loaddata(self):
        '''
        This loads the original data
        '''
        os.system("say it will take more than 100 seconds to import data")
        t1 = time.time()
        self.data = mat73.loadmat(self.data_path)
        data = self.data["data"]
        self.lon = data["X"]
        self.lat = data["Y"]
        self.depth = data["Z"]
        self.Time = data['Time']
        self.timestamp_data = (self.Time - 719529) * 24 * 3600  # 719529 is how many days have passed from Jan1 0,
        # to Jan1 1970. Since 1970Jan1, is used as the starting index for datetime
        self.sal_data = data["Val"]
        self.string_date = datetime.fromtimestamp(self.timestamp_data[0]).strftime("%Y_%m")
        t2 = time.time()
        print("Data is loaded correctly!")
        print("Lat: ", self.lat.shape)
        print("Lon: ", self.lon.shape)
        print("Depth: ", self.depth.shape)
        print("salinity: ", self.sal_data.shape)
        print("Date: ", self.string_date)
        print("Time consumed: ", t2 - t1, " seconds.")
        os.system("say Congrats, it takes only {:.1f} seconds to import data.".format(t2 - t1))

    def mat2hdf(self):
        t1 = time.time()
        data_hdf = h5py.File(self.data_path_new, 'w')
        print("Finished: file creation")
        data_hdf.create_dataset("lon", data=self.lon)
        print("Finished: lon dataset creation")
        data_hdf.create_dataset("lat", data=self.lat)
        print("Finished: lat dataset creation")
        data_hdf.create_dataset("timestamp", data=self.timestamp_data)
        print("Finished: timestamp dataset creation")
        data_hdf.create_dataset("depth", data=self.depth)
        print("Finished: depth dataset creation")
        data_hdf.create_dataset("salinity", data=self.sal_data)
        print("Finished: salinity dataset creation")
        t2 = time.time()
        print("Time consumed: ", t2 - t1, " seconds.")
        os.system("say finished data conversion, it takes {:.1f} seconds.".format(t2 - t1))

# class DataMerger(Mat2HDF5): # only used for converting all mat to hdf5
#     data_folder = None
#     data_folder_new = None
#
#     def __init__(self, data_folder, data_folder_new):
#         self.data_folder = data_folder
#         self.data_folder_new = data_folder_new
#         # self.mergeAll()
#
#     def mergeAll(self):
#         for s in os.listdir(self.data_folder):
#             if s.endswith(".mat"):
#                 print(s)
#                 self.data_path = self.data_folder + s
#                 t = Mat2HDF5(self.data_path, self.data_path[:-4] + ".h5")
#                 t.loaddata()
#                 t.mat2hdf()

# if __name__ == "__main__":
#     data_folder = '/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Sep_Prior/'
#     data_folder = "/Volumes/LaCie/MASCOT/Data/"
#     data_folder_new = '/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Sep_Prior/New'
#     a = DataMerger(data_folder, data_folder_new)
# data_path = '/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/D2_3D_salinity-021.mat'
# data_path_new = '/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Delft3D/Delft3D.h5'
# a = Mat2HDF5(data_path, data_path_new)
# a.mat2hdf()

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


# if __name__ == "__main__":
#     data_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Delft3D/Delft3D.h5"
#     wind_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Porto/Wind/wind_data.txt"
#     datahandler = DataHandler_Delft3D(data_path, wind_path, rough = True)
#     datahandler.set_figpath("/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Porto/Delft3D/fig/")
#     # datahandler.plot_grouppeddata()
#     # datahandler.plot_grid_on_data(Grid())
#     datahandler.plotscatter3D(layers=1, frame = -1)
    # datahandler.plot3Danimation()
    # datahandler.plot_surface_timeseries() # it has problems, needs to be fixed


class MergeTide:
    tide_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Porto/Tide/Data/"

    def __init__(self):
        print("hellor")

    def getdata(self):
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

    def select_data(self):
        self.ind_selected = self.month == 9
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

    def extractEbby(self):
        self.ebby_start = []
        self.ebby_end = []
        self.ind_ebby_start = []
        self.ind_ebby_end = []
        self.tide = []
        for i in range(len(self.tide_type_numerical_selected)):
            if self.tide_type_numerical_selected[i] == 1:
                if i < len(self.tide_type_numerical_selected) - 1:
                    if self.tide_type_numerical_selected[i + 1] == 0:
                        self.ebby_start.append(self.tide_timestamp[i, 0])
                        self.ebby_end.append(self.tide_timestamp[i + 1, 0])
                        self.ind_ebby_start.append(i)
                        self.ind_ebby_end.append(i + 1)
                        self.tide.append(self.tide_type_numerical_selected[i])
                        self.tide.append(self.tide_type_numerical_selected[i + 1])
                    else:
                        pass
                else:
                    pass
        self.ebby_start = np.array(self.ebby_start).reshape(-1, 1)
        self.ebby_end = np.array(self.ebby_end).reshape(-1, 1)
        self.ebby = np.hstack((self.ebby_start, self.ebby_end))
        np.savetxt(self.tide_path[:-5] + "ebby.txt", self.ebby, delimiter = ", ")

class DataGetter(Mat2HDF5, DataHandler_Delft3D):
    '''
    Get data according to date specified and wind direction
    '''
    data_folder = None
    data_folder_new = None
    wind_path = None
    ebby_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Porto/Tide/ebby.txt"

    def __init__(self, data_folder, data_folder_new, wind_path):
        # GridPoly.__init__(self, debug = False)
        self.depth_layers = 5 # top five layers will be chosen for saving the data, to make sure it has enough space to operate
        self.data_folder = data_folder
        self.data_folder_new = data_folder_new
        self.wind_path = wind_path
        # self.load_ebby()

        # self.getAllPriorData()
        # self.mergedata("/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Prior/Sep_Prior/salinity_2019_SEP_2.h5")
        # self.merge_all()
        self.checkMerged()
        pass

    def load_ebby(self):
        print("Loading ebby...")
        self.ebby = np.loadtxt(self.ebby_path, delimiter=", ")
        print("Ebby is loaded correctly!")

    def mergedata(self, data_path): #load merged data with Delft3D and wind data
        t1 = time.time()
        datahandler = DataHandler_Delft3D(data_path, self.wind_path, rough = True, voiceControl = False)
        self.file_string = data_path[-13:-3]
        print("File string: ", self.file_string)
        self.lat = datahandler.lat
        self.lon = datahandler.lon
        self.depth = datahandler.depth
        self.salinity = datahandler.salinity
        self.wind_dir = datahandler.wind_dir
        self.wind_level = datahandler.wind_level
        self.timestamp_data = datahandler.timestamp_data
        print("lat: ", datahandler.lat.shape)
        print("lon: ", datahandler.lon.shape)
        print("depth: ", datahandler.depth.shape)
        print("salinity", datahandler.salinity.shape)
        print("timestamp_data: ", self.timestamp_data.shape)
        print("wind_dir: ", np.array(self.wind_dir).shape, len(self.wind_dir))
        print("wind_level: ", np.array(self.wind_level).shape, len(self.wind_level))
        t2 = time.time()
        print(t2 - t1)

    def isEbby(self, timestamp):
        if len(np.where(timestamp < self.ebby[:, 0])[0]) > 0:
            ind = np.where(timestamp < self.ebby[:, 0])[0][0] - 1 # check the index for ebby start
            if timestamp < self.ebby[ind, 1]:
                return True
            else:
                return False
        else:
            return False

    def getdata4wind(self, wind_dir, wind_level, data_path):

        self.mergedata(data_path)
        self.lat_merged = self.lat[:, :, :self.depth_layers] # only extract top layers
        self.lon_merged = self.lon[:, :, :self.depth_layers]

        self.ind_selected = np.where((np.array(self.wind_dir) == wind_dir) & (np.array(self.wind_level) == wind_level))[0] # indices for selecting the time frames

        if np.any(self.ind_selected):
            self.test_timestamp = self.timestamp_data[self.ind_selected]
            self.ind_selected_ebby = []
            print("before ebby checking: ", len(self.ind_selected))
            print("len of test timestamp: ", len(self.test_timestamp))

            for i in range(len(self.test_timestamp)):
                if self.isEbby(self.test_timestamp[i]):
                    self.ind_selected_ebby.append(self.ind_selected[i])

            print("after ebby checking: ", len(self.ind_selected_ebby))
            if len(self.ind_selected_ebby) > 0:
                print("Found ", wind_dir, wind_level, " {:d} timeframes are used to average".format(len(self.ind_selected_ebby)))
                self.salinity_merged = np.mean(self.salinity[self.ind_selected_ebby, :, :, :self.depth_layers], axis = 0)
                self.depth_merged = np.mean(self.depth[self.ind_selected_ebby, :, :, :self.depth_layers], axis = 0)
                t1 = time.time()
                if os.path.exists(self.data_folder_new + "Merged_" + wind_dir + "_" + wind_level + "_" + self.file_string + ".h5"):
                    print("rm -rf " + self.data_folder_new + "Merged_" + wind_dir + "_" + wind_level + "_" + self.file_string + ".h5")
                    os.system("rm -rf " + self.data_folder_new + "Merged_" + wind_dir + "_" + wind_level + "_" + self.file_string + ".h5")
                data_file = h5py.File(
                    self.data_folder_new + "Merged_" + wind_dir + "_" + wind_level + "_" + self.file_string + ".h5", 'w')
                data_file.create_dataset("lat", data=self.lat_merged)
                data_file.create_dataset("lon", data=self.lon_merged)
                data_file.create_dataset("depth", data=self.depth_merged)
                data_file.create_dataset("salinity", data=self.salinity_merged)
                t2 = time.time()
                print("Finished data creation! Time consumed: ", t2 - t1)
        else:
            print("Not enough data, no corresponding ", wind_dir, wind_level, "data is found.")

    def getAllPriorData(self):
        # wind_dirs = ['North', 'South', 'West', 'East'] # get wind_data for all conditions
        wind_dirs = ['West', 'East'] # get wind_data for all conditions
        wind_levels = ['Mild', 'Moderate', 'Heavy'] # get data for all conditions
        # wind_dirs = ['North'] # get wind_data for all conditions
        # wind_levels = ['Moderate'] # get data for all conditions
        counter = 0
        os.system("say Start merging all the data")
        for wind_dir in wind_dirs:
            for wind_level in wind_levels:
                print("wind_dir: ", wind_dir)
                print("wind_level: ", wind_level)
                for file in os.listdir(self.data_folder):
                    if file.endswith(".h5"):
                        print(self.data_folder + file)
                        # if counter == 3:
                        #     break
                        t1 = time.time()
                        self.getdata4wind(wind_dir, wind_level, self.data_folder + file)
                        t2 = time.time()
                        os.system("say Step {:d} of {:d}, time consumed {:.1f} seconds".format(counter, len(os.listdir(self.data_folder)) * len(wind_dirs) * len(wind_levels), t2 - t1))
                        print("Step {:d} of {:d}, time consumed {:.1f} seconds".format(counter, len(os.listdir(self.data_folder)) * len(wind_dirs) * len(wind_levels), t2 - t1))
                        counter = counter + 1

    def merge_all(self):
        new_data_path = self.data_folder + "Merged_all/"
        wind_dirs = ['North', 'South', 'West', 'East'] # get wind_data for all conditions
        wind_levels = ['Mild', 'Moderate', 'Heavy'] # get data for all conditions
        # wind_dirs = ['North'] # get wind_data for all conditions
        # wind_levels = ['Moderate'] # get data for all conditions
        counter = 0
        for wind_dir in wind_dirs:
            for wind_level in wind_levels:
                print("wind_dir: ", wind_dir)
                print("wind_level: ", wind_level)
                self.lat_merged_all = 0
                self.lon_merged_all = 0
                self.depth_merged_all = 0
                self.salinity_merged_all = 0
                exist = False
                counter_mean = 0
                for file in os.listdir(self.data_folder_new):
                    if "Merged_" + wind_dir + "_" + wind_level in file:
                        print(file)
                        print(counter)
                        counter = counter + 1
                        counter_mean = counter_mean + 1
                        exist = True
                        data = h5py.File(self.data_folder_new + file, 'r')
                        self.lat_merged_all = self.lat_merged_all + np.array(data.get("lat"))
                        self.lon_merged_all = self.lon_merged_all + np.array(data.get("lon"))
                        self.depth_merged_all = self.depth_merged_all + np.array(data.get("depth"))
                        self.salinity_merged_all = self.salinity_merged_all + np.array(data.get("salinity"))
                # print("lat_merged_all: ", self.lat_merged_all.shape)
                # print("lon_merged_all: ", self.lon_merged_all.shape)
                # print("depth_merged_all:, ", self.depth_merged_all.shape)
                # print("salinity_merged_all: ", self.salinity_merged_all.shape)
                if exist:
                    print("counter_mean: ", counter_mean)
                    self.lat_mean = self.lat_merged_all / counter_mean
                    self.lon_mean = self.lon_merged_all / counter_mean
                    self.depth_mean = self.depth_merged_all / counter_mean
                    self.salinity_mean = self.salinity_merged_all / counter_mean
                    data_file = h5py.File(self.data_folder + "Merged_all/" + wind_dir + "_" + wind_level + "_all" + ".h5", 'w')
                    data_file.create_dataset("lat", data=self.lat_mean)
                    data_file.create_dataset("lon", data=self.lon_mean)
                    data_file.create_dataset("depth", data=self.depth_mean)
                    data_file.create_dataset("salinity", data=self.salinity_mean)
                else:
                    print("No data found for " + wind_dir + wind_level)
                    pass

    def checkMerged(self):
        files = os.listdir(self.data_folder + "Merged_all/")
        wind_dirs = ['North', 'South', 'West', 'East']  # get wind_data for all conditions
        wind_levels = ['Mild', 'Moderate', 'Heavy']  # get data for all conditions
        plt.figure(figsize = (20, 30))
        counter_plot = 0
        for wind_dir in wind_dirs:
            for wind_level in wind_levels:
                for i in range(len(files)):
                    if wind_dir + "_" + wind_level in files[i]:
                        break
                plt.subplot(4, 3, counter_plot + 1)
                self.data_test = h5py.File(self.data_folder + "Merged_all/" + files[i], 'r')
                self.lat_test = np.array(self.data_test.get("lat"))
                self.lon_test = np.array(self.data_test.get("lon"))
                self.depth_test = np.array(self.data_test.get("depth"))
                self.salinity_test = np.array(self.data_test.get("salinity"))
                # print("lat_test: ", self.lat_test)
                # print("lon_test: ", self.lon_test)
                # print("depth_test: ", self.depth_test)
                # print("salinity_test: ", self.salinity_test)
                plt.scatter(self.lon_test[:, :, 0], self.lat_test[:, :, 0], c = self.salinity_test[:, :, 0], cmap = "Paired")
                plt.xlabel('Lon [deg]')
                plt.ylabel("Lat [deg]")
                plt.title(files[i])
                plt.colorbar()
                counter_plot = counter_plot + 1
        figpath = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Prior/Sep_Prior/fig/"
        plt.savefig(figpath + "prior.pdf")
        plt.show()
        # for wind_dir in wind_dirs:
        #     for wind_level in wind_levels:
        #         for folder in folders:
        #             folder_content = os.listdir(folder)


class MaretecDataHandler:
    data_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Prior/Maretec/Exemplo_Douro/2021-09-22_2021-09-23/WaterProperties.hdf5"
    delft_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Delft3D/Surface/D2_201609_surface_salinity.h5"
    figpath = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Porto/Setup/Polygon/fig/"
    path_onboard = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Adaptive_script/Porto/Onboard/"
    path_laptop = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Adaptive_script/Porto/Laptop/"
    figpath_comp = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Porto/Setup/MareTec_Delft_Comp/" + delft_path[-29:-20] + "/"
    circumference = 40075000 # [m], circumference of the earth

    def __init__(self):
        self.lat_origin, self.lon_origin = 41.061874, -8.650977  # origin location
        self.loaddata()
        self.loadDelft3DSurface()
        self.loadSatellite()
        # self.getsimilarDelft()
        self.plotSurfaceData()
        pass

    def loaddata(self):
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
        print("Data is loaded correctly, time consumed: ", t2 - t1)

    def checkFolder(self):
        i = 0
        while os.path.exists(self.figpath + "P%s" % i):
            i += 1
        self.figpath = self.figpath + "P%s" % i
        if not os.path.exists(self.figpath):
            print(self.figpath + " is created")
            os.mkdir(self.figpath)
        else:
            print(self.figpath + " is already existed")

    def visualiseData(self):
        self.checkFolder()
        print("Here it comes the plotting for the updated results from Maretec.")
        files = os.listdir(self.data_path[:81])
        files.sort()
        counter = 0
        for i in range(len(files)):
            if files[i] != ".DS_Store":
                datapath = self.data_path[:81] + files[i] + "/WaterProperties.hdf5"
                self.data_path = datapath
                self.loaddata()
                for j in range(self.salinity.shape[0]):
                    print(j)
                    plt.figure(figsize=(10, 10))
                    plt.scatter(self.lon[:self.lon.shape[1], :], self.lat[:self.lon.shape[1], :],
                                c=self.salinity[j, :self.lon.shape[1], :], vmin=26, vmax=36, cmap="Paired")
                    plt.colorbar()
                    plt.title("Surface salinity estimation from Maretec at time: {:02d}:00 during ".format(
                        j) + self.data_path[81:102])
                    plt.xlabel("Lon [deg]")
                    plt.ylabel("Lat [deg]")
                    plt.savefig(self.figpath + "/P_{:04d}.png".format(counter))
                    plt.close("all")
                    counter = counter + 1
                # plt.show()

    @staticmethod
    def deg2rad(deg):
        return deg / 180 * np.pi

class DataGetter(Mat2HDF5, DataHandler_Delft3D):
    '''
    Get data according to date specified and wind direction
    '''
    data_folder = None
    data_folder_new = None
    wind_path = None
    ebby_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Porto/Tide/ebby.txt"

    def __init__(self, data_folder, data_folder_new, wind_path):
        # GridPoly.__init__(self, debug = False)
        self.depth_layers = 5 # top five layers will be chosen for saving the data, to make sure it has enough space to operate
        self.data_folder = data_folder
        self.data_folder_new = data_folder_new
        self.wind_path = wind_path
        # self.load_ebby()

        # self.getAllPriorData()
        # self.mergedata("/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Prior/Sep_Prior/salinity_2019_SEP_2.h5")
        # self.merge_all()
        self.checkMerged()
        pass

    def load_ebby(self):
        print("Loading ebby...")
        self.ebby = np.loadtxt(self.ebby_path, delimiter=", ")
        print("Ebby is loaded correctly!")

    def mergedata(self, data_path): #load merged data with Delft3D and wind data
        t1 = time.time()
        datahandler = DataHandler_Delft3D(data_path, self.wind_path, rough = True, voiceControl = False)
        self.file_string = data_path[-13:-3]
        print("File string: ", self.file_string)
        self.lat = datahandler.lat
        self.lon = datahandler.lon
        self.depth = datahandler.depth
        self.salinity = datahandler.salinity
        self.wind_dir = datahandler.wind_dir
        self.wind_level = datahandler.wind_level
        self.timestamp_data = datahandler.timestamp_data
        print("lat: ", datahandler.lat.shape)
        print("lon: ", datahandler.lon.shape)
        print("depth: ", datahandler.depth.shape)
        print("salinity", datahandler.salinity.shape)
        print("timestamp_data: ", self.timestamp_data.shape)
        print("wind_dir: ", np.array(self.wind_dir).shape, len(self.wind_dir))
        print("wind_level: ", np.array(self.wind_level).shape, len(self.wind_level))
        t2 = time.time()
        print(t2 - t1)

    def isEbby(self, timestamp):
        if len(np.where(timestamp < self.ebby[:, 0])[0]) > 0:
            ind = np.where(timestamp < self.ebby[:, 0])[0][0] - 1 # check the index for ebby start
            if timestamp < self.ebby[ind, 1]:
                return True
            else:
                return False
        else:
            return False

    def getdata4wind(self, wind_dir, wind_level, data_path):

        self.mergedata(data_path)
        self.lat_merged = self.lat[:, :, :self.depth_layers] # only extract top layers
        self.lon_merged = self.lon[:, :, :self.depth_layers]

        self.ind_selected = np.where((np.array(self.wind_dir) == wind_dir) & (np.array(self.wind_level) == wind_level))[0] # indices for selecting the time frames

        if np.any(self.ind_selected):
            self.test_timestamp = self.timestamp_data[self.ind_selected]
            self.ind_selected_ebby = []
            print("before ebby checking: ", len(self.ind_selected))
            print("len of test timestamp: ", len(self.test_timestamp))

            for i in range(len(self.test_timestamp)):
                if self.isEbby(self.test_timestamp[i]):
                    self.ind_selected_ebby.append(self.ind_selected[i])

            print("after ebby checking: ", len(self.ind_selected_ebby))
            if len(self.ind_selected_ebby) > 0:
                print("Found ", wind_dir, wind_level, " {:d} timeframes are used to average".format(len(self.ind_selected_ebby)))
                self.salinity_merged = np.mean(self.salinity[self.ind_selected_ebby, :, :, :self.depth_layers], axis = 0)
                self.depth_merged = np.mean(self.depth[self.ind_selected_ebby, :, :, :self.depth_layers], axis = 0)
                t1 = time.time()
                if os.path.exists(self.data_folder_new + "Merged_" + wind_dir + "_" + wind_level + "_" + self.file_string + ".h5"):
                    print("rm -rf " + self.data_folder_new + "Merged_" + wind_dir + "_" + wind_level + "_" + self.file_string + ".h5")
                    os.system("rm -rf " + self.data_folder_new + "Merged_" + wind_dir + "_" + wind_level + "_" + self.file_string + ".h5")
                data_file = h5py.File(
                    self.data_folder_new + "Merged_" + wind_dir + "_" + wind_level + "_" + self.file_string + ".h5", 'w')
                data_file.create_dataset("lat", data=self.lat_merged)
                data_file.create_dataset("lon", data=self.lon_merged)
                data_file.create_dataset("depth", data=self.depth_merged)
                data_file.create_dataset("salinity", data=self.salinity_merged)
                t2 = time.time()
                print("Finished data creation! Time consumed: ", t2 - t1)
        else:
            print("Not enough data, no corresponding ", wind_dir, wind_level, "data is found.")

    def getAllPriorData(self):
        # wind_dirs = ['North', 'South', 'West', 'East'] # get wind_data for all conditions
        wind_dirs = ['West', 'East'] # get wind_data for all conditions
        wind_levels = ['Mild', 'Moderate', 'Heavy'] # get data for all conditions
        # wind_dirs = ['North'] # get wind_data for all conditions
        # wind_levels = ['Moderate'] # get data for all conditions
        counter = 0
        os.system("say Start merging all the data")
        for wind_dir in wind_dirs:
            for wind_level in wind_levels:
                print("wind_dir: ", wind_dir)
                print("wind_level: ", wind_level)
                for file in os.listdir(self.data_folder):
                    if file.endswith(".h5"):
                        print(self.data_folder + file)
                        # if counter == 3:
                        #     break
                        t1 = time.time()
                        self.getdata4wind(wind_dir, wind_level, self.data_folder + file)
                        t2 = time.time()
                        os.system("say Step {:d} of {:d}, time consumed {:.1f} seconds".format(counter, len(os.listdir(self.data_folder)) * len(wind_dirs) * len(wind_levels), t2 - t1))
                        print("Step {:d} of {:d}, time consumed {:.1f} seconds".format(counter, len(os.listdir(self.data_folder)) * len(wind_dirs) * len(wind_levels), t2 - t1))
                        counter = counter + 1

    def merge_all(self):
        new_data_path = self.data_folder + "Merged_all/"
        wind_dirs = ['North', 'South', 'West', 'East'] # get wind_data for all conditions
        wind_levels = ['Mild', 'Moderate', 'Heavy'] # get data for all conditions
        # wind_dirs = ['North'] # get wind_data for all conditions
        # wind_levels = ['Moderate'] # get data for all conditions
        counter = 0
        for wind_dir in wind_dirs:
            for wind_level in wind_levels:
                print("wind_dir: ", wind_dir)
                print("wind_level: ", wind_level)
                self.lat_merged_all = 0
                self.lon_merged_all = 0
                self.depth_merged_all = 0
                self.salinity_merged_all = 0
                exist = False
                counter_mean = 0
                for file in os.listdir(self.data_folder_new):
                    if "Merged_" + wind_dir + "_" + wind_level in file:
                        print(file)
                        print(counter)
                        counter = counter + 1
                        counter_mean = counter_mean + 1
                        exist = True
                        data = h5py.File(self.data_folder_new + file, 'r')
                        self.lat_merged_all = self.lat_merged_all + np.array(data.get("lat"))
                        self.lon_merged_all = self.lon_merged_all + np.array(data.get("lon"))
                        self.depth_merged_all = self.depth_merged_all + np.array(data.get("depth"))
                        self.salinity_merged_all = self.salinity_merged_all + np.array(data.get("salinity"))
                # print("lat_merged_all: ", self.lat_merged_all.shape)
                # print("lon_merged_all: ", self.lon_merged_all.shape)
                # print("depth_merged_all:, ", self.depth_merged_all.shape)
                # print("salinity_merged_all: ", self.salinity_merged_all.shape)
                if exist:
                    print("counter_mean: ", counter_mean)
                    self.lat_mean = self.lat_merged_all / counter_mean
                    self.lon_mean = self.lon_merged_all / counter_mean
                    self.depth_mean = self.depth_merged_all / counter_mean
                    self.salinity_mean = self.salinity_merged_all / counter_mean
                    data_file = h5py.File(self.data_folder + "Merged_all/" + wind_dir + "_" + wind_level + "_all" + ".h5", 'w')
                    data_file.create_dataset("lat", data=self.lat_mean)
                    data_file.create_dataset("lon", data=self.lon_mean)
                    data_file.create_dataset("depth", data=self.depth_mean)
                    data_file.create_dataset("salinity", data=self.salinity_mean)
                else:
                    print("No data found for " + wind_dir + wind_level)
                    pass

    @staticmethod
    def rad2deg(rad):
        return rad / np.pi * 180

    @staticmethod
    def latlon2xy(lat, lon, lat_origin, lon_origin):
        x = MaretecDataHandler.deg2rad((lat - lat_origin)) / 2 / np.pi * MaretecDataHandler.circumference
        y = MaretecDataHandler.deg2rad((lon - lon_origin)) / 2 / np.pi * MaretecDataHandler.circumference * np.cos(MaretecDataHandler.deg2rad(lat))
        return x, y

    def getPolygonArea(self):
        area = 0
        prev = self.polygon[-1]
        for i in range(self.polygon.shape[0] - 1):
            now = self.polygon[i]
            xnow, ynow = MaretecDataHandler.latlon2xy(now[0], now[1], self.lat_origin, self.lon_origin)
            xpre, ypre = MaretecDataHandler.latlon2xy(prev[0], prev[1], self.lat_origin, self.lon_origin)
            area += xnow * ypre - ynow * xpre
            prev = now
        self.PolyArea = area / 2 / 1e6
        return self.PolyArea
        # print("Area: ", self.PolyArea / 1e6, " km2")

    def loadSatellite(self):
        print("Satellite data is loading")
        self.data_satellite = np.loadtxt(self.path_laptop + "satellite_data.txt", delimiter=", ")
        self.lat_satellite = self.data_satellite[:, 0]
        self.lon_satellite = self.data_satellite[:, 1]
        self.ref_satellite = self.data_satellite[:, -1]
        print("Satellite data is loaded successfully...")

    def loadDelft3DSurface(self):
        delft3d = h5py.File(self.delft_path, 'r')
        self.lat_delft3d = np.mean(np.array(delft3d.get("lat"))[:-1, :-1, :], axis = 2)
        self.lon_delft3d = np.mean(np.array(delft3d.get("lon"))[:-1, :-1, :], axis = 2)
        self.salinity_delft3d = np.array(delft3d.get("salinity"))
        print("lat_delft3d: ", self.lat_delft3d.shape)
        print("lon_delft3d: ", self.lon_delft3d.shape)
        print("salinity_delft3d: ", self.salinity_delft3d.shape)

    def plotSurfaceData(self):
        i = 0
        while os.path.exists(self.figpath_comp + "P%s" % i):
            i += 1
        self.figpath_comp = self.figpath_comp + "P%s" % i
        if not os.path.exists(self.figpath_comp):
            print(self.figpath_comp + " is created")
            path = Path(self.figpath_comp)
            path.mkdir(parents=True, exist_ok=True)
        else:
            print(self.figpath_comp + " is already existed")

        for i in range(self.salinity_delft3d.shape[0]):
            print(i)
            plt.figure(figsize = (10, 10))
            plt.scatter(self.lon_delft3d, self.lat_delft3d, c = self.salinity_delft3d[i, :, :], cmap = "Paired", vmin = 33, vmax = 36)
            plt.colorbar()
            plt.scatter(self.lon, self.lat, c = np.mean(self.salinity, axis = 0), cmap = "Paired", vmin = 33, vmax = 36, alpha = .05)
            plt.scatter(self.lon_satellite, self.lat_satellite, c = self.ref_satellite, cmap = "Paired", vmin = 20, vmax = 60, alpha = .03)
            plt.xlabel("Lon [deg]")
            plt.ylabel("Lat [deg]")
            plt.title("Frame: " + str(i))
            plt.savefig(self.figpath_comp + "/P_{:04d}.png".format(i))
            plt.close("all")
            # break

    # def getsimilarDelft(self):
    #     self.mse = []
    #     t1 = time.time()
    #     for k in range(self.salinity_delft3d.shape[0]):
    #         self.lat_comp = []
    #         self.lon_comp = []
    #         self.sal_delft = []
    #         self.sal_maretec = []
    #         for i in range(self.lat.shape[0]):
    #             for j in range(self.lat.shape[1]):
    #                 if self.lat[i, j] >= 41.1 and self.lat[i, j] <= 41.16 and self.lon[i, j] <= -8.65:
    #                     row_ind, col_ind = self.getDelftAtLoc([self.lat[i, j], self.lon[i, j]])
    #                     self.lat_comp.append(self.lat[i, j])
    #                     self.lon_comp.append(self.lon[i, j])
    #                     self.sal_delft.append(self.salinity_delft3d[k, row_ind, col_ind])
    #                     self.sal_maretec.append(np.mean(self.salinity[:, i, j], axis = 0))
    #         self.sal_delft = np.array(self.sal_delft)
    #         self.MSE = np.nansum(np.sqrt((self.sal_delft - self.sal_maretec) ** 2))
    #         self.mse.append(self.MSE)
    #         t2 = time.time()
    #         print("Time consumed: ", t2 - t1)
    #         print("sal_delft: ", self.sal_delft.shape)
    #         print("MSE: ", self.mse)
    #         # plt.scatter(self.lon_comp, self.lat_comp, c = self.sal_maretec, cmap = "Paired", vmin = 33, vmax = 36)
    #         # plt.scatter(self.lon_comp, self.lat_comp, c = self.sal_delft, alpha = .3, cmap = "Paired", vmin = 33, vmax = 36)
    #         # plt.colorbar()
    #         # plt.show()
    #     ind_min =

    def getDelftAtLoc(self, loc):
        lat, lon = loc
        distLat = self.lat_delft3d - lat
        distLon = self.lon_delft3d - lon
        dist = np.sqrt(distLat ** 2 + distLon ** 2)
        row_ind = np.where(dist == np.nanmin(dist))[0]
        col_ind = np.where(dist == np.nanmin(dist))[1]
        return row_ind, col_ind

    def loadSatellite(self):
        print("Satellite data is loading")
        self.data_satellite = np.loadtxt(self.path_laptop + "satellite_data.txt", delimiter=", ")
        self.lat_satellite = self.data_satellite[:, 0]
        self.lon_satellite = self.data_satellite[:, 1]
        self.ref_satellite = self.data_satellite[:, -1]
        print("lat_latellite: ", self.lat_satellite.shape)
        print("lon_satellite: ", self.lon_satellite.shape)
        print("ref_satellite: ", self.ref_satellite.shape)
        print("Satellite data is loaded successfully...")

    # def plotdataonDay(self, day, hour, wind_dir, wind_level):
    #     print("This will plot the data on day " + day)
    #     datapath = self.data_path[:81] + day + "/WaterProperties.hdf5"
    #     hour_start = int(hour[:2])
    #     hour_end = int(hour[3:])
    #     self.data_path = datapath
    #     self.loaddata()
    #     self.loadDelft3D(wind_dir, wind_level)
    #     plt.figure(figsize=(10, 10))
    #     plt.scatter(self.lon_delft3d, self.lat_delft3d, c=self.salinity_delft3d, vmin=26, vmax=36, alpha=1,
    #                 cmap="Paired")
    #     plt.colorbar()
    #     plt.scatter(self.lon[:self.lon.shape[1], :], self.lat[:self.lon.shape[1], :],
    #                 c=self.salinity[hour_start, :self.lon.shape[1], :], vmin=26, vmax=36, alpha = .25, cmap="Paired")
    #     plt.scatter(self.lon[:self.lon.shape[1], :], self.lat[:self.lon.shape[1], :],
    #                 c=self.salinity[hour_end, :self.lon.shape[1], :], vmin=26, vmax=36, alpha=.05, cmap="Paired")
    #     # plt.scatter(self.lon_satellite, self.lat_satellite, c = self.ref_satellite, alpha = .01, cmap = "Paired")
    #     plt.title("Surface salinity estimation from Maretec during " + self.data_path[81:102])
    #     plt.xlabel("Lon [deg]")
    #     plt.ylabel("Lat [deg]")
    #     polygon = plt.ginput(n = 100, timeout = 0) # wait for the click to select the polygon
    #     plt.show()
    #     self.polygon = []
    #     for i in range(len(polygon)):
    #         self.polygon.append([polygon[i][1], polygon[i][0]])
    #     self.polygon = np.array(self.polygon)
    #     np.savetxt(self.path_onboard + "polygon.txt", self.polygon, delimiter=", ")
    #     print("Ploygon is selected successfully. Total area: ", self.getPolygonArea())
    #     os.system("say Congrats, Polygon is selected successfully, total area is {:.2f} km2".format(self.getPolygonArea()))
    #     print("Total area: ", self.getPolygonArea())
    #     self.save_wind_condition(wind_dir, wind_level)

    def save_wind_condition(self, wind_dir, wind_level):
        f_wind = open(self.path_onboard + "wind_condition.txt", 'w')
        f_wind.write("wind_dir=" + wind_dir + ", wind_level=" + wind_level)
        f_wind.close()
        print("wind_condition is saved successfully!")

data_folder = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Prior/Sep_Prior/"
data_folder_new = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Prior/Sep_Prior/Merged/"
wind_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Porto/Wind/wind_data.txt"
path_maretec = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Prior/Maretec/Exemplo_Douro/2021-09-22_2021-09-23/"

# if __name__ == "__main__":
    # a = MaretecDataHandler()
    # a = DataGetter(data_folder, data_folder_new, wind_path)


class PortoMissionAnalyser:
    path_mission_data = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/20210923/Mission/lauv-xplore-1-backseat/MASCOT_PORTO/Data/"
    timestamp_mission = np.empty(0)
    salinity_mission = np.empty(0)
    lat_mission = np.empty(0)
    lon_mission = np.empty(0)
    depth_mission = np.empty(0)

    timestamp_presurvey = np.empty(0)
    salinity_presurvey = np.empty(0)
    lat_presurvey = np.empty(0)
    lon_presurvey = np.empty(0)
    depth_presurvey = np.empty(0)

    def __init__(self):
        # self.loaddata_mission()
        self.loaddata_presurvey()

    def loaddata_mission(self):
        files = os.listdir(self.path_mission_data)
        files.sort()
        for file in files:
            if file != ".DS_Store":
                if file.startswith("MissionData"):
                    print(file)
                    temp_timestamp = np.loadtxt(self.path_mission_data + file + "/data_timestamp.txt", delimiter = ", ")
                    temp_salinity = np.loadtxt(self.path_mission_data + file + "/data_salinity.txt", delimiter=", ")
                    temp_path = np.loadtxt(self.path_mission_data + file + "/data_path.txt", delimiter=", ")
                    self.timestamp_mission = np.append(self.timestamp_mission, temp_timestamp)
                    self.salinity_mission = np.append(self.salinity_mission, temp_salinity)
                    self.lat_mission = np.append(self.lat_mission, temp_path[:, 0])
                    self.lon_mission = np.append(self.lon_mission, temp_path[:, 1])
                    self.depth_mission = np.append(self.depth_mission, temp_path[:, -1])

    def loaddata_presurvey(self):
        files = os.listdir(self.path_mission_data)
        for file in files:
            if file != ".DS_Store":
                if file.startswith("Pre_survey_data"):
                    temp_timestamp = np.loadtxt(self.path_mission_data + file + "/data_timestamp.txt", delimiter = ", ")
                    temp_salinity = np.loadtxt(self.path_mission_data + file + "/data_salinity.txt", delimiter=", ")
                    temp_path = np.loadtxt(self.path_mission_data + file + "/data_path.txt", delimiter=", ").reshape(-1, 3)
                    self.timestamp_presurvey = np.append(self.timestamp_presurvey, temp_timestamp)
                    self.salinity_presurvey = np.append(self.salinity_presurvey, temp_salinity)
                    # print(file)
                    # print(temp_path.shape)
                    self.lat_presurvey = np.append(self.lat_presurvey, temp_path[:, 0])
                    self.lon_presurvey = np.append(self.lon_presurvey, temp_path[:, 1])
                    self.depth_presurvey = np.append(self.depth_presurvey, temp_path[:, -1])

# if __name__ == "__main__":

# a = PortoMissionAnalyser()
# ind_start = 500
# ind_end = 3000
# # plt.scatter(a.lon_mission, a.lat_mission, c = a.salinity_mission)
# plt.scatter(a.lon_presurvey[ind_start:ind_end], a.lat_presurvey[ind_start:ind_end], c = a.salinity_presurvey[ind_start:ind_end])
# plt.colorbar()
# plt.show()


import matplotlib.path as mplPath # used to determine whether a point is inside the grid or not
class WaypointNode:
    '''
    generate node for each waypoint
    '''
    waypoint_loc = None
    subwaypoint_len = 0
    subwaypoint_loc = []

    def __init__(self, subwaypoints_len, subwaypoints_loc, waypoint_loc):
        self.subwaypoint_len = subwaypoints_len
        self.subwaypoint_loc = subwaypoints_loc
        self.waypoint_loc = waypoint_loc


class GridPoly(WaypointNode):
    '''
    generate the polygon grid with equal-distance from one to another
    '''
    lat_origin, lon_origin = 41.10251, -8.669811  # the right bottom corner coordinates
    circumference = 40075000  # circumference of the earth, [m]
    distance_poly = 90  # [m], distance between two neighbouring points
    depth_obs = [-.5, -1.25, -2]  # [m], distance in depth, depth to be explored
    pointsPr = 10000  # points per layer
    polygon = None
    loc_start = None
    counter_plot = 0  # counter for plot number
    counter_grid = 0  # counter for grid points
    debug = True
    voiceCtrl = False
    figpath = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Porto/Setup/Grid/fig/"
    gridpath = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Adaptive_script/Porto/Onboard/"

    def __init__(self, polygon=np.array([[41.12251, -8.707745],
                                         [41.12413, -8.713079],
                                         [41.11937, -8.715101],
                                         [41.11509, -8.717317],
                                         [41.11028, -8.716535],
                                         [41.10336, -8.716813],
                                         [41.10401, -8.711306],
                                         [41.11198, -8.710787],
                                         [41.11764, -8.710245],
                                         [41.12251, -8.707745]]), debug=False, voiceCtrl=False):
        if debug:
            self.checkFolder()
        self.lat_origin, self.lon_origin = 41.061874, -8.650977  # origin location
        self.grid_poly = []
        self.polygon = polygon
        self.debug = debug
        self.voiceCtrl = voiceCtrl
        self.polygon_path = mplPath.Path(self.polygon)
        self.angle_poly = self.deg2rad(np.arange(0, 6) * 60)  # angles for polygon
        self.getPolygonArea()

        print("Grid polygon is activated!")
        print("Distance between neighbouring points: ", self.distance_poly)
        print("Depth to be observed: ", self.depth_obs)
        print("Starting location: ", self.loc_start)
        print("Polygon: ", self.polygon.shape)
        print("Points desired: ", self.pointsPr)
        print("Debug mode: ", self.debug)
        print("fig path: ", self.figpath)
        t1 = time.time()
        self.getGridPoly()
        # self.plotGridonMap(self.grid_poly)
        # self.savegrid()
        t2 = time.time()
        print("Grid discretisation takes: {:.2f} seconds".format(t2 - t1))

    def checkFolder(self):
        i = 0
        while os.path.exists(self.figpath + "P%s" % i):
            i += 1
        self.figpath = self.figpath + "P%s" % i
        if not os.path.exists(self.figpath):
            print(self.figpath + " is created")
            os.mkdir(self.figpath)
        else:
            print(self.figpath + " is already existed")

    def savegrid(self):
        grid = []
        for i in range(len(self.grid_poly)):
            for j in range(len(self.depth_obs)):
                grid.append([self.grid_poly[i, 0], self.grid_poly[i, 1], self.depth_obs[j]])
        np.savetxt(self.gridpath + "grid.txt", grid, delimiter=", ")
        print("Grid is created correctly, it is saved to grid.txt")

    def revisit(self, loc):
        '''
        func determines whether it revisits the points it already have
        '''
        temp = np.array(self.grid_poly)
        if len(self.grid_poly) > 0:
            dist_min = np.min(np.sqrt((temp[:, 0] - loc[0]) ** 2 + (temp[:, 1] - loc[1]) ** 2))
            ind = np.argmin(np.sqrt((temp[:, 0] - loc[0]) ** 2 + (temp[:, 1] - loc[1]) ** 2))
            if dist_min <= .00001:
                return [True, ind]
            else:
                return [False, []]
        else:
            return [False, []]

    def getNewLocations(self, loc):
        '''
        get new locations around the current location
        '''
        lat_delta, lon_delta = self.xy2latlon(self.distance_poly * np.sin(self.angle_poly),
                                              self.distance_poly * np.cos(self.angle_poly), 0, 0)
        return lat_delta + loc[0], lon_delta + loc[1]

    def getStartLocation(self):
        lat_min = np.amin(self.polygon[:, 0])
        lat_max = np.amax(self.polygon[:, 0])
        lon_min = np.amin(self.polygon[:, 1])
        lon_max = np.amax(self.polygon[:, 1])
        path_polygon = mplPath.Path(self.polygon)
        while True:
            lat_random = np.random.uniform(lat_min, lat_max)
            lon_random = np.random.uniform(lon_min, lon_max)
            if path_polygon.contains_point((lat_random, lon_random)):
                break
        print("The generated random starting location is: ")
        print([lat_random, lon_random])
        self.loc_start = [lat_random, lon_random]

    def getGridPoly(self):
        '''
        get the polygon grid discretisation
        '''
        self.getStartLocation()
        lat_new, lon_new = self.getNewLocations(self.loc_start)
        start_node = []
        for i in range(len(self.angle_poly)):
            if self.polygon_path.contains_point((lat_new[i], lon_new[i])):
                start_node.append([lat_new[i], lon_new[i]])
                self.grid_poly.append([lat_new[i], lon_new[i]])
                self.counter_grid = self.counter_grid + 1

        WaypointNode_start = WaypointNode(len(start_node), start_node, self.loc_start)
        Allwaypoints = self.getAllWaypoints(WaypointNode_start)
        self.grid_poly = np.array(self.grid_poly)
        if len(self.grid_poly) > self.pointsPr:
            print("{:d} waypoints are generated, only {:d} waypoints are selected!".format(len(self.grid_poly),
                                                                                           self.pointsPr))
            self.grid_poly = self.grid_poly[:self.pointsPr, :]
        else:
            print("{:d} waypoints are generated, all are selected!".format(len(self.grid_poly)))
        print("Grid: ", self.grid_poly.shape)

    def getAllWaypoints(self, waypoint_node):
        if self.counter_grid > self.pointsPr:  # stopping criterion to end the recursion
            return WaypointNode(0, [], waypoint_node.waypoint_loc)
        for i in range(waypoint_node.subwaypoint_len):  # loop through all the subnodes
            subsubwaypoint = []
            length_new = 0
            lat_subsubwaypoint, lon_subsubwaypoint = self.getNewLocations(
                waypoint_node.subwaypoint_loc[i])  # generate candidates location
            for j in range(len(self.angle_poly)):
                if self.polygon_path.contains_point((lat_subsubwaypoint[j], lon_subsubwaypoint[j])):
                    testRevisit = self.revisit([lat_subsubwaypoint[j], lon_subsubwaypoint[j]])
                    if not testRevisit[0]:
                        subsubwaypoint.append([lat_subsubwaypoint[j], lon_subsubwaypoint[j]])
                        self.grid_poly.append([lat_subsubwaypoint[j], lon_subsubwaypoint[j]])
                        self.counter_grid = self.counter_grid + 1
                        length_new = length_new + 1
            if len(subsubwaypoint) > 0:
                Subwaypoint = WaypointNode(len(subsubwaypoint), subsubwaypoint, waypoint_node.subwaypoint_loc[i])
                self.getAllWaypoints(Subwaypoint)
            else:
                return WaypointNode(0, [], waypoint_node.subwaypoint_loc[i])
        return WaypointNode(0, [], waypoint_node.waypoint_loc)

    def getPolygonArea(self):
        area = 0
        prev = self.polygon[-2]
        for i in range(self.polygon.shape[0] - 1):
            now = self.polygon[i]
            xnow, ynow = GridPoly.latlon2xy(now[0], now[1], self.lat_origin, self.lon_origin)
            xpre, ypre = GridPoly.latlon2xy(prev[0], prev[1], self.lat_origin, self.lon_origin)
            area += xnow * ypre - ynow * xpre
            prev = now
        self.PolyArea = area / 2
        print("Area: ", self.PolyArea / 1e6, " km2")
        if self.voiceCtrl:
            os.system("say Area is: {:.1f} squared kilometers".format(self.PolyArea / 1e6))

    @staticmethod
    def deg2rad(deg):
        return deg / 180 * np.pi

    @staticmethod
    def rad2deg(rad):
        return rad / np.pi * 180

    @staticmethod
    def latlon2xy(lat, lon, lat_origin, lon_origin):
        x = GridPoly.deg2rad((lat - lat_origin)) / 2 / np.pi * GridPoly.circumference
        y = GridPoly.deg2rad((lon - lon_origin)) / 2 / np.pi * GridPoly.circumference * np.cos(GridPoly.deg2rad(lat))
        return x, y

    @staticmethod
    def xy2latlon(x, y, lat_origin, lon_origin):
        lat = lat_origin + GridPoly.rad2deg(x * np.pi * 2.0 / GridPoly.circumference)
        lon = lon_origin + GridPoly.rad2deg(y * np.pi * 2.0 / (GridPoly.circumference * np.cos(GridPoly.deg2rad(lat))))
        return lat, lon

    @staticmethod
    def getDistance(coord1, coord2):
        x1, y1 = GridPoly.latlon2xy(coord1[0], coord1[1], GridPoly.lat_origin, GridPoly.lon_origin)
        x2, y2 = GridPoly.latlon2xy(coord2[0], coord2[1], GridPoly.lat_origin, GridPoly.lon_origin)
        dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        return dist


from gmplot import GoogleMapPlotter
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
class Presentation:
    # datapath = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/20210923/Backup_mission_scripts/Onboard/"
    datapath = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/20210923/Mission/lauv-xplore-1-backseat/MASCOT_PORTO/"
    figpath = '/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Report/Porto/fig/'
    lat_origin, lon_origin = 41.10251, -8.669811  # the right bottom corner coordinates
    circumference = 40075000

    def __init__(self):
        self.loaddata()
        self.plotpolygon_presurvey()
        self.plot_corrected_prior()

        # self.plotgrid()
        # self.calpolygon()
        pass

    def loaddata(self):
        self.grid = np.loadtxt(self.datapath + "grid.txt", delimiter = ", ")
        self.polygon = np.loadtxt(self.datapath + "polygon.txt", delimiter=", ")
        self.path_initial_survey = np.loadtxt(self.datapath + "path_initial_survey.txt", delimiter=", ")
        self.prior_polygon = np.loadtxt(self.datapath + "Prior_polygon.txt", delimiter=", ")
        self.prior_corrected = np.loadtxt(self.datapath + "prior_corrected.txt", delimiter=", ")
        print("Data is loaded successfully!")
        print("Polygon: ", self.polygon.shape)
        print("Grid: ", self.grid.shape)
        print("Path_initial_survey: ", self.path_initial_survey.shape)
        print("Prior polygon: ", self.prior_polygon.shape)
        print("Prior corrected: ", self.prior_corrected.shape)

    def plotpolygon_presurvey(self):
        plt.figure(figsize = (10, 10))
        self.polygon = np.append(self.polygon, self.polygon[0, :].reshape(1, -1), axis = 0)
        print(self.polygon)
        ind_surface = self.prior_polygon[:, 2] == -.5
        plt.scatter(self.prior_polygon[ind_surface, 1], self.prior_polygon[ind_surface, 0], vmin = 15, vmax = 38, c = self.prior_polygon[ind_surface, -1], s = 250, cmap = "Paired")
        plt.colorbar()
        plt.plot(self.polygon[:, -1], self.polygon[:, 0], 'k-.', label = "Polygon boundary")
        for i in range(self.path_initial_survey.shape[0]-1):
            if i % 2 != 0:
                if i <= 1:
                    plt.plot(self.path_initial_survey[i:i + 2, 1], self.path_initial_survey[i:i + 2, 0], 'ro-', label = "Surfacing")
                else:
                    plt.plot(self.path_initial_survey[i:i + 2, 1], self.path_initial_survey[i:i + 2, 0], 'ro-')
            else:
                if i <= 0:
                    plt.plot(self.path_initial_survey[i:i + 1, 1], self.path_initial_survey[i:i + 1, 0], 'bo',
                                 markersize=20)
                    plt.plot(self.path_initial_survey[i:i + 2, 1], self.path_initial_survey[i:i + 2, 0], 'go-',
                             label="Diving")
                else:
                    if i % 4 == 0:
                        plt.plot(self.path_initial_survey[i:i + 1, 1], self.path_initial_survey[i:i + 1, 0], 'bo',
                                 markersize=40)
                    else:
                        plt.plot(self.path_initial_survey[i:i + 1, 1], self.path_initial_survey[i:i + 1, 0], 'bo',
                                 markersize=20)
                    plt.plot(self.path_initial_survey[i:i + 2, 1], self.path_initial_survey[i:i + 2, 0], 'go-')

        plt.xlabel("Lon [deg]")
        plt.ylabel("Lat [deg]")
        plt.title("Polygon & Presurvey path")
        plt.legend()
        plt.savefig(self.figpath + "combined.png")
        plt.show()

    def plot_corrected_prior(self):
        plt.figure(figsize = (10, 10))
        self.polygon = np.append(self.polygon, self.polygon[0, :].reshape(1, -1), axis = 0)
        print(self.polygon)
        ind_surface = self.prior_corrected[:, 2] == -.5
        plt.scatter(self.prior_corrected[ind_surface, 1], self.prior_corrected[ind_surface, 0], vmin = 15, vmax = 38, c = self.prior_corrected[ind_surface, -1], s = 250, cmap = "Paired")
        plt.colorbar()
        plt.plot(self.polygon[:, -1], self.polygon[:, 0], 'k-.', label = "Polygon boundary")
        plt.xlabel("Lon [deg]")
        plt.ylabel("Lat [deg]")
        plt.title("Corrected Prior and initial path")
        plt.legend()
        plt.savefig(self.figpath + "prior.png")
        plt.show()

    def plotgrid(self):
        self.plotGridonMap(self.grid)

    def calpolygon(self):
        area = self.getPolygonArea(self.polygon)
        print("Area: ", area)

    @staticmethod
    def deg2rad(deg):
        return deg / 180 * np.pi

    @staticmethod
    def rad2deg(rad):
        return rad / np.pi * 180

    @staticmethod
    def latlon2xy(lat, lon, lat_origin, lon_origin):
        x = Presentation.deg2rad((lat - lat_origin)) / 2 / np.pi * Presentation.circumference
        y = Presentation.deg2rad((lon - lon_origin)) / 2 / np.pi * Presentation.circumference * np.cos(Presentation.deg2rad(lat))
        return x, y

    @staticmethod
    def xy2latlon(x, y, lat_origin, lon_origin):
        lat = lat_origin + Presentation.rad2deg(x * np.pi * 2.0 / Presentation.circumference)
        lon = lon_origin + Presentation.rad2deg(y * np.pi * 2.0 / (Presentation.circumference * np.cos(Presentation.deg2rad(lat))))
        return lat, lon

    def getPolygonArea(self, polygon):
        area = 0
        prev = polygon[-1]
        for i in range(polygon.shape[0] - 1):
            now = polygon[i]
            xnow, ynow = self.latlon2xy(now[0], now[1], self.lat_origin, self.lon_origin)
            xpre, ypre = self.latlon2xy(prev[0], prev[1], self.lat_origin, self.lon_origin)
            area += xnow * ypre - ynow * xpre
            prev = now
        self.PolyArea = area / 2 / 1e6
        return self.PolyArea

    def plotGridonMap(self, grid):
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
        gmap = GoogleMapPlotter(grid[0, 0], grid[0, 1], initial_zoom, apikey=apikey)
        color_scatter(gmap, grid[:, 0], grid[:, 1], np.zeros_like(grid[:, 0]), size=20, colormap='hsv')
        gmap.draw("/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/MapPlot/map.html")

# a = Presentation()

class DataExtractor:
    datapath = '/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/20210923/Mission/lauv-xplore-1/20210923/082919_follow_ntnu/mra/csv/'
    circumference = 40075000

    def __init__(self):
        self.extract_data()

    @staticmethod
    def deg2rad(deg):
        return deg / 180 * np.pi

    @staticmethod
    def rad2deg(rad):
        return rad / np.pi * 180

    def extract_data(self):
        #% Data extraction from the raw data
        rawTemp = pd.read_csv(self.datapath + "Temperature.csv", delimiter=', ', header=0, engine='python')
        rawLoc = pd.read_csv(self.datapath + "EstimatedState.csv", delimiter=', ', header=0, engine='python')
        rawSal = pd.read_csv(self.datapath + "Salinity.csv", delimiter=', ', header=0, engine='python')
        rawDepth = pd.read_csv(self.datapath + "Depth.csv", delimiter=', ', header=0, engine='python')
        # rawGPS = pd.read_csv(datapath + "GpsFix.csv", delimiter=', ', header=0, engine='python')
        # rawCurrent = pd.read_csv(datapath + "EstimatedStreamVelocity.csv", delimiter=', ', header=0, engine='python')

        # To group all the time stamp together, since only second accuracy matters
        rawSal.iloc[:, 0] = np.ceil(rawSal.iloc[:, 0])
        rawTemp.iloc[:, 0] = np.ceil(rawTemp.iloc[:, 0])
        rawCTDTemp = rawTemp[rawTemp.iloc[:, 2] == 'Water Quality Sensor']
        rawLoc.iloc[:, 0] = np.ceil(rawLoc.iloc[:, 0])
        rawDepth.iloc[:, 0] = np.ceil(rawDepth.iloc[:, 0])
        rawDepth.iloc[:, 0] = np.ceil(rawDepth.iloc[:, 0])

        depth_ctd = rawDepth[rawDepth.iloc[:, 2] == 'Water Quality Sensor']["value (m)"].groupby(rawDepth["timestamp"]).mean()
        depth_dvl = rawDepth[rawDepth.iloc[:, 2] == 'DVL']["value (m)"].groupby(rawDepth["timestamp"]).mean()
        depth_est = rawLoc["depth (m)"].groupby(rawLoc["timestamp"]).mean()

        # indices used to extract data
        lat_origin = rawLoc["lat (rad)"].groupby(rawLoc["timestamp"]).mean()
        lon_origin = rawLoc["lon (rad)"].groupby(rawLoc["timestamp"]).mean()
        x_loc = rawLoc["x (m)"].groupby(rawLoc["timestamp"]).mean()
        y_loc = rawLoc["y (m)"].groupby(rawLoc["timestamp"]).mean()
        z_loc = rawLoc["z (m)"].groupby(rawLoc["timestamp"]).mean()
        depth = rawLoc["depth (m)"].groupby(rawLoc["timestamp"]).mean()
        time_loc = rawLoc["timestamp"].groupby(rawLoc["timestamp"]).mean()
        time_sal= rawSal["timestamp"].groupby(rawSal["timestamp"]).mean()
        time_temp = rawCTDTemp["timestamp"].groupby(rawCTDTemp["timestamp"]).mean()
        dataSal = rawSal["value"].groupby(rawSal["timestamp"]).mean()
        dataTemp = rawCTDTemp.iloc[:, -1].groupby(rawCTDTemp["timestamp"]).mean()

        #% Rearrange data according to their timestamp
        data = []
        time_mission = []
        x = []
        y = []
        z = []
        d = []
        sal = []
        temp = []
        lat = []
        lon = []

        for i in range(len(time_loc)):
            if np.any(time_sal.isin([time_loc.iloc[i]])) and np.any(time_temp.isin([time_loc.iloc[i]])):
                time_mission.append(time_loc.iloc[i])
                x.append(x_loc.iloc[i])
                y.append(y_loc.iloc[i])
                z.append(z_loc.iloc[i])
                d.append(depth.iloc[i])
                lat_temp = DataExtractor.rad2deg(lat_origin.iloc[i]) + DataExtractor.rad2deg(x_loc.iloc[i] * np.pi * 2.0 / DataExtractor.circumference)
                lat.append(lat_temp)
                lon.append(DataExtractor.rad2deg(lon_origin.iloc[i]) + DataExtractor.rad2deg(y_loc.iloc[i] * np.pi * 2.0 / (DataExtractor.circumference * np.cos(DataExtractor.deg2rad(lat_temp)))))
                sal.append(dataSal[time_sal.isin([time_loc.iloc[i]])].iloc[0])
                temp.append(dataTemp[time_temp.isin([time_loc.iloc[i]])].iloc[0])
            else:
                print(datetime.fromtimestamp(time_loc.iloc[i]))
                continue

        lat = np.array(lat).reshape(-1, 1)
        lon = np.array(lon).reshape(-1, 1)
        x = np.array(x).reshape(-1, 1)
        y = np.array(y).reshape(-1, 1)
        z = np.array(z).reshape(-1, 1)
        d = np.array(d).reshape(-1, 1)
        sal = np.array(sal).reshape(-1, 1)
        temp = np.array(temp).reshape(-1, 1)
        time_mission = np.array(time_mission).reshape(-1, 1)

        datasheet = np.hstack((time_mission, lat, lon, x, y, z, d, sal, temp))
        np.savetxt(self.datapath + "data.txt", datasheet, delimiter = ", ")

# a = DataExtractor()

