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


data_folder = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Prior/Sep_Prior/"
data_folder_new = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Prior/Sep_Prior/Merged/"
wind_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Porto/Wind/wind_data.txt"

if __name__ == "__main__":
    a = DataGetter(data_folder, data_folder_new, wind_path)


