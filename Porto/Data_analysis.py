import numpy as np
from datetime import datetime
import time
import mat73
import os
import h5py
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from progress.bar import IncrementalBar
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
        self.loaddata()

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


#%%

class DataHandler_Delft3D:
    data_path = None
    data = None
    wind_path = None
    wind_data = None
    figpath = None
    ROUGH = False
    voiceControl = True

    def __init__(self, datapath, windpath, rough = False, voiceControl = True):
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
        self.set_datapath(datapath)
        self.set_windpath(windpath)
        self.load_data()
        self.load_wind()
        self.merge_data()

    def set_figpath(self, figpath):
        self.figpath = figpath
        print("figpath is set correctly, ", self.figpath)

    def set_datapath(self, path):
        self.data_path = path
        print("Path to data is set correctly, ", self.data_path)

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

    def set_windpath(self, path):
        self.wind_path = path
        print("Path to wind data is set correctly, ", self.wind_path)

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
        self.levels = ['Mild', 'Moderate', 'Great']
        id = len(speeds[speeds < wind_speed]) - 1
        return self.levels[id]

    def windangle2directionRough(self, wind_angle):
        angles = np.arange(4) * 90 + 45
        self.directions = ['East', 'South', 'West', 'North']
        id = len(angles[angles < wind_angle]) - 1
        return self.directions[id]

    def windspeed2levelRough(self, wind_speed):
        speeds = np.array([0, 2.5])
        self.levels = ['Calm', 'Windy']
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

    def wind2uv(self, wind_speed, wind_angle):
        wind_angle = self.angle2angle(wind_angle)
        u = wind_speed * np.cos(wind_angle)
        v = wind_speed * np.sin(wind_angle)
        return u, v

    def uv2wind(self, u, v):
        wind_speed = np.sqrt(u ** 2 + v ** 2)
        wind_angle = np.arctan2(v, u) # v is the speed component in y, u is the speed component in x, cartisian normal
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

    def merge_data_explicit(self):
        self.wind_v = []
        self.wind_dir = []
        self.wind_level = []
        for i in range(len(self.timestamp_data)):
            id_wind = (np.abs(self.timestamp_wind - self.timestamp_data[i])).argmin()
            self.wind_v.append(self.wind_speed[id_wind])
            if self.ROUGH:
                self.wind_dir.append(self.wind_angle[id_wind]) # here one can choose whether
                self.wind_level.append(self.wind_speed[id_wind]) # to use rough or not
            else:
                self.wind_dir.append(self.wind_angle[id_wind])
                self.wind_level.append(self.wind_speed[id_wind])
        print("Data is merged correctly!!")
        print("wind levels: ", len(np.unique(self.wind_level)))
        print("wind directions: ", len(np.unique(self.wind_dir)))

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
        bar = IncrementalBar("Countdown", max = len(self.levels) * len(self.directions))
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
                bar.next()
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
        bar = IncrementalBar("Countdown", max = self.salinity.shape[0])
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
            bar.next()
        if self.voiceControl:
            os.system("say Finished plotting time series")
        bar.finish()

    def plotscatter3D(self, frame, layers, camera = dict(x=-1.25, y=-1.25, z=1.25)):
        if self.voiceControl:
            os.system("say I am plotting the scatter data on the 3D grid now")
        Lon = self.lon[:, :, :layers].reshape(-1, 1)
        Lat = self.lat[:, :, :layers].reshape(-1, 1)
        Depth = self.depth[0, :, :, :layers].reshape(-1, 1)
        sal_val = self.salinity[frame, :, :, :layers].reshape(-1, 1)
        # Make 3D plot # #
        fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scene'}]])
        fig.add_trace(
            go.Scatter3d(
                x=Lon.squeeze(), y=Lat.squeeze(), z=Depth.squeeze(),
                mode='markers',
                marker=dict(
                    size=4,
                    color=sal_val.squeeze(),
                    colorscale = "RdBu",
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
            title="Delft 3D data visualisation on " + self.string_date,
            scene_camera_eye=camera,
        )
        plotly.offline.plot(fig, filename=self.figpath + "Scatter3D/Data_" + self.string_date + ".html", auto_open=False)
        if self.voiceControl:
            os.system("say Finished plotting 3D")
        # fig.write_image(self.figpath + "Scatter3D/S_{:04}.png".format(frame), width=1980, height=1080, engine = "orca")

    def plot3Danimation(self):
        x_eye = -1.25
        y_eye = -1.25
        z_eye = .5

        def rotate_z(x, y, z, theta):
            w = x + 1j * y
            return np.real(np.exp(1j * theta) * w), np.imag(np.exp(1j * theta) * w), z
        for i in range(self.salinity.shape[0]):
            xe, ye, ze = rotate_z(x_eye, y_eye, z_eye, -i * .005)
            camera = dict(x=xe, y=ye, z=ze)
            self.plotscatter3D(i, 5, camera)
            print(i)

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

# data_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Delft3D/Delft3D.hdf5"
wind_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Porto/Wind/wind_data.txt"
# datahandler = DataHandler_Delft3D(data_path, wind_path, rough = True)
# datahandler.set_figpath("/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Porto/Delft3D/fig/")
# datahandler.plot_grouppeddata()
# datahandler.plot_grid_on_data(Grid())
# datahandler.plotscatter3D(frame = 150, layers=5)
# datahandler.plot3Danimation()
# datahandler.plot_surface_timeseries() # it has problems, needs to be fixed

class DataGetter(Mat2HDF5, DataHandler_Delft3D):
    '''
    Get data according to date specified
    '''
    data_folder = None
    data_folder_new = None
    date_string = None

    def __init__(self, data_folder, date_string, data_folder_new):
        self.data_folder = data_folder
        self.data_folder_new = data_folder_new
        self.date_string = date_string
        pass

    def mergedata(self, wind_path):
        t1 = time.time()
        lat = []
        lon = []
        depth = []
        salinity = []
        wind_u = []
        wind_v = []
        for file in os.listdir(self.data_folder_new):
            if file.endswith(".h5"):
                datahandler = DataHandler_Delft3D(self.data_folder_new + file, wind_path, rough = True, voiceControl = False)
                datahandler.merge_data_explicit()
                lat.append(datahandler.lat)
                lon.append(datahandler.lon)
                depth.append(datahandler.depth)
                salinity.append(datahandler.salinity)
                u, v = datahandler.wind2uv(np.array(datahandler.wind_level).reshape(-1, 1),
                                           np.array(datahandler.wind_dir).reshape(-1, 1))
                wind_u.append(u)
                wind_v.append(v)
                print(datahandler.lat.shape)
                print(datahandler.lon.shape)
                print(datahandler.depth.shape)
                print(datahandler.salinity.shape)
                print(u.shape)
                print(v.shape)
        print("Here comes the averaging")
        self.lat_merged = np.mean(lat, axis = 0)
        self.lon_merged = np.mean(lon, axis = 0)
        self.depth_merged = np.mean(depth, axis = 0)
        self.salinity_merged = np.mean(salinity, axis = 0)
        wind_u = np.array(wind_u).squeeze()
        wind_v = np.array(wind_v).squeeze()
        wind_u_merged = np.sum(wind_u, axis = 0)
        wind_v_merged = np.sum(wind_v, axis = 0)
        self.wind_speed_merged, self.wind_angle_merged = self.uv2wind(wind_u_merged, wind_v_merged)
        self.wind_level_merged = []
        self.wind_dir_merged = []
        if self.ROUGH:
            for i in range(len(self.wind_speed_merged)):
                self.wind_dir_merged.append(self.windangle2directionRough(self.angle2angle(self.wind_angle_merged[i])))
                self.wind_level_merged.append(self.windspeed2levelRough(self.wind_speed_merged[i]))
        else:
            for i in range(len(self.wind_speed_merged)):
                self.wind_dir_merged.append(self.windangle2direction(self.angle2angle(self.wind_angle_merged[i])))
                self.wind_level_merged.append(self.windspeed2level(self.wind_speed_merged[i]))
        t2 = time.time()
        print(t2 - t1)


    def getdata4wind(self, wind_dir, wind_level):
        print(np.unique(self.wind_dir_merged))


    def getfiles(self):
        self.FOUND = False
        for file in os.listdir(self.data_folder):
            if file.endswith(".mat"):
                if len(self.date_string) == 2:
                    if self.date_string in file[7:9]:
                        print("Found file: ")
                        print(file)
                        self.FOUND = True
                        data_mat = Mat2HDF5(self.data_folder + file, self.data_folder_new + file[:-4] + ".h5")
                        data_mat.mat2hdf()
                else:
                    if self.date_string in file[3:9]:
                        print("Found file: ")
                        print(file)
                        data_mat = Mat2HDF5(self.data_folder + file, self.data_folder_new + file[:-4] + ".h5")
                        data_mat.mat2hdf()
                        self.FOUND = True

        if not self.FOUND:
            if len(self.date_string) == 2:
                print("There is no month ", self.date_string, ", file does not exist, please check!")
            else:
                print("There is no date ", self.date_string, ", file does not exist, please check!")


data_folder = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/D2/"
data_folder_new = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/D2_HDF/"

a = DataGetter(data_folder, "09", data_folder_new)
# a.getfiles()
a.mergedata(wind_path)
a.getdata4wind("okay", "okay")
#%%

plt.scatter(a.lon_merged[:-1, :-1, 0], a.lat_merged[:-1, :-1, 0], c = a.salinity_merged[0, :, :], cmap = 'Paired')
plt.colorbar()
plt.show()





#%%
from Adaptive_script.Porto.Grid import Grid
from skgstat import Variogram

class Coef(Grid):
    data_path = None

    def __init__(self, data_path):
        self.data_path = data_path
        Grid.__init__(self)

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
        os.system("say loading data correctly, it takes {:.1f} seconds".format(t2 - t1))
        print("Lat: ", self.lat.shape)
        print("Lon: ", self.lon.shape)
        print("Depth: ", self.depth.shape)
        print("Salinity: ", self.salinity.shape)
        print("Date: ", self.string_date)
        print("Time: ", self.timestamp_data.shape)


    def latlon2xy(self, lat, lon):
        x = self.deg2rad(lat - self.lat_origin) / 2 / np.pi * self.circumference
        y = self.deg2rad(lon - self.lon_origin) / 2 / np.pi * self.circumference * np.cos(self.deg2rad(lat))
        return x, y

    # def getcoef(self):
        # ind = np.random.randint(0, .shape[0] - 1, size=5000)
        # V_v = Variogram(coordinates=np.hstack((x[ind], y[ind])), values=residual[ind].squeeze(), n_lags=20, maxlag=1500,
        #                 use_nugget=True)
        # # V_v.fit_method = 'trf' # moment method
        # fig = V_v.plot(hist=True)
        # fig.savefig("test1.pdf")
        # print(V_v)



