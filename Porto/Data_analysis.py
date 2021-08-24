import os
import mat73
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'font.style': 'oblique'})

# data_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/D2/D2_201612_surface_salinity.mat"
# data_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/D2/D2_201605_surface_salinity.mat"
# data_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/D2/D2_201606_surface_salinity.mat"
# data_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/D2/D2_201607_surface_salinity.mat"
data_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/D2_3D_salinity-021.mat"
os.system("say It will take a long time to import data")
sal_data = mat73.loadmat(data_path)
os.system('say Finished Data importing')

#%%
from Adaptive_script.Porto.Grid import Grid
a = Grid()
lat_grid = a.grid_coord[:, 0].reshape(-1, 1)
lon_grid = a.grid_coord[:, 1].reshape(-1, 1)
depth_grid = np.ones_like(lat_grid) * 1
color_grid = np.zeros_like(lat_grid)

#%%
# wind_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/conditions/wind_Era5_douro_2012_a_2016.wnd"
wind_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/conditions/wind_Era5_douro_2017_a_2019.wnd"
wind_data = np.array(pd.read_csv(wind_path, sep = "\t ", header = None, engine = 'python'))
ref_t_wind = datetime(2005, 1, 1).timestamp()
timestamp_wind = wind_data[:, 0] * 60 + ref_t_wind
wind_speed = wind_data[:, 1]
wind_angle = wind_data[:, -1]

#%%
# data = sal_data["data"]
# lon = data["X"]
# lat = data["Y"]
# depth = data["Z"]
# Time = data['Time']
# timestamp_data = (Time - 719529) * 24 * 3600 # 719529 is how many days have passed from Jan1 0,
# # to Jan1 1970. Since 1970Jan1, is used as the starting index for datetime
# salinity = data["Val"]

lon = sal_data['data']["X"]
lat = sal_data['data']['Y']
depth = sal_data['data']['Z']
timestamp = (sal_data['data']['Time'] - 719529) * 24 * 3600
# salinity = np.mean(Delft3D['data']['Val'], axis = 0)
S = np.mean(sal_data['data']['Val'][:, :, :, 0], axis=0)
Lon = lon[:, :, 0].reshape(-1, 1)
Lat = lat[:, :, 0].reshape(-1, 1)
Depth = depth[0, :, :, 0].reshape(-1, 1)

figpath = '/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Porto/Delft3D/fig/rawdata'

def filterNaN(val):
    temp = []
    for i in range(len(val)):
        if not np.isnan(val[i]):
            temp.append(val[i])
    val = np.array(val).reshape(-1, 1)
    return val




#%%

lat_origin, lon_origin = 41.10251, -8.669811
circumference = 40075000
def deg2rad(deg):
    return deg / 180 * np.pi
def rad2deg(rad):
    return rad / np.pi * 180
def latlon2xy(lat, lon, lat_origin, lon_origin):
    x = deg2rad(lat - lat_origin) / 2 / np.pi * circumference
    y = deg2rad(lon - lon_origin) / 2 / np.pi * circumference * np.cos(deg2rad(lat))
    # x_, y_ = self.R.T @ np.vstack(x, y) # convert it back
    return x, y
x, y = latlon2xy(lat, lon, lat_origin, lon_origin)

# X = np.hstack((np.ones_like(x), x, y))
# beta = np.linalg.solve(X.T @ X, X.T @ sal)

# print(beta.round(3))
# for i in beta:
#     print('%f' % i)
plt.scatter(lon, lat, c = sal, cmap = 'RdBu')
plt.title("Salinity mean in Dec, 2017")
plt.xlabel("Lon [deg]")
plt.ylabel("Lat [deg]")
plt.colorbar()
plt.show()
#%% find the residuals

plt.scatter(lon, lat, c = S, cmap = 'RdBu')
plt.title("Salinity mean in Dec, 2017")
plt.xlabel("Lon [deg]")
plt.ylabel("Lat [deg]")
plt.colorbar()
plt.show()


#%%
mu = X @ beta
plt.scatter(y, x, c = mu, cmap = "RdBu", vmin = np.amin(sal), vmax = np.amax(sal))
plt.title("Trend in Dec, 2017")
plt.xlabel("y [m]")
plt.ylabel("x [m]")
plt.colorbar()
plt.show()
residual = sal - mu
plt.scatter(y, x, c = residual, cmap = "RdBu")
plt.title("Residual in Dec, 2017")
plt.xlabel("y [m]")
plt.ylabel("x [m]")
plt.colorbar()
plt.show()

#%%
ind_box = np.where(y >= -7500)[0]

X = np.hstack((np.ones_like(x[ind_box]), x[ind_box], y[ind_box]))
beta = np.linalg.solve(X.T @ X, X.T @ sal[ind_box])
for i in beta:
    print('%f' % i)
mu = X @ beta
plt.scatter(y[ind_box], x[ind_box], c = mu, cmap = "RdBu", vmin = np.amin(sal), vmax = np.amax(sal))
plt.title("Trend in Dec, 2017")
plt.xlabel("y [m]")
plt.ylabel("x [m]")
plt.colorbar()
plt.show()
residual = sal[ind_box] - mu
plt.scatter(y[ind_box], x[ind_box], c = residual, cmap = "RdBu")
plt.title("Residual in Dec, 2017")
plt.xlabel("y [m]")
plt.ylabel("x [m]")
plt.colorbar()
plt.show()

#%%
sigma = np.sqrt(7.17)
eta = 4.5 / 20000
grid = np.hstack((x, y, np.zeros_like(x)))

def compute_H(grid, ksi):
    '''
    :param grid:
    :param ksi:
    :return:
    '''
    X = grid[:, 0].reshape(-1, 1)
    Y = grid[:, 1].reshape(-1, 1)
    Z = grid[:, -1].reshape(-1, 1)

    distX = X @ np.ones([1, X.shape[0]]) - np.ones([X.shape[0], 1]) @ X.T
    distY = Y @ np.ones([1, Y.shape[0]]) - np.ones([Y.shape[0], 1]) @ Y.T
    distXY = distX ** 2 + distY ** 2
    distZ = Z @ np.ones([1, Z.shape[0]]) - np.ones([Z.shape[0], 1]) @ Z.T
    dist = np.sqrt(distXY + (ksi * distZ) ** 2)

    return dist
## Functions used
def Matern_cov(sigma, eta, H):
    '''
    :param sigma: scaling coef
    :param eta: range coef
    :param H: distance matrix
    :return: matern covariance
    '''
    return sigma ** 2 * (1 + eta * H) * np.exp(-eta * H)

ksi = 0
t = compute_H(grid, ksi)
Sigma = Matern_cov(sigma, eta, t)
plt.imshow(Sigma)
plt.show()


#%%
xnew = x[ind_box]
ynew = y[ind_box]
sal_new = sal[ind_box]

from skgstat import Variogram
ind = np.random.randint(0, xnew.shape[0] - 1, size = 5000)
V_v = Variogram(coordinates = np.hstack((xnew[ind], ynew[ind])), values = residual[ind].squeeze(), use_nugget=True)
# V_v.fit_method = 'trf' # moment method
fig = V_v.plot(hist = True)
print(V_v)

#%%
def extractDelft3DFromLocation(Delft3D, location):
    lat, lon, depth, salinity = filterNaN(Delft3D)
    print(lat.shape)
    print(lon.shape)
    print(salinity.shape)
    Sal = []
    for i in range(location.shape[0]):
        lat_desired = location[i, 0].reshape(-1, 1)
        lon_desired = location[i, 1].reshape(-1, 1)
        depth_desired = location[i, 2].reshape(-1, 1)
        print(lat_desired, lon_desired, depth_desired)
        ind = np.argmin((lat - lat_desired) ** 2 + (lon - lon_desired) ** 2)
        print(ind)
        Sal.append(salinity[ind])
        print(salinity[ind])
    Sal = np.array(Sal).reshape(-1, 1)
    return Sal, lat, lon, depth, salinity

coordinates = a.grid_coord
depth = np.ones([coordinates.shape[0], 1]) * .15
location = np.hstack((coordinates, depth))

sal, lat, lon, depth, salinity = extractDelft3DFromLocation(sal_data, location)
#%%
plt.scatter(lon, lat, c = salinity)
plt.scatter(a.grid_coord[:, 1], a.grid_coord[:, 0], c = sal)
plt.colorbar()
plt.show()

#%%
import numpy as np
from datetime import datetime

class DataHandler_Delft3D:
    sal_path = None
    sal_data = None
    wind_path = None
    wind_data = None
    figpath = None

    def __init__(self, datapath, windpath, rough = False):
        self.ROUGH = rough
        if self.ROUGH:
            print("Rough mode is activated!")
        else:
            print("Fine mode is activated!")
        self.set_datapath(datapath)
        self.set_windpath(windpath)
        self.loaddata()
        self.load_wind()
        self.merge_data()

    def set_figpath(self, figpath):
        self.figpath = figpath
        print("figpath is set correctly, ", self.figpath)

    def set_datapath(self, path):
        self.sal_path = path
        print("Path to data is set correctly, ", self.sal_path)

    def loaddata(self):
        import mat73
        self.data = mat73.loadmat(self.sal_path)
        data = self.data["data"]
        self.x = data["X"]
        self.y = data["Y"]
        self.z = data["Z"]
        self.Time = data['Time']
        self.timestamp_data = (self.Time - 719529) * 24 * 3600  # 719529 is how many days have passed from Jan1 0,
        # to Jan1 1970. Since 1970Jan1, is used as the starting index for datetime
        self.sal_data = data["Val"]
        self.string_date = datetime.fromtimestamp(self.timestamp_data[0]).strftime("%Y_%m")
        print("Data is loaded correctly!")
        print("X: ", self.x.shape)
        print("Y: ", self.y.shape)
        print("Z: ", self.z.shape)
        print("sal_data: ", self.sal_data.shape)
        print("Date: ", self.string_date)

    def set_windpath(self, path):
        self.wind_path = path
        print("Path to wind data is set correctly, ", self.wind_path)

    def load_wind(self):
        import pandas as pd
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
            print("Wind data: ", self.wind_path[-31:])
            self.wind_data = np.array(pd.read_csv(self.wind_path, sep="\t ", header = None, engine = 'python'))
            ref_t_wind = datetime(2005, 1, 1).timestamp()
            self.timestamp_wind = self.wind_data[:, 0] * 60 + ref_t_wind
            self.wind_speed = self.wind_data[:, 1]
            self.wind_angle = self.wind_data[:, -1]
        print("Wind data is loaded correctly!")
        print("wind_data: ", self.wind_data.shape)
        print("wind speed: ", self.wind_speed.shape)
        print("wind angle: ", self.wind_angle.shape)
        print("wind timestamp: ", self.timestamp_wind.shape)

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
        # print("Rough mode is activated!")
        angles = np.arange(4) * 90 + 45
        self.directions = ['East', 'South', 'West', 'North']
        id = len(angles[angles < wind_angle]) - 1
        return self.directions[id]

    def windspeed2levelRough(self, wind_speed):
        speeds = np.array([0, 2.5])
        self.levels = ['Calm', 'Windy']
        id = len(speeds[speeds < wind_speed]) - 1
        return self.levels[id]

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
        for i in range(self.x.shape[0]):
            fill_column.append(np.nan)
        fill_column = np.array(fill_column).reshape([-1, 1])
        fill_row = np.array(fill_row).reshape([1, -1])
        sal_ave = np.concatenate((sal_ave, fill_row), axis=0)
        sal_ave = np.concatenate((sal_ave, fill_column), axis=1)
        return sal_ave

    def plot_grouppeddata(self):
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        fig = plt.figure(figsize=(len(self.directions) * 10, len(self.levels) * 10))
        gs = GridSpec(ncols=len(self.directions), nrows=len(self.levels), figure=fig)
        counter = 0
        for i in range(len(self.levels)):
            idx = np.where(np.array(self.wind_level) == self.levels[i])[0]
            if len(self.sal_data.shape) == 4:
                sal_temp = self.sal_data[idx, :, :, 0] # only extract surface data
            else:
                sal_temp = self.sal_data[idx, :, :]
            for j in range(len(self.directions)):
                idy = np.where(np.array(self.wind_dir)[idx] == self.directions[j])[0]
                ax = fig.add_subplot(gs[i, j])
                if len(idy):
                    sal_total = sal_temp[idy, :, :]
                    sal_ave = np.mean(sal_total, axis=0)
                    if sal_ave.shape[0] != self.x.shape[0]:
                        sal_ave = self.refill_unmatched_data(sal_ave)
                    im = ax.scatter(self.x[:, :, 0], self.y[:, :, 0], c=sal_ave)
                    plt.colorbar(im)
                else:
                    ax.scatter(self.x[:, :, 0], self.y[:, :, 0], c='w')
                ax.set_xlabel('Lon [deg]')
                ax.set_ylabel('Lat [deg]')
                ax.set_title(self.levels[i] + " " + self.directions[j])

                counter = counter + 1
                print(counter)

        if self.ROUGH:
            plt.savefig(self.figpath + "WindCondition_" + self.string_date + "_Rough.png")
        else:
            plt.savefig(self.figpath + "WindCondition_" + self.string_date + ".png")
        print(self.figpath + "WindCondition_" + self.string_date + ".png")
        plt.close("all")
        # plt.show()

    def plotscatter3D(self):
        import plotly.graph_objects as go
        import plotly
        plotly.io.orca.config.executable = '/Users/yaoling/anaconda3/bin/orca/'
        plotly.io.orca.config.save()
        from plotly.subplots import make_subplots

        X = self.x[:, :, 0].reshape(-1, 1)
        Y = self.y[:, :, 0].reshape(-1, 1)
        # for i in range(len(self.z.shape[0])):
        for i in [0]:
            print(i)
            Z = self.z[i, :, :, 0].reshape(-1, 1)
            sal_val = self.sal_data[i, :, :, 0].reshape(-1, 1)
            # Make 3D plot # #
            fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scene'}]])
            fig.add_trace(
                go.Scatter3d(
                    x=X, y=Y, z=Z,
                    marker=dict(
                        size=4,
                        color=sal_val,
                        coloraxis="coloraxis",
                        showscale=False
                    ),
                    line=dict(
                        color='darkblue',
                        width=.1
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
                scene_camera_eye=dict(x=-1.25, y=-1.25, z=1.25),
            )
            plotly.offline.plot(fig, filename=self.figpath + "Scatter3D/Data_" + self.string_date + ".html", auto_open=False)
            # fig.write_image(figpath + "sal.png".format(j), width=1980, height=1080)

    def plot_grid_on_data(self, grid):
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        import plotly
        plotly.io.orca.config.executable = '/Users/yaoling/anaconda3/bin/orca/'
        plotly.io.orca.config.save()
        X = self.x[:, :, 0].reshape(-1, 1)
        Y = self.y[:, :, 0].reshape(-1, 1)
        Z = self.z[0, :, :, 0].reshape(-1, 1)
        S = np.mean(self.sal_data[:, :, :, 0], axis = 0).reshape(-1, 1)

        lat_grid = grid.grid_coord[:, 0]
        lon_grid = grid.grid_coord[:, 1]
        depth_grid = np.ones_like(lat_grid) * 1
        fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scene'}]])
        fig.add_trace(
            go.Scatter3d(
                x=X.squeeze(), y=Y.squeeze(), z=Z.squeeze(),
                mode='markers',
                marker=dict(
                    size=4,
                    color=S.squeeze(),
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

    def plot_data(self):
        import matplotlib.pyplot as plt
        for i in range(self.sal_data.shape[0]):
            print(i)
            plt.figure()
            plt.imshow(self.sal_data[i, :, :])
            plt.colorbar()
            plt.title("%s".format(i))
            plt.savefig(self.figpath + "rawdata/I_{:04d}.png".format(i))
            plt.close("all")

# data_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/D2/D2_201909_surface_salinity.mat"
data_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/D2_3D_salinity-021.mat"
# data_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/D2/D2_201612_surface_salinity.mat"

wind_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/conditions/wind_Era5_douro_2017_a_2019.wnd"
# wind_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/conditions/wind_Era5_douro_2005_a_2006.wnd"
# wind_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/conditions/wind_Era5_douro_2012_a_2016.wnd"
# wind_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/wind_times_serie_porto_obs_2015_2020.txt"
datahandler = DataHandler_Delft3D(data_path, wind_path, rough = True)
datahandler.set_figpath("/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Porto/Delft3D/fig/")

# print(datahandler.sal_data.shape)
# print(datahandler.wind_data.shape)
# print(datahandler.Time.shape)
# datahandler.merge_data()
# datahandler.plot_grouppeddata()
datahandler.plot_grid_on_data(Grid())


#%%
datahandler.plot_data()
#%%

import os
os.system('say "Now it is done, congratulations"')
print("Done")

#%%
import os
for i in range(0, 255):
    os.system('ping 192.168.1.' + str(i))
    print(i)

