import os
import mat73
import pandas as pd
import numpy as np
from datetime import datetime
# root_dir = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/D2/"
# for i in range(16, len(os.listdir(root_dir))):
#     datapath = root_dir + os.listdir(root_dir)[i]
#     print(datapath)

# data_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/D2/D2_201612_surface_salinity.mat"
# data_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/D2/D2_201605_surface_salinity.mat"
# data_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/D2/D2_201606_surface_salinity.mat"
# data_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/D2/D2_201607_surface_salinity.mat"
data_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/D2_3D_salinity-021.mat"
sal_data = mat73.loadmat(data_path)

#%%
# wind_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/conditions/wind_Era5_douro_2012_a_2016.wnd"
wind_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/conditions/wind_Era5_douro_2017_a_2019.wnd"
wind_data = np.array(pd.read_csv(wind_path, sep = "\t ", header = None, engine = 'python'))
ref_t_wind = datetime(2005, 1, 1).timestamp()
timestamp_wind = wind_data[:, 0] * 60 + ref_t_wind
wind_speed = wind_data[:, 1]
wind_angle = wind_data[:, -1]

# wind_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/wind_times_serie_porto_obs_2015_2020.txt"
# wind_data = np.array(pd.read_csv(wind_path, sep = "\t", engine = 'python'))
# wind_data = wind_data[:-3, :5]
# yr_wind = wind_data[:, 0]
# hr_wind = wind_data[:, 1]
# timestamp_wind = []
# for i in range(len(yr_wind)):
#     year = int(yr_wind[i][6:])
#     month = int(yr_wind[i][3:5])
#     day = int(yr_wind[i][:2])
#     hour = int(hr_wind[i][:2])
#     timestamp_wind.append(datetime(year, month, day, hour).timestamp())
# timestamp_wind = np.array(timestamp_wind)
# wind_speed = wind_data[:, 3]
# wind_maxspeed = wind_data[:, 4]
# wind_angle = wind_data[:, 2]

data = sal_data["data"]
x = data["X"]
y = data["Y"]
z = data["Z"]
Time = data['Time']
timestamp_data = (Time - 719529) * 24 * 3600 # 719529 is how many days have passed from Jan1 0,
# to Jan1 1970. Since 1970Jan1, is used as the starting index for datetime
salinity = data["Val"]

sal = []
wind_v = []
wind_dir = []
wind_level = []

def windangle2direction(wind_angle):
    angles = np.arange(8) * 45 + 22.5
    directions = ['NorthEast', 'East', 'SouthEast', 'South',
                  'SouthWest', 'West', 'NorthWest', 'North']
    id = len(angles[angles < wind_angle]) - 1
    return directions[id]

def windspeed2level(wind_speed):
    speeds = np.array([0, 2.5, 10])
    levels = ['Mild', 'Moderate', 'Great']
    id = len(speeds[speeds < wind_speed]) - 1
    return levels[id]


def merge_data():
    for i in range(len(timestamp_data)):
        id_wind = (np.abs(timestamp_wind - timestamp_data[i])).argmin()
        wind_v.append(wind_speed[id_wind])
        wind_dir.append(windangle2direction(wind_angle[id_wind]))
        wind_level.append(windspeed2level(wind_speed[id_wind]))

merge_data()

levels = ['Mild', 'Moderate', 'Great']
directions = ['NorthEast', 'East', 'SouthEast', 'South',
              'SouthWest', 'West', 'NorthWest', 'North']

figpath = '/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Porto/Delft3D/fig/'
from matplotlib.gridspec import GridSpec
def group_data():
    counter = 0
    fig = plt.figure(figsize = (80, 30))
    gs = GridSpec(ncols = 8, nrows = 3, figure=fig)
    for i in range(len(levels)):
        idx = np.where(np.array(wind_level) == levels[i])[0]
        sal_temp = salinity[idx, :, :, 0]
        for j in range(len(directions)):
            idy = np.where(np.array(wind_dir)[idx] == directions[j])[0]
            # print(idy)
            ax = fig.add_subplot(gs[i, j])
            if len(idy):
                sal_total = sal_temp[idy, :, :]
                sal_ave = np.mean(sal_total, axis = 0)
                counter = counter + 1
                print(counter)
                im = ax.scatter(x[:, :, 0], y[:, :, 0], c=sal_ave)
                plt.colorbar(im)
            else:
                sal_ave = None
                ax.scatter(x[:, :, 0], y[:, :, 0], c='w')

            ax.set_xlabel('Lon [deg]')
            ax.set_ylabel('Lat [deg]')
            ax.set_title(levels[i] + " " +  directions[j])
    plt.savefig(figpath + "wc2.png")
    plt.show()


group_data()




#%%
# for i in timestamp_data:
    # print(datetime.fromtimestamp(i))

# for i in timestamp_wind:
#     print(datetime.fromtimestamp(i))

#%%
import numpy as np
from datetime import datetime

class DataHandler_Delft3D:
    sal_path = None
    sal_data = None
    wind_path = None
    wind_data = None

    def __init__(self, datapath, windpath):
        self.set_datapath(datapath)
        self.set_windpath(windpath)
        self.loaddata()
        self.load_wind()
        self.merge_data()

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
        print("Data is loaded correctly!")
        print("X: ", self.x.shape)
        print("Y: ", self.y.shape)
        print("Z: ", self.z.shape)
        print("sal_data: ", self.sal_data.shape)

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

    def merge_data(self):
        self.wind_v = []
        self.wind_dir = []
        self.wind_level = []
        for i in range(len(self.timestamp_data)):
            id_wind = (np.abs(self.timestamp_wind - self.timestamp_data[i])).argmin()
            self.wind_v.append(self.wind_speed[id_wind])
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
        figpath = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Porto/Delft3D/fig/"
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
                    im = ax.scatter(x[:, :, 0], y[:, :, 0], c=sal_ave, cmap = 'RdBu')
                    plt.colorbar(im)
                else:
                    ax.scatter(x[:, :, 0], y[:, :, 0], c='w')
                ax.set_xlabel('Lon [deg]')
                ax.set_ylabel('Lat [deg]')
                ax.set_title(self.levels[i] + " " + self.directions[j])

                counter = counter + 1
                print(counter)
        string_date = datetime.fromtimestamp(self.timestamp_data[0]).strftime("%Y_%m")
        plt.savefig(figpath + "WindCondition_" + string_date + ".png")
        print(figpath + "WindCondition_" + string_date + ".png")
        plt.close("all")
        # plt.show()

    @staticmethod
    def plot_data():
        import matplotlib.pyplot as plt
        figpath = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Porto/Delft3D/fig/"
        for i in range(DataHandler_Delft3D.sal_data.shape[0]):
            print(i)
            plt.figure()
            plt.imshow(DataHandler_Delft3D.sal_data[i, :, :])
            plt.colorbar()
            plt.title("%s".format(i))
            plt.savefig(figpath + "I_{:04d}.png".format(i))
            plt.close("all")

data_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/D2/D2_201709_surface_salinity.mat"
# data_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/D2_3D_salinity-021.mat"
# data_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/D2/D2_201612_surface_salinity.mat"

wind_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/conditions/wind_Era5_douro_2017_a_2019.wnd"
# wind_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/conditions/wind_Era5_douro_2005_a_2006.wnd"
# wind_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/conditions/wind_Era5_douro_2012_a_2016.wnd"
# wind_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/wind_times_serie_porto_obs_2015_2020.txt"
datahandler = DataHandler_Delft3D(data_path, wind_path)

# print(datahandler.sal_data.shape)
# print(datahandler.wind_data.shape)
# print(datahandler.Time.shape)
# datahandler.merge_data()
datahandler.plot_grouppeddata()

#%%
datahandler.plot_data()


print("it comes here")
fill_row = []
fill_column = []
for i in range(259 * sal_ave.shape[0]):
    fill_row.append(0)
    # fill_row.append(np.nan)
for i in range(410 * sal_ave.shape[0]):
    fill_column.append(0)
    # fill_column.append(np.nan)
fill_column = np.array(fill_column).reshape([sal_ave.shape[0], -1, 1])
fill_row = np.array(fill_row).reshape([sal_ave.shape[0], 1, -1])

sal_ave = np.concatenate((sal_ave, fill_row), axis = 1)
sal_ave = np.concatenate((sal_ave, fill_column), axis = 2)

ind_east = np.where(np.array(direction_wind) == 'East')[0]
ind_west = np.where(np.array(direction_wind) == 'West')[0]
ind_north = np.where(np.array(direction_wind) == 'North')[0]
ind_south = np.where(np.array(direction_wind) == 'South')[0]


sal_east = np.mean(sal_ave[ind_east], axis = 0)
sal_west = np.mean(sal_ave[ind_west], axis = 0)
sal_north = np.mean(sal_ave[ind_north], axis = 0)
sal_south = np.mean(sal_ave[ind_south], axis = 0)

import matplotlib.pyplot as plt
plt.figure(figsize = (20, 20))
plt.subplot(221)
plt.scatter(x[:, :, 0], y[:, :, 0], c = sal_east)
plt.title("east")

plt.subplot(222)
plt.scatter(x[:, :, 0], y[:, :, 0], c = sal_west)
plt.title("west")

plt.subplot(223)
plt.scatter(x[:, :, 0], y[:, :, 0], c = sal_north)
plt.title("north")

plt.subplot(224)
plt.scatter(x[:, :, 0], y[:, :, 0], c = sal_south)
plt.title("south")

plt.tight_layout()
plt.show()