import os
import mat73
import pandas as pd
import numpy as np

# root_dir = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/D2/"
# for i in range(16, len(os.listdir(root_dir))):
#     datapath = root_dir + os.listdir(root_dir)[i]
#     print(datapath)

# data_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/D2/D2_201612_surface_salinity.mat"
# data_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/D2/D2_201605_surface_salinity.mat"
# data_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/D2/D2_201606_surface_salinity.mat"
data_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/D2/D2_201607_surface_salinity.mat"
sal_data = mat73.loadmat(data_path)

wind_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/conditions/wind_Era5_douro_2012_a_2016.wnd"
wind_data = np.array(pd.read_csv(wind_path, sep = "\t ", header = None))

data = sal_data["data"]
x = data["X"]
y = data["Y"]
z = data["Z"]
Time = data['Time']
salinity = data["Val"]
from datetime import datetime
ref_t_wind = datetime(2005, 1, 1).timestamp()
t_wind = wind_data[:, 0] * 60 + ref_t_wind
v_wind = wind_data[:, 1]
r_wind = wind_data[:, -1]

time_wind = []
speed_wind = []
direction_wind = []
year = int(data_path[59:63])
month = int(data_path[63:65])
yr1 = year
mn1 = month
if month == 12:
    yr2 = yr1 + 1
    mn2 = 1
else:
    yr2 = yr1
    mn2 = mn1 + 1
print(yr1, mn1, yr2, mn2)
start_time = datetime(yr1, mn1, 1)
end_time = datetime(yr2, mn2, 1)

for i in range(len(t_wind)):
    if datetime.fromtimestamp(int(t_wind[i])) >= start_time and datetime.fromtimestamp(int(t_wind[i])) <= end_time:
        time_wind.append(datetime.fromtimestamp(int(t_wind[i])))
        # print(r_wind[i])
        if r_wind[i] < 45 or r_wind[i] >= 315:
            direction_wind.append('North')
        elif r_wind[i] >= 45 and r_wind[i] < 135:
            direction_wind.append('East')
        elif r_wind[i] >= 135 and r_wind[i] < 225:
            direction_wind.append('South')
        else:
            direction_wind.append('West')
        # if r_wind[i] < 45 or r_wind[i] >= 315:
        #     direction_wind.append('South')
        # elif r_wind[i] >= 45 and r_wind[i] < 135:
        #     direction_wind.append('West')
        # elif r_wind[i] >= 135 and r_wind[i] < 225:
        #     direction_wind.append('North')
        # else:
        #     direction_wind.append('East')
sal_ave = []
for i in range(len(time_wind)):
    if (i + 1) * 6 <= salinity.shape[0]:
        sal_ave.append(np.mean(salinity[i * 6 : (i + 1) * 6, :, :], axis = 0))
    else:
        sal_ave.append(np.mean(salinity[i * 6 : -1, :, :], axis=0))
sal_ave = np.array(sal_ave)
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

print("over")

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


#%%
class DataHandler_Delft3D:
    sal_path = None
    wind_path = None
    sal_data = None
    wind_data = None

    def __init__(self, datapath, windpath):
        print("hello world")
        self.set_datapath(datapath)
        self.set_windpath(windpath)
        self.loaddata()
        self.load_wind()

    def set_datapath(self, path):
        self.sal_path = path

    def loaddata(self):
        import mat73
        self.data = mat73.loadmat(self.sal_path)
        data = self.data["data"]
        self.x = data["X"]
        self.y = data["Y"]
        self.z = data["Z"]
        self.Time = data['Time']
        self.sal_data = data["Val"]

    def set_windpath(self, path):
        self.wind_path = path

    def load_wind(self):
        import pandas as pd
        self.wind_data = np.array(pd.read_csv(wind_path, sep="\t ", header = None))
        pass

    def merge_data(self):
        from datetime import datetime
        ref_t_wind = datetime(2005, 1, 1).timestamp()
        t_wind = self.wind_data[:, 0] * 60 + ref_t_wind
        v_wind = self.wind_data[:, 1]
        r_wind = self.wind_data[:, -1]
        self.time_wind = []
        self.speed_wind = []
        self.direction_wind = []
        start_time = datetime(2016, 12, 1)
        end_time = datetime(2017, 1, 1)
        for i in range(len(t_wind)):
            if datetime.fromtimestamp(int(t_wind[i])) >= start_time and datetime.fromtimestamp(
                    int(t_wind[i])) <= end_time:
                self.time_wind.append(datetime.fromtimestamp(int(t_wind[i])))
                self.speed_wind.append(v_wind[i])
                if r_wind[i] < 45 or r_wind[i] >= 315:
                    self.direction_wind.append('South')
                elif r_wind[i] >= 45 and r_wind[i] < 135:
                    self.direction_wind.append('West')
                elif r_wind[i] >= 135 and r_wind[i] < 225:
                    self.direction_wind.append('North')
                else:
                    self.direction_wind.append('East')

        for i in range(len(self.time_wind)):
            pass

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

data_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/D2/D2_201612_surface_salinity.mat"
# wind_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/conditions/wind_Era5_douro_2005_a_2006.wnd"
wind_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/conditions/wind_Era5_douro_2012_a_2016.wnd"
datahandler = DataHandler_Delft3D(data_path, wind_path)
print(datahandler.sal_data.shape)
print(datahandler.wind_data.shape)
print(datahandler.Time.shape)
datahandler.merge_data()

#%%
datahandler.plot_data()

