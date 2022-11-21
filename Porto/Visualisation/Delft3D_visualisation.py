import mat73
import time
import matplotlib.pyplot as plt
import os
from datetime import datetime
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'font.style': 'oblique'})
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable

# server = False
server = True
if server:
    from usr_func import *
else:
    from Porto.Visualisation.usr_func import *


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
            self.path = "/home/ahomea/y/yaoling/MASCOT/Porto_Data_Processing/Data/"
            self.path_wind = "wind_data.txt"
            self.path_tide = "tide.txt"
            self.path_water_discharge = "data_water_discharge.txt"
            self.figpath = "/home/ahomea/y/yaoling/MASCOT/Porto_Data_Processing/Data/fig/NHEbbW/"
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

    def make_png(self):
        counter = 0
        files = os.listdir(self.path)
        files = sorted(files)
        for file in files:
            # if file.endswith("9_sal_3.mat"):
            if file.endswith(".mat"):
                self.load_mat_data(file)
                for i in range(self.sal_data.shape[0]):
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

                    fig = plt.figure(figsize=(10, 10))
                    gs = GridSpec(ncols=10, nrows=10, figure=fig)
                    ax = fig.add_subplot(gs[:, :-1])
                    im = ax.scatter(self.lon[:, :, 0], self.lat[:, :, 0], c=self.sal_data[i, :, :, 0], vmin=15,
                                    vmax=36,
                                    cmap="Paired")
                    ax.quiver(self.lon_river_mouth, self.lat_river_mouth, u, v, scale=30)

                    if self.tide[ind_tide, 2] == 1:
                        if self.timestamp_data[i] >= self.tide[ind_tide, 0]:
                            ax.quiver(-8.64, 41.2, -1, 0, color='g')
                            ax.text(-8.65, 41.205, 'Ebb tide')
                        else:
                            ax.quiver(-8.65, 41.2, 1, 0, color="r")
                            ax.text(-8.65, 41.205, 'Flood tide')
                    else:
                        if self.timestamp_data[i] >= self.tide[ind_tide, 0]:
                            ax.quiver(-8.65, 41.2, 1, 0, color="r")
                            ax.text(-8.65, 41.205, 'Flood tide')
                        else:
                            ax.quiver(-8.64, 41.2, -1, 0, color='g')
                            ax.text(-8.65, 41.205, 'Ebb tide')

                    ax.text(-8.7, 41.04, "Wind direction: " + wind_dir)
                    ax.text(-8.7, 41.035, "Wind level: " + wind_level)

                    ax.set_xlabel("Lon [deg]")
                    ax.xaxis.set_label_position('top')
                    ax.xaxis.tick_top()
                    ax.set_ylabel("Lat [deg]")
                    ax.set_title("Surface salinity on " + datetime.fromtimestamp(self.timestamp_data[i]).strftime(
                        "%Y%m%d - %H:%M"))

                    divider = make_axes_locatable(ax)
                    cax = divider.new_vertical(size="2%", pad=0.05, pack_start=True)
                    fig.add_axes(cax)
                    fig.colorbar(im, cax=cax, orientation="horizontal")

                    ax = fig.add_subplot(gs[:, -1])
                    if self.water_discharge[ind_water_discharge, 1] <= 400:
                        ax.bar(0, self.water_discharge[ind_water_discharge, 1], width=.1, align="edge", color='green')
                    elif self.water_discharge[ind_water_discharge, 1] <= 800:
                        ax.bar(0, self.water_discharge[ind_water_discharge, 1], width=.1, align="edge", color='orange')
                    else:
                        ax.bar(0, self.water_discharge[ind_water_discharge, 1], width=.1, align="edge", color="red")
                    ax.text(.015, self.water_discharge[ind_water_discharge, 1] + 10,
                            str(self.water_discharge[ind_water_discharge, 1]))
                    ax.set_ylim([0, 1000])
                    ax.set_ylabel(r'Water discharge $m^3/s$')
                    ax.yaxis.set_label_position("right")
                    ax.yaxis.tick_right()
                    plt.tick_params(
                        axis='x',  # changes apply to the x-axis
                        which='both',  # both major and minor ticks are affected
                        bottom=False,  # ticks along the bottom edge are off
                        top=False,  # ticks along the top edge are off
                        labelbottom=False)  # labels along the bottom edge are off
                    plt.tick_params(
                        axis='y',  # changes apply to the x-axis
                        which='both',  # both major and minor ticks are affected
                        right=False,  # ticks along the bottom edge are off
                        left=False,  # ticks along the top edge are off
                        labelright=False)  # labels along the bottom edge are off

                    if self.server:
                        plt.savefig(self.figpath + "I_{:05d}.png".format(counter))
                        plt.close("all")
                    else:
                        plt.show()
                    counter = counter + 1
                    print(counter)

if __name__ == "__main__":
    a = Delft3D(server = server)


