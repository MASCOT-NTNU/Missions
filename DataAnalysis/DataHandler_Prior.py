# ! /usr/bin/env python3
__author__ = "Yaolin Ge"
__copyright__ = "Copyright 2021, The MASCOT Project, NTNU (https://wiki.math.ntnu.no/mascot)"
__credits__ = ["Yaolin Ge"]
__license__ = "MIT"
__version__ = "1.0.1"
__maintainer__ = "Yaolin Ge"
__email__ = "yaolin.ge@ntnu.no"
__status__ = "UnderDevelopment"


from DataAnalysis.usr_func import *
import time
import os
import h5py
from DataAnalysis.DataHandler_Delft3D import DataHandler_Delft3D


class DataHandler_Prior:
    '''
    Get data according to date specified and wind direction
    '''
    data_folder = None # folder contains Delft3D data
    data_folder_new = None # folder contains merged data
    wind_path = None # wind path
    ebb_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Tide/ebb.txt"

    def __init__(self, data_folder, data_folder_new, wind_path):
        self.depth_layers = 8 # top 8 layers will be chosen for saving the data, to make sure it has enough space to operate
        self.data_folder = data_folder
        self.data_folder_new = data_folder_new
        self.wind_path = wind_path
        self.load_ebb()

        # self.getAllPriorData()
        # self.checkMerged()
        pass

    def load_ebb(self):
        print("Loading ebb...")
        self.ebb = np.loadtxt(self.ebb_path, delimiter=", ")
        print("Ebb is loaded correctly!")

    def load_delft3d(self, data_path): #load merged data with Delft3D and wind data
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

    def isEbb(self, timestamp): # check if it is in phase with ebb
        if len(np.where(timestamp < self.ebb[:, 0])[0]) > 0:
            ind = np.where(timestamp < self.ebb[:, 0])[0][0] - 1 # check the index for ebb start
            if timestamp < self.ebb[ind, 1]:
                return True
            else:
                return False
        else:
            return False

    def merge_data_for_wind(self, wind_dir, wind_level, data_path):
        print("Now the merging will start...")
        self.load_delft3d(data_path)
        self.lat_merged = self.lat[:, :, :self.depth_layers] # only extract top layers
        self.lon_merged = self.lon[:, :, :self.depth_layers]
        self.ind_selected = np.where((np.array(self.wind_dir) == wind_dir) & (np.array(self.wind_level) == wind_level))[0] # indices for selecting the time frames

        if np.any(self.ind_selected):
            self.timestamp_selected = self.timestamp_data[self.ind_selected]
            self.ind_selected_ebb = []
            print("before ebb checking: ", len(self.ind_selected))
            print("len of selected timestamp: ", len(self.timestamp_selected))
            for i in range(len(self.timestamp_selected)):
                if self.isEbb(self.timestamp_selected[i]):
                    self.ind_selected_ebb.append(self.ind_selected[i])
            print("after ebb checking: ", len(self.ind_selected_ebb))

            if len(self.ind_selected_ebb) > 0:
                # print(self.ind_selected_ebb)
                print("Found ", wind_dir, wind_level, " {:d} timeframes are used to average".format(len(self.ind_selected_ebb)))
                self.salinity_merged = np.mean(self.salinity[self.ind_selected_ebb, :, :, :self.depth_layers], axis = 0)
                self.depth_merged = np.mean(self.depth[self.ind_selected_ebb, :, :, :self.depth_layers], axis = 0)
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
        print("Finished merging...")

    def merge_all_data_from_file(self):
        wind_dirs = ['North', 'South', 'West', 'East'] # get wind_data for all conditions
        wind_levels = ['Mild', 'Moderate', 'Heavy'] # get data for all conditions

        counter = 0
        os.system("say Start merging all the data")
        for wind_dir in wind_dirs:
            for wind_level in wind_levels:
                print("wind_dir: ", wind_dir)
                print("wind_level: ", wind_level)
                for file in os.listdir(self.data_folder):
                    if file.endswith(".h5"):
                        print(self.data_folder + file)
                        t1 = time.time()
                        self.merge_data_for_wind(wind_dir, wind_level, self.data_folder + file)
                        t2 = time.time()
                        os.system("say Step {:d} of {:d}, time consumed {:.1f} seconds".format(counter, len(os.listdir(self.data_folder)) * len(wind_dirs) * len(wind_levels), t2 - t1))
                        print("Step {:d} of {:d}, time consumed {:.1f} seconds".format(counter, len(os.listdir(self.data_folder)) * len(wind_dirs) * len(wind_levels), t2 - t1))
                        counter = counter + 1

    def check_for_wind_condition(self):
        wind_dir = "North"
        wind_level = "Heavy"
        data_folder = "/Volumes/LaCie/MASCOT/Data/Nov/h5/"

        for file in os.listdir(data_folder):
            if file.endswith(".h5"):
                self.merge_data_for_wind(wind_dir, wind_level, data_folder + file)

    def Average_all(self):
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
        path_operation_area = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Adaptive_script/Porto/Onboard/Config/OperationArea.txt"
        OpArea = np.loadtxt(path_operation_area, delimiter = ", ")
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
                plt.scatter(self.lon_test[:, :, 0], self.lat_test[:, :, 0], c = self.salinity_test[:, :, 0], vmin = 15, vmax = 36, cmap = "Paired")
                plt.plot(OpArea[:, 1], OpArea[:, 0], 'k-.')
                plt.xlabel('Lon [deg]')
                plt.ylabel("Lat [deg]")
                plt.title(files[i])
                plt.colorbar()
                counter_plot = counter_plot + 1
        figpath = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Prior/Nov_Prior/fig/"
        plt.savefig(figpath + "prior.pdf")
        plt.show()
        # for wind_dir in wind_dirs:
        #     for wind_level in wind_levels:
        #         for folder in folders:
        #             folder_content = os.listdir(folder)



if __name__ == "__main__":
    # data_folder = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Prior/Nov_Prior/"
    # data_folder_new = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Prior/Nov_Prior/Merged/"
    wind_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Wind/wind_data.txt"

    data_folder = "/Volumes/LaCie/MASCOT/Data/Nov/h5/"
    data_folder_new = "/Volumes/LaCie/MASCOT/Data/Nov/h5/"
    a = DataHandler_Prior(data_folder, data_folder_new, wind_path)
    a.check_for_wind_condition()
    # a.merge_all_data_from_file()
    # a.Average_all()
    # a.checkMerged()
    # data_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Prior/Nov_Prior/Nov2019_sal_1.h5"
    # wind_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Wind/wind_data.txt"
    # datahandler = DataHandler_Delft3D(data_path, wind_path, rough = True)
    # datahandler.set_figpath("/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Porto/Delft3D/fig/")
    # datahandler.plot_grouppeddata()
    # datahandler.plot_grid_on_data(Grid())
    # datahandler.plotscatter3D(layers=1, frame = -1)
    # datahandler.plot3Danimation()
    # datahandler.plot_surface_timeseries() # it has problems, needs to be fixed


