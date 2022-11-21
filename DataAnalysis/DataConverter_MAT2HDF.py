# ! /usr/bin/env python3
__author__ = "Yaolin Ge"
__copyright__ = "Copyright 2021, The MASCOT Project, NTNU (https://wiki.math.ntnu.no/mascot)"
__credits__ = ["Yaolin Ge"]
__license__ = "MIT"
__version__ = "1.0.1"
__maintainer__ = "Yaolin Ge"
__email__ = "yaolin.ge@ntnu.no"
__status__ = "UnderDevelopment"


from datetime import datetime
import time
import mat73
import os
import h5py


class Mat2HDF5:
    data_path = None # data_path contains the path for the mat file
    data_path_new = None # data_path_new contains the path for the new hdf5 file

    def __init__(self, data_path, data_path_new):
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


class DataMerger(Mat2HDF5): # only used for converting all mat to hdf5
    data_folder = None
    data_folder_new = None

    def __init__(self, data_folder, data_folder_new):
        self.data_folder = data_folder
        self.data_folder_new = data_folder_new
        self.mergeAll()

    def mergeAll(self):
        for s in os.listdir(self.data_folder):
            if s.endswith(".mat"):
                print(s)
                self.data_path = self.data_folder + s
                t = Mat2HDF5(self.data_path, self.data_path[:-4] + ".h5")
                t.loaddata()
                t.mat2hdf()


if __name__ == "__main__":

    # Usage I: convert all mat file in one folder
    data_folder = '/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Prior/Nov_Prior/'
    # data_folder = "/Volumes/LaCie/MASCOT/Data/"
    data_folder_new = '/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Prior/Nov_Prior/'
    a = DataMerger(data_folder, data_folder_new)

    # Usage II: convert only one single mat file
    # data_path = '/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/D2_3D_salinity-021.mat'
    # data_path_new = '/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Delft3D/Delft3D.h5'
    # a = Mat2HDF5(data_path, data_path_new)
    # a.mat2hdf()
