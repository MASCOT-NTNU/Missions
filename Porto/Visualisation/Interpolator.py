import os
import time
import mat73
from usr_func import *
from scipy.ndimage import gaussian_filter
from datetime import datetime

'''
Interpolator will convert the irregular grid data to regular grid data to simplfy the process afterwards
'''

class Interpolator:

    server = False
    IDUN = False


    def __init__(self):
        print("Hello world")
        self.which_path()
        self.make_new_grid()
        self.get_value_on_new_grid()

    def make_new_grid(self):
        self.lat_lc, self.lon_lc = 41.045, -8.819  # left bottom corner coordinate
        self.nlat = 20  # number of points along lat direction
        self.nlon = 25  # number of points along lon direction
        self.ndepth = 4  # number of layers in the depth direction
        self.max_distance_lat = 18000  # distance of the box along lat
        self.max_distance_lon = 14000  # distance of the box along lon
        self.max_depth = -5  # max depth
        self.alpha = 12
        self.k_n = 1  # k neighbours will be used for averaging

        self.X = np.linspace(0, self.max_distance_lat, self.nlat) # only distance along x direction
        self.Y = np.linspace(0, self.max_distance_lon, self.nlon)
        self.depth_domain = np.linspace(0, self.max_depth, self.ndepth)
        self.Rm = self.get_rotational_matrix(self.alpha)  # rotational matrix used to find the new grid

        self.grid_3d_xy = []
        self.grid_3d_original = []
        self.grid_3d_latlon = []
        for i in range(self.nlat):
            print("nlat: ", i)
            for j in range(self.nlon):
                tmp = self.Rm @ np.array([self.X[i], self.Y[j]])
                xnew, ynew = tmp
                lat_loc, lon_loc = xy2latlon(xnew, ynew, self.lat_lc, self.lon_lc)
                lat_old, lon_old = xy2latlon(self.X[i], self.Y[i], self.lat_lc, self.lon_lc)
                for k in range(self.ndepth):
                    # values_3d[i, j, k] = get_value_at_loc([lat_loc, lon_loc, depth_domain[k]], k_n)
                    self.grid_3d_xy.append([self.X[i], self.Y[j], self.depth_domain[k]])
                    self.grid_3d_original.append([lat_loc, lon_loc, self.depth_domain[k]])
                    self.grid_3d_latlon.append([lat_old, lon_old, self.depth_domain[k]])
        self.grid_3d_xy = np.array(self.grid_3d_xy) # grid in xy
        self.grid_3d_original = np.array(self.grid_3d_original) # grid in original axis
        self.grid_3d_latlon = np.array(self.grid_3d_latlon) # grid in rotated axis
        print("Grid is generated successfully! ")
                # values_gussian_filtered = gaussian_filter(values_3d, 1)

    def get_rotational_matrix(self, alpha):
        R = np.array([[np.cos(deg2rad(alpha)), np.sin(deg2rad(alpha))],
                      [-np.sin(deg2rad(alpha)), np.cos(deg2rad(alpha))]])
        return R

    # def get_value_at_loc(self, loc, k_neighbour, lat_f, lon_f, depth_f, salinity_f):
    #     lat_loc, lon_loc, depth_loc = loc
    #     x_dist, y_dist = latlon2xy(lat_f, lon_f, lat_loc, lon_loc)
    #     depth_dist = depth_f - depth_loc
    #     dist = np.sqrt(x_dist ** 2 + y_dist ** 2 + depth_dist ** 2)  # cannot use lat lon since deg/rad will mix metrics
    #     ind_neighbour = np.argsort(dist)[:k_neighbour]  # use nearest k neighbours to compute the average
    #     value = np.nanmean(salinity_f[ind_neighbour])
    #     return value

    def which_path(self):
        if self.server:
            print("Server mode is activated")
            if self.IDUN:
                self.path = "/cluster/work/yaoling/mascot/delft3d_data_plot/data/"
                self.path_wind = "/cluster/work/yaoling/mascot/delft3d_data_plot/data/wind_data.txt"
                self.path_tide = "/cluster/work/yaoling/mascot/delft3d_data_plot/data/tide.txt"
                self.path_water_discharge = "/cluster/work/yaoling/mascot/delft3d_data_plot/data/data_water_discharge.txt"
                self.figpath = "/cluster/work/yaoling/mascot/delft3d_data_plot/fig/"
            else:
                self.path = "/home/ahomea/y/yaoling/MASCOT/Porto_Data_Processing/Data/"
                self.path_wind = "/home/ahomea/y/yaoling/MASCOT/Porto_Data_Processing/Data/wind_data.txt"
                self.path_tide = "/home/ahomea/y/yaoling/MASCOT/Porto_Data_Processing/Data/tide.txt"
                self.path_water_discharge = "/home/ahomea/y/yaoling/MASCOT/Porto_Data_Processing/Data/data_water_discharge.txt"
                self.figpath = "/home/ahomea/y/yaoling/MASCOT/Porto_Data_Processing/fig/"
        else:
            print("Local mode is activated")
            self.path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Prior/Nov_Prior/"
            self.path_wind = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Wind/wind_data.txt"
            self.path_tide = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Tide/tide.txt"
            self.path_water_discharge = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/WaterDischarge/data_water_discharge.txt"
            self.figpath = None

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

    def get_value_on_new_grid(self):
        files = os.listdir(self.path)
        for file in files:
            if file.endswith(".mat"):
                print(self.path + file)
                # self.load_mat_data(file)


# Import data




# values_3d = np.zeros([nlat, nlon, ndepth])


# for i in range(nlat):
#     print("nlat: ", i)
#     for j in range(nlon):
#         tmp = Rm @ np.array([X[i], Y[j]])
#         xnew, ynew = tmp
#         lat_loc, lon_loc = xy2latlon(xnew, ynew, lat_lc, lon_lc)
#         for k in range(ndepth):
#             values_3d[i, j, k] = get_value_at_loc([lat_loc, lon_loc, depth_domain[k]], k_n)
#             grid_3d.append([X[i], Y[j], depth_domain[k]])
# grid_3d = np.array(grid_3d)
# values_gussian_filtered = gaussian_filter(values_3d, 1)

if __name__ == "__main__":
    a = Interpolator()

#%%
a.lon_lc = -8.819
a.make_new_grid()

#%% Use distance matrix instead to boost the speed of computation

# self = a
# def ind_distance_matrix():
#     self.lat_f = self.lat.flatten()
#     self.lon_f = self.lon.flatten()
#     self.x_f, self.y_f = latlon2xy(self.lat_f, self.lon_f, self.lat_lc, self.lon_lc) # convert to xy domain
#     self.x_f = self.x_f.reshape(-1, 1)
#     self.y_f = self.y_f.reshape(-1, 1)
#
#     self.x_grid = self.grid_3d_xy[:, 0].reshape(-1, 1)
#     self.y_grid = self.grid_3d_xy[:, 1].reshape(-1, 1)
#     self.depth_grid = self.grid_3d_xy[:, 2].reshape(-1, 1)
#
#     self.DM_xgrid = self.x_grid @ np.ones([1, len(self.x_f)])
#     self.DM_xf = np.ones([len(self.x_grid), 1]) @ self.x_f.T
#     self.DM_x = (self.DM_xf - self.DM_xgrid) ** 2
#
#     self.DM_ygrid = self.y_grid @ np.ones([1, len(self.y_f)])
#     self.DM_yf = np.ones([len(self.y_grid), 1]) @ self.y_f.T
#     self.DM_y = (self.DM_yf - self.DM_ygrid) ** 2
#
#     self.DM_zgrid = self.depth_grid @ np.ones([1, len(self.y_f)])

    # for i in range(self.sal_data.shape[0]):
    # for i in [0]:
    # #     print(i)
    #     self.depth_f = self.depth[i, :, :, :].flatten()
    #     self.salinity_f = self.sal_data[i, :, :, :].flatten()
    #
    #     self.DM_zf = np.ones([len(self.depth_grid), 1]) @ self.depth_f.T
    #     self.DM_z = (self.DM_zf - self.DM_zgrid) ** 2
    #
    #     self.DM = self.DM_x + self.DM_y + self.DM_z


# ind_distance_matrix()


#%%
# import logging
#
# logging.basicConfig(filename="test.log",
#                             filemode='a',
#                             format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
#                             datefmt='%H:%M:%S',
#                             level=logging.DEBUG)
#
# logging.info("Running Urban Planning")
#
# logger = logging.getLogger('urbanGUI')


