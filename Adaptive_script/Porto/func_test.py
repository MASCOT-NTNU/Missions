import numpy as np

print("hello world")
from usr_func import *

# figpath = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Projects_practice/ES_3D_scratch/fig/EIBV_2D1_Rule_Based/"

#%%
data_path = '/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/In-situ/pmel-20170813T154236.000-lauv-xplore-1.nc'
# data_path = '/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/In-situ/pmel-20170813T152758.000-lauv-xplore-2.nc'
import netCDF4
data = netCDF4.Dataset(data_path)
lat = np.array(data['lat']).reshape(-1, 1)
lon = np.array(data['lon']).reshape(-1, 1)
salinity = np.array(data['sal']).reshape(-1, 1)
import matplotlib.pyplot as plt
plt.scatter(lon, lat, c = salinity, vmin = 33, vmax = 36, cmap = "Paired")
plt.colorbar()
plt.show()

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
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)

ind = np.random.randint(0, x.shape[0] - 1, size = 100) # take out only 5000 random locations, otherwise it takes too long to run
S_sample = salinity.reshape(-1, 1) # sample each frame
residual = S_sample
V_v = Variogram(coordinates = np.hstack((x[ind], y[ind])), values = residual[ind].squeeze(), n_lags = 20, maxlag = 3000, use_nugget=True)

V_v.plot()
print(V_v)

#%%
import h5py
data_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Prior/Maretec/Exemplo_Douro/2021-09-14_2021-09-15/WaterProperties.hdf5"
t = h5py.File(data_path, 'r')
grid = t.get("Grid")
z = grid.get("VerticalZ")
z_0 = np.array(z.get("Vertical_00005"))
print(z_0)
# plt.plot(z_0.flatten())
# plt.show()
lat = grid.get("Latitude")
lon = grid.get("Longitude")
plt.scatter(lon[:-1, :-1], lat[:-1, :-1], c = z_0[0, :, :], vmin = 0, vmax = 30, cmap = "Paired")
plt.colorbar()
plt.show()
# import matplotlib.pyplot as plt
result = t.get("Results")
salinity = result.get("salinity").get('salinity_00001')
print(salinity.shape)
plt.figure()
plt.scatter(lon[:-1, :-1], lat[:-1, :-1], c = salinity[0, :, :],  vmin = 10, vmax = 36, cmap = "Paired")
plt.show()
# plt.scatter(lon[:, :, 0], lat[:, :, 0], c = np.mean(salinity, axis = 0)[:, :, 0], cmap = "Paired")
# plt.colorbar()
# plt.show()

#%%
s = a.timestamp_data[a.ind_selected]
ebby_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Porto/Tide/ebby.txt"
ebby = np.loadtxt(ebby_path, delimiter=", ")

for i in range(len(s)):
    ind = np.where(s[i] < ebby[:, 0])[0][0] - 1
    if s[i] < ebby[ind, 1]:
        print("check: ", datetime.fromtimestamp(s[i]))
        print("high: ", datetime.fromtimestamp(ebby[ind, 0]))
        print("low: ", datetime.fromtimestamp(ebby[ind, 1]))


#%%
import matplotlib.pyplot as plt
import h5py
# data_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Prior/Sep_Prior/Merged/Merged_North_Heavy_2016_SEP_1.h5"
data_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Prior/Sep_Prior/Prior_polygon.h5"
data = h5py.File(data_path, 'r')
lat = data.get("lat_selected")
lon = data.get("lon_selected")
depth = data.get("depth_selected")
salinity = data.get("salinity_selected")
plt.scatter(lon[:, 0], lat[:, 0], c = salinity[:, 0], cmap = "Paired")
plt.colorbar()
plt.show()

#%%

grid_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Prior/Sep_Prior/grid.txt"
grid = np.loadtxt(grid_path, delimiter=", ")

lat = grid[:, 0]
lon = grid[:, 1]
# depth = grid[:, 2]
dist_tolerance =
plt.plot(lon, lat, 'k.')
plt.show()


def find_neighbour_points(loc):
    x, y = loc
    distX = X - x
    distY = Y - y
    dist = np.sqrt(distX ** 2 + distY ** 2)
    ind = np.where(dist <= dist_tolerance)[0]
    return ind

def find_next(ind):
    id = np.random.randint(len(ind))
    return ind[id]

def find_nearest_path(a, b):
    xa, ya = a
    xb, yb = b
    xnext = xa
    ynext = ya
    path_x = []
    path_y = []
    path_x.append(xnext)
    path_y.append(ynext)
    counter = 0
    figpath = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Porto/Setup/Pre_survey/fig/"
    while True:
        if xnext == xb and ynext == yb:
            break
        else:
            ind_next = find_next(find_neighbour_points([xnext, ynext]))
            print(ind_next)
            xnext = X[ind_next]
            ynext = Y[ind_next]
            plt.figure()
            plt.plot(X, Y, 'k.')
            plt.plot(xnext, ynext, 'r.')
            plt.plot(path_x, path_y, 'b-')
            plt.savefig(figpath + "P_{:04d}.png".format(counter))
            plt.close("all")
            path_x.append(xnext)
            path_y.append(ynext)
            counter = counter + 1
            print(counter)
        if counter > 100:
            break


find_nearest_path([0, 0], [4, 4])

#%%
path = '/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Prior/Sep_Prior/Prior_polygon.h5'



