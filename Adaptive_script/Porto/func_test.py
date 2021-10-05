import numpy as np

print("hello world")
from usr_func import *
<<<<<<< HEAD
<<<<<<< HEAD

# figpath = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Projects_practice/ES_3D_scratch/fig/EIBV_2D1_Rule_Based/"

#%%
import numpy as np
a = np.random.rand(10000, 10000)
import time
t1 = time.time()
b = np.linalg.inv(a)
t2 = time.time()
print("Time consumed: ", t2 - t1)

#%%
grid_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Adaptive_script/Porto/ROS/grid.txt"
grid = np.loadtxt(grid_path, delimiter = ", ")
import matplotlib.pyplot as plt
plt.scatter(grid[:, 1], grid[:, 0])
plt.show()
=======

# figpath = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Projects_practice/ES_3D_scratch/fig/EIBV_2D1_Rule_Based/"

#%%
grid_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Adaptive_script/Porto/ROS/grid.txt"
grid = np.loadtxt(grid_path, delimiter = ", ")
import matplotlib.pyplot as plt
plt.scatter(grid[:, 1], grid[:, 0])
plt.show()


#%%
from bs4 import BeautifulSoup
filepath = "/Users/yaoling/Downloads/test/doc.kml"
with open(filepath, 'r') as f:
    soup = BeautifulSoup(f)
lat_max = float(str(soup.find("south"))[7:-8])
lat_min = float(str(soup.find("north"))[7:-8])
lon_max = float(str(soup.find("east"))[6:-7])
lon_min = float(str(soup.find("west"))[6:-7])

figpath = "/Users/yaoling/Downloads/test/image.png"
path_onboard = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Adaptive_script/Porto/Laptop/"
import cv2
img = cv2.imread(figpath)
val = img[:, :, 0]

lat_img = np.linspace(lat_min, lat_max, val.shape[0])
lon_img = np.linspace(lon_min, lon_max, val.shape[1])

Lat = []
Lon = []
Ref = []
for i in range(len(lat_img)):
    for j in range(len(lon_img)):
        if lat_img[i] >= 41.1 and lat_img[i] <= 41.16 and lon_img[j] >= -8.75 and lon_img[j] <= -8.64:
            Lat.append(lat_img[i])
            Lon.append(lon_img[j])
            Ref.append(val[i, j])

Lat = np.array(Lat).reshape(-1, 1)
Lon = np.array(Lon).reshape(-1, 1)
Ref = np.array(Ref).reshape(-1, 1)
data_satellite = np.hstack((Lat, Lon, Ref))
np.savetxt(path_onboard + "satellite_data.txt", data_satellite, delimiter=", ")
# import matplotlib.pyplot as plt
# plt.figure(figsize = (20, 20))
# plt.scatter(Lon, Lat, c = Ref)
=======

# figpath = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Projects_practice/ES_3D_scratch/fig/EIBV_2D1_Rule_Based/"

#%%
grid_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Adaptive_script/Porto/ROS/grid.txt"
grid = np.loadtxt(grid_path, delimiter = ", ")
import matplotlib.pyplot as plt
plt.scatter(grid[:, 1], grid[:, 0])
plt.show()


#%%
from bs4 import BeautifulSoup
filepath = "/Users/yaoling/Downloads/test/doc.kml"
with open(filepath, 'r') as f:
    soup = BeautifulSoup(f)
lat_max = float(str(soup.find("south"))[7:-8])
lat_min = float(str(soup.find("north"))[7:-8])
lon_max = float(str(soup.find("east"))[6:-7])
lon_min = float(str(soup.find("west"))[6:-7])

figpath = "/Users/yaoling/Downloads/test/image.png"
path_onboard = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Adaptive_script/Porto/Laptop/"
import cv2
img = cv2.imread(figpath)
val = img[:, :, 0]

lat_img = np.linspace(lat_min, lat_max, val.shape[0])
lon_img = np.linspace(lon_min, lon_max, val.shape[1])

Lat = []
Lon = []
Ref = []
for i in range(len(lat_img)):
    for j in range(len(lon_img)):
        if lat_img[i] >= 41.1 and lat_img[i] <= 41.16 and lon_img[j] >= -8.75 and lon_img[j] <= -8.64:
            Lat.append(lat_img[i])
            Lon.append(lon_img[j])
            Ref.append(val[i, j])

Lat = np.array(Lat).reshape(-1, 1)
Lon = np.array(Lon).reshape(-1, 1)
Ref = np.array(Ref).reshape(-1, 1)
data_satellite = np.hstack((Lat, Lon, Ref))
np.savetxt(path_onboard + "satellite_data.txt", data_satellite, delimiter=", ")
# import matplotlib.pyplot as plt
# plt.figure(figsize = (20, 20))
# plt.scatter(Lon, Lat, c = Ref)
# plt.colorbar()
# plt.show()

#%%
data_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Simulator/MissionData_on2021_0921_2237/"
waypoint = np.loadtxt(data_path + "data_path.txt", delimiter=", ")
waypoint = waypoint[1:, :]
sal = np.loadtxt(data_path + "data_salinity.txt", delimiter=", ")
sal = sal[1:]
lat = waypoint[:, 0]
lon = waypoint[:, 1]
depth = waypoint[:, -1]
plt.scatter(lon, lat, c = sal, cmap = "Paired", vmin = 35, vmax =36)
plt.colorbar()
plt.show()
# import plotly.graph_objects as go
# import plotly
# fig = go.Figure(data=[go.Scatter3d(
#     x=lon.squeeze(),
#     y=lat.squeeze(),
#     z=depth.squeeze(),
#     mode='markers',
#     marker=dict(
#         size=12,
#
#         color=sal.squeeze(),  # set color to an array/list of desired values
#         showscale=True,
#         coloraxis="coloraxis"
#     )
# )])
# fig.update_coloraxes(colorscale="jet")
# figpath = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Porto/Setup/Simulation"
# plotly.offline.plot(fig, filename=figpath + "Simulation.html", auto_open=True)


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
import numpy as np
import matplotlib.pyplot as plt

data_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Prior/Sep_Prior/Merged_all/North_Mild_all.h5"
data = h5py.File(data_path, 'r')
lat = data.get("lat")
lon = data.get("lon")
salinity = data.get("salinity")

plt.scatter(lon[:,:, 0], lat[:, :, 0], c = salinity[:, :, 0], vmin = 26, vmax = 36, cmap = "Paired")
plt.colorbar()
plt.show()

#%%

f = open("wind.txt", 'w')
f.write("wind_dir=North, wind_level=Moderate")
f.close()

#%%
f = open ("wind.txt", 'r')
s = f.read()
print(s)
f.close()




#%%



# data_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Prior/Maretec/Exemplo_Douro/2021-09-14_2021-09-15/WaterProperties.hdf5"
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
plt.axvline(-8.7)
plt.axhline(41.14)
plt.show()
# plt.scatter(lon[:, :, 0], lat[:, :, 0], c = np.mean(salinity, axis = 0)[:, :, 0], cmap = "Paired")
>>>>>>> 8e3a5f860eb663746c8363489cee9bb01653ffdd
# plt.colorbar()
# plt.show()

#%%
<<<<<<< HEAD
data_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Simulator/MissionData_on2021_0921_2237/"
waypoint = np.loadtxt(data_path + "data_path.txt", delimiter=", ")
waypoint = waypoint[1:, :]
sal = np.loadtxt(data_path + "data_salinity.txt", delimiter=", ")
sal = sal[1:]
lat = waypoint[:, 0]
lon = waypoint[:, 1]
depth = waypoint[:, -1]
plt.scatter(lon, lat, c = sal, cmap = "Paired", vmin = 35, vmax =36)
plt.colorbar()
plt.show()
# import plotly.graph_objects as go
# import plotly
# fig = go.Figure(data=[go.Scatter3d(
#     x=lon.squeeze(),
#     y=lat.squeeze(),
#     z=depth.squeeze(),
#     mode='markers',
#     marker=dict(
#         size=12,
#
#         color=sal.squeeze(),  # set color to an array/list of desired values
#         showscale=True,
#         coloraxis="coloraxis"
#     )
# )])
# fig.update_coloraxes(colorscale="jet")
# figpath = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Porto/Setup/Simulation"
# plotly.offline.plot(fig, filename=figpath + "Simulation.html", auto_open=True)


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
import numpy as np
import matplotlib.pyplot as plt
>>>>>>> 8e3a5f860eb663746c8363489cee9bb01653ffdd

data_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Prior/Sep_Prior/Merged_all/North_Mild_all.h5"
data = h5py.File(data_path, 'r')
lat = data.get("lat")
lon = data.get("lon")
salinity = data.get("salinity")

<<<<<<< HEAD
#%%
from bs4 import BeautifulSoup
filepath = "/Users/yaoling/Downloads/test/doc.kml"
with open(filepath, 'r') as f:
    soup = BeautifulSoup(f)
lat_max = float(str(soup.find("south"))[7:-8])
lat_min = float(str(soup.find("north"))[7:-8])
lon_max = float(str(soup.find("east"))[6:-7])
lon_min = float(str(soup.find("west"))[6:-7])

figpath = "/Users/yaoling/Downloads/test/image.png"
path_onboard = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Adaptive_script/Porto/Laptop/"
import cv2
img = cv2.imread(figpath)
val = img[:, :, 0]

lat_img = np.linspace(lat_min, lat_max, val.shape[0])
lon_img = np.linspace(lon_min, lon_max, val.shape[1])

Lat = []
Lon = []
Ref = []
for i in range(len(lat_img)):
    for j in range(len(lon_img)):
        if lat_img[i] >= 41.1 and lat_img[i] <= 41.16 and lon_img[j] >= -8.75 and lon_img[j] <= -8.67:
            Lat.append(lat_img[i])
            Lon.append(lon_img[j])
            Ref.append(val[i, j])

Lat = np.array(Lat).reshape(-1, 1)
Lon = np.array(Lon).reshape(-1, 1)
Ref = np.array(Ref).reshape(-1, 1)
data_satellite = np.hstack((Lat, Lon, Ref))
np.savetxt(path_onboard + "satellite_data.txt", data_satellite, delimiter=", ")
import matplotlib.pyplot as plt
plt.figure(figsize = (20, 20))
plt.scatter(Lon, Lat, c = Ref, cmap = "Paired", vmin = 20, vmax = 60)
plt.colorbar()
plt.show()




#%%
data_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Simulator/MissionData_on2021_0921_2237/"
waypoint = np.loadtxt(data_path + "data_path.txt", delimiter=", ")
waypoint = waypoint[1:, :]
sal = np.loadtxt(data_path + "data_salinity.txt", delimiter=", ")
sal = sal[1:]
lat = waypoint[:, 0]
lon = waypoint[:, 1]
depth = waypoint[:, -1]
plt.scatter(lon, lat, c = sal, cmap = "Paired", vmin = 35, vmax =36)
plt.colorbar()
plt.show()
# import plotly.graph_objects as go
# import plotly
# fig = go.Figure(data=[go.Scatter3d(
#     x=lon.squeeze(),
#     y=lat.squeeze(),
#     z=depth.squeeze(),
#     mode='markers',
#     marker=dict(
#         size=12,
#
#         color=sal.squeeze(),  # set color to an array/list of desired values
#         showscale=True,
#         coloraxis="coloraxis"
#     )
# )])
# fig.update_coloraxes(colorscale="jet")
# figpath = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Porto/Setup/Simulation"
# plotly.offline.plot(fig, filename=figpath + "Simulation.html", auto_open=True)


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
import numpy as np
import matplotlib.pyplot as plt

data_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Prior/Sep_Prior/Merged_all/North_Mild_all.h5"
data = h5py.File(data_path, 'r')
lat = data.get("lat")
lon = data.get("lon")
salinity = data.get("salinity")

plt.scatter(lon[:,:, 0], lat[:, :, 0], c = salinity[:, :, 0], vmin = 26, vmax = 36, cmap = "Paired")
plt.colorbar()
plt.show()

#%%

f = open("wind.txt", 'w')
f.write("wind_dir=North, wind_level=Moderate")
f.close()

#%%
f = open ("wind.txt", 'r')
s = f.read()
print(s)
f.close()




=======
plt.scatter(lon[:,:, 0], lat[:, :, 0], c = salinity[:, :, 0], vmin = 26, vmax = 36, cmap = "Paired")
plt.colorbar()
plt.show()

#%%

f = open("wind.txt", 'w')
f.write("wind_dir=North, wind_level=Moderate")
f.close()

#%%
f = open ("wind.txt", 'r')
s = f.read()
print(s)
f.close()




>>>>>>> 8e3a5f860eb663746c8363489cee9bb01653ffdd
#%%



# data_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Prior/Maretec/Exemplo_Douro/2021-09-14_2021-09-15/WaterProperties.hdf5"
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
plt.axvline(-8.7)
plt.axhline(41.14)
plt.show()
# plt.scatter(lon[:, :, 0], lat[:, :, 0], c = np.mean(salinity, axis = 0)[:, :, 0], cmap = "Paired")
# plt.colorbar()
# plt.show()

#%%
=======
>>>>>>> 8e3a5f860eb663746c8363489cee9bb01653ffdd
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
path = '/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Prior/Maretec/Exemplo_Douro/2021-09-18_2021-09-19/WaterProperties.hdf5'



