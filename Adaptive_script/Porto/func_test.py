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
# test logistic regression

lat = a.lat[:, :, 0].reshape(-1, 1)
lon = a.lon[:, :, 0].reshape(-1, 1)
depth = a.depth_ave[:, :, 0].reshape(-1, 1)
salinity = a.salinity_ave[:, :, 0].reshape(-1, 1)
ind_nnan = ~np.isnan(lat) & ~np.isnan(lon) & ~np.isnan(depth) & ~np.isnan(salinity)

data = np.hstack((lat[ind_nnan].reshape(-1, 1), lon[ind_nnan].reshape(-1, 1), depth[ind_nnan].reshape(-1, 1), salinity[ind_nnan].reshape(-1, 1)))

np.savetxt("test.txt", data, delimiter = ',')

#%%


#%%
from sklearn.linear_model import LogisticRegression

X = np.hstack((lat, lon, lon ** 2, lon **3 , lon ** 4, lon ** 5))
y = salinity.flatten()
clf = LogisticRegression().fit(X, y)
print(clf.score(X, y))
print(clf.coef_, clf.intercept_)
coef = clf.coef_
beta0 = clf.intercept_

lat_y = - (coef[0, 1] * lon + coef[0, 2] * lon ** 2 + beta0) / coef[0, 0]
plt.scatter(lon, lat, c = salinity, cmap = "Paired")
plt.plot(lon, lat_y, 'r-')
plt.show()
