import os

import matplotlib.pyplot as plt
import numpy as np

from usr_func import *
figpath = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Nidelva/May27/fig/"


#%% Section I: Compute variogram and trend coefficients
data = np.loadtxt("data.txt", delimiter=",")
timestamp = data[:, 0]
lat = data[:, 1]
lon = data[:, 2]
xauv = data[:, 3]
yauv = data[:, 4]
zauv = data[:, 5]
depth = data[:, 6]
sal = data[:, 7]
temp = data[:, 8]

lat_auv = lat + rad2deg(xauv * np.pi * 2.0 / circumference)
lon_auv = lon + rad2deg(yauv * np.pi * 2.0 / (circumference * np.cos(deg2rad(lat_auv))))

depth_obs = [0.5, 1.0, 1.5, 2.0, 2.5]

fp='samples_2020.05.01.nc'
nc = netCDF4.Dataset(fp)
beta0, beta1, sal_residual, temp_residual, x_loc, y_loc = getCoefficients(data, nc, [0.5, 1.0, 1.5, 2.0, 2.5])

# This tries to find the suitable methods and estimators
# for i in range(len(depth_obs)):
for i in range(1):
    V_v = Variogram(coordinates = np.hstack((y_loc[i].reshape(-1, 1), x_loc[i].reshape(-1, 1))),
                    values = temp_residual[i].squeeze(), use_nugget=True, model = "Matern", normalize = False,
                    n_lags = 100) # model = "Matern" check
    # V_v.estimator = 'cressie'
    V_v.fit_method = 'trf' # moment method

    fig = V_v.plot(hist = False)
    fig.suptitle("test")
    # fig = V_v.plot(hist = True)
    print(V_v)

#%%
plt.figure()
plt.plot(sal_residual[0], temp_residual[0], 'k.')
plt.show()
from scipy.stats import pearsonr
print(pearsonr(sal_residual[0].squeeze(), temp_residual[0].squeeze())[0])


#%% Section II: Fit the prior model
lat4, lon4 = 63.446905, 10.419426 # right bottom corner
origin = [lat4, lon4]
distance = 1000
box = BBox(lat4, lon4, distance, 60)
N1 = 25 # number of grid points along north direction
N2 = 25 # number of grid points along east direction
N3 = 5 # number of layers in the depth dimension
N = N1 * N2 * N3 # total number of grid points

XLIM = [0, 1000]
YLIM = [0, 1000]
ZLIM = [0.5, 2.5]
x = np.linspace(XLIM[0], XLIM[1], N1)
y = np.linspace(YLIM[0], YLIM[1], N2)
z = np.array([0.5, 1.0, 1.5, 2.0, 2.5]).reshape(-1, 1)
xm, ym, zm = np.meshgrid(x, y, z)
xv = xm.reshape(-1, 1) # sites1v is the vectorised version
yv = ym.reshape(-1, 1)
zv = zm.reshape(-1, 1)
dx = x[1] - x[0]
dy = y[1] - y[0]
dz = z[1] - z[0]

mu_prior_sal = []
mu_prior_temp = []
coordinates= getCoordinates(box, N1, N2, dx, 60)

TEST_SAL = []
TEST_TEMP = []

for i in range(len(depth_obs)):
    sal_sinmod, temp_sinmod = GetSINMODFromCoordinates(nc, coordinates, depth_obs[i])
    mu_prior_sal.append(beta0[i, 0] + beta1[i, 0] * sal_sinmod) # build the prior based on SINMOD data
    mu_prior_temp.append(beta0[i, 1] + beta1[i, 1] * temp_sinmod)
    TEST_SAL.append(sal_sinmod)
    TEST_TEMP.append(temp_sinmod)

mu_prior_sal = np.array(mu_prior_sal).reshape(-1, 1)
mu_prior_temp = np.array(mu_prior_temp).reshape(-1, 1)
TEST_SAL = np.array(TEST_SAL).reshape(-1, 1)
TEST_TEMP = np.array(TEST_TEMP).reshape(-1, 1)

beta0_sal, beta1_sal, beta0_temp, beta1_temp = getTrend(data, depth_obs) # Compute trend



#%% Create the map plotter:
apikey = 'AIzaSyAZ_VZXoJULTFQ9KSPg1ClzHEFjyPbJUro' # (your API key here)
gmap = gmplot.GoogleMapPlotter(box[-1, 0], box[-1, 1], 14, apikey=apikey)

# Highlight some attractions:
attractions_lats = coordinates[:, 0]
attractions_lngs = coordinates[:, 1]
gmap.scatter(attractions_lats, attractions_lngs, color='#3B0B39', size=4, marker=False)

# Mark a hidden gem:
for i in range(box.shape[0]):
    gmap.marker(box[i, 0], box[i, 1], color='cornflowerblue')

# Draw the map:
gmap.draw(os.getcwd() + '/map.html')
