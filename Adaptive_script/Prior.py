from usr_func import *
lat4, lon4 = 63.446905, 10.419426 # right bottom corner
origin = [lat4, lon4]
distance = 1000
depth_obs = [0.5, 1.0, 1.5, 2.0, 2.5] # planned depth to be observed
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

data = np.loadtxt("data.txt", delimiter=",")
fp='samples_2020.05.01.nc'
nc = netCDF4.Dataset(fp)
beta0, beta1, sal_residual, temp_residual, x_loc, y_loc = getCoefficients(data, nc, depth_obs)

for i in range(len(depth_obs)):
    sal_sinmod, temp_sinmod = GetSINMODFromCoordinates(nc, coordinates, depth_obs[i])
    mu_prior_sal.append(beta0[i, 0] + beta1[i, 0] * sal_sinmod) # build the prior based on SINMOD data
    mu_prior_temp.append(beta0[i, 1] + beta1[i, 1] * temp_sinmod)

mu_prior_sal = np.array(mu_prior_sal).reshape(-1, 1)
mu_prior_temp = np.array(mu_prior_temp).reshape(-1, 1)

