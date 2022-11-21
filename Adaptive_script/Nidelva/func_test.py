import numpy as np
a = np.random.rand(3, 1)
b = np.random.rand(3, 1)
c = np.random.rand(3, 1)
d = []
d.append(np.hstack((a, b, c)))
d = np.array(d)
print(d.shape)
print(d)



#%%
# figpath = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Nidelva/Path/"
# for i in range(len(Path_PreRun)):
#     plt.figure(figsize=(5, 5))
#     plt.plot(coordinates[:, 1], coordinates[:, 0], 'k.')
#     plt.plot(rad2deg(Path_PreRun[i][1]), rad2deg(Path_PreRun[i][0]), 'r.')
#     plt.savefig(figpath + "P_{:03d}.pdf".format(i))
#     plt.show()

#%%
figpath = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Nidelva/Path/"

for i in range(F.shape[0]):
# for i in range(100):
    plt.figure(figsize=(5, 5))
    plt.plot(coordinates[:, 1], coordinates[:, 0], 'k.')
    loc = F[i, :] @ coordinates
    plt.plot(loc[1], loc[0], 'r.')
    plt.savefig(figpath + "P_{:03d}.pdf".format(i))
    plt.close("all")


#%%
EP_prior = EP_1D(mu_prior_sal, Sigma_prior_sal, Threshold_S)

mup = EP_prior.reshape(N3, N1, N2)
fig = plt.figure(figsize=(35, 5))
gs = GridSpec(nrows = 1, ncols = 5)
for i in range(len(depth_obs)):
    ax = fig.add_subplot(gs[i])
    im = ax.imshow(np.rot90(mup[i, :, :]), vmin = 0.45, vmax = .7, extent = (0, 1000, 0, 1000))
    ax.set(title = "Prior excursion probabilities at depth {:.1f} meter".format(depth_obs[i]))
    plt.colorbar(im)
# fig.savefig(figpath + "EP_Prior.pdf")
plt.show()


#%%


def getTrend(data, depth):
    xauv = data[:, 3].reshape(-1, 1)
    yauv = data[:, 4].reshape(-1, 1)
    depth_auv = data[:, 6].reshape(-1, 1)
    sal_auv = data[:, 7].reshape(-1, 1)
    temp_auv = data[:, 8].reshape(-1, 1)

    depthl = np.array(depth) - err_bound
    depthu = np.array(depth) + err_bound
    beta0_sal = []
    beta1_sal = []
    beta0_temp = []
    beta1_temp = []

    for i in range(len(depth)):
        ind_obs = (depthl[i] <= depth_auv) & (depth_auv <= depthu[i])
        xobs = xauv[ind_obs].reshape(-1, 1)
        yobs = yauv[ind_obs].reshape(-1, 1)
        sal_obs = sal_auv[ind_obs].reshape(-1, 1)
        temp_obs = temp_auv[ind_obs].reshape(-1, 1)
        X = np.hstack((xobs, yobs))

        model_sal = LinearRegression()
        model_sal.fit(X, sal_obs)
        beta0_sal.append(model_sal.intercept_)
        beta1_sal.append(model_sal.coef_)

        model_temp = LinearRegression()
        model_temp.fit(X, temp_obs)
        beta0_temp.append(model_temp.intercept_)
        beta1_temp.append(model_temp.coef_)

    return np.array(beta0_sal), np.array(beta1_sal).squeeze(), np.array(beta0_temp), np.array(beta1_temp).squeeze()




from usr_func import *

a = xy2latlon(0, 0, origin, distance, -60)
b = xy2latlon(0, N2 - 1, origin, distance, -60)
c = xy2latlon(N1 - 1, 0, origin, distance, -60)
d = xy2latlon(N1 - 1, N2 - 1, origin, distance, -60)
plt.figure(figsize=(5, 5))
plt.plot(a[1], a[0], 'k.')
plt.plot(b[1], b[0], 'r.')
plt.plot(c[1], c[0], 'y.')
plt.plot(d[1], d[0], 'b.')
plt.show()

#%%
figpath = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Nidelva/Adaptive/"
mup = mu_prior_sal.reshape(N3, N1, N2)
fig = plt.figure(figsize=(35, 5))
gs = GridSpec(nrows = 1, ncols = 5)
for i in range(len(depth_obs)):
    ax = fig.add_subplot(gs[i])
    im = ax.imshow(np.rot90(mup[i, :, :]), extent = (0, 1000, 0, 1000))
    ax.set(title = "Prior at depth {:.1f} meter".format(depth_obs[i]))
    plt.colorbar(im)
fig.savefig(figpath + "Prior.pdf")
plt.show()

#%%
# This tries out the directional variogram,
# test = DirectionalVariogram(coordinates = np.hstack((y_loc[i].reshape(-1, 1), x_loc[i].reshape(-1, 1))),
#                             values = sal_residual[i].squeeze(),azimuth=90, tolerance=90, maxlag=80, n_lags=200,
#                             use_nugget = True)
# fig = test.plot(hist = False)
# print(test)
# V_v.estimator = 'dowd'
# fig = V_v.plot(hist = True)
# print(V_v)

beta0_sal = np.kron(beta0_sal, np.ones([N1 * N2, 1]))
beta1_sal = np.kron(beta1_sal, np.ones([N1 * N2, 1]))
beta0_temp = np.kron(beta0_temp, np.ones([N1 * N2, 1]))
beta1_temp = np.kron(beta1_temp, np.ones([N1 * N2, 1]))
muTrend_sal = np.zeros([grid.shape[0], 1])
muTrend_temp = np.zeros([grid.shape[0], 1])
#%% Trend for the field
for i in range(grid.shape[0]):
    muTrend_sal[i] = beta0_sal[i, 0] + grid[i, 0:2] @ beta1_sal[i, :]
    muTrend_temp[i] = beta1_temp[i, 0] + grid[i, 0:2] @ beta1_temp[i, :]

# figpath = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Nidelva/Adaptive/"
mup = muTrend_sal.reshape(N3, N1, N2)
fig = plt.figure(figsize=(35, 5))
gs = GridSpec(nrows = 1, ncols = 5)
for i in range(len(depth_obs)):
    ax = fig.add_subplot(gs[i])
    im = ax.imshow(mup[i, :, :], extent = (0, 1000, 0, 1000))
    # plt.gca().invert_yaxis()
    ax.set(title = "Trend at depth {:.1f} meter".format(depth_obs[i]))
    plt.colorbar(im)
# fig.savefig(figpath + "Trend.pdf")
plt.show()


#%%
x1 = np.arange(20)
y1 = np.arange(20)
xx1, yy1 = np.meshgrid(x1, y1)
plt.figure(figsize=(5, 5))
plt.scatter(xx1, yy1)

x2 = np.zeros_like(xx1)
y2 = np.zeros_like(yy1)
for i in range(len(x1)):
    for j in range(len(y1)):
        x2[i, j], y2[i, j] = ind2xy(x1[i], y1[j], distance = 1, alpha = 30)
plt.scatter(x2, y2)
plt.xlim(-25, 25)
plt.ylim(-25, 25)
plt.show()



sal_sinmod, temp_sinmod = GetSINMODFromCoordinates(nc, coordinates, [1, 2, 3])
plt.figure()
salinity = np.mean(nc['salinity'][:, :, :, :], axis=0)
lat_sinmod = np.array(nc['gridLats'][:, :]).reshape(-1, 1)
lon_sinmod = np.array(nc['gridLons'][:, :]).reshape(-1, 1)
plt.scatter(lon_sinmod, lat_sinmod, c = salinity[0, :, :])
plt.scatter(coordinates[:, 1], coordinates[:, 0], c = sal_sinmod[:, 0])
plt.colorbar()
plt.show()




#%%
fig = plt.figure(figsize=(20, 20))
gs = GridSpec(nrows = 3, ncols = 3)

dataFit = np.zeros([sum(ind_obs[0]), 4])


Ny_sinm = len(yc)
Nx_sinm = len(xc)

for i in [0]:
# for i in range(len(ind_obs)):
    ind_depth = np.where(np.array(zc) == depth_obs[i])[0][0]
    xobs = x[ind_obs[i]].reshape(-1, 1)
    yobs = y[ind_obs[i]].reshape(-1, 1)
    zobs = z[ind_obs[i]].reshape(-1, 1)
    depthobs = depth[ind_obs[i]].reshape(-1, 1)
    lat_obs = rad2deg(lat[ind_obs[i]].reshape(-1, 1) + xobs / circumference * 2 * np.pi)
    lon_obs = rad2deg(lon[ind_obs[i]].reshape(-1, 1) + yobs / (circumference * np.cos(deg2rad(lat_obs))) * 2 * np.pi)
    sal_sinm = []
    temp_sinm = []
    y_sinm = []
    x_sinm = []
    sal_obs = sal[ind_obs[i]].reshape(-1, 1)
    temp_obs = temp[ind_obs[i]].reshape(-1, 1)
    for k in range(len(lat_obs)):
        idx = np.argmin((lat_sinmod - lat_obs[k]) ** 2 + (lon_sinmod - lon_obs[k]) ** 2)
        sal_sinm.append(salinity[ind_depth].reshape(-1, 1)[idx])
        temp_sinm.append(temperature[ind_depth].reshape(-1, 1)[idx])
        y_ind, x_ind = np.unravel_index(idx, (Ny_sinm, Nx_sinm))
        y_sinm.append(y_ind)
        x_sinm.append(x_ind)

    axes = fig.add_subplot(gs[i, 0])
    im = axes.scatter(yobs, xobs, c = sal_obs, cmap = 'viridis', vmin = 2, vmax = 27)
    plt.colorbar(im)
    axes.set(title = "Salinity variation along transect lines at {:.1f} m depth".format(depth_obs[i]))

    axes = fig.add_subplot(gs[i, 1])
    im = axes.scatter(y_sinm, x_sinm, c = sal_sinm, cmap = 'viridis', vmin = 2, vmax = 27)
    axes.set(title = "Salinity variation along transect lines at {:.1f} m depth".format(depth_obs[i]))
    plt.colorbar(im)

    axes = fig.add_subplot(gs[i, 2])
    im = axes.plot(sal_sinm, sal_obs, 'k.')
    axes.set(title="Cross plot for depth {} m".format(depth_obs[i]), xlabel = "SINMOD", ylabel = "AUV data")
    # axes.set_xlim([5, 30])
    # axes.set_ylim([5, 30])

plt.show()
# plt.savefig(figpath + "CrossPlot.pdf")

dataFit[:, 0] = sal_sinm
dataFit[:, 1] = temp_sinm
dataFit[:, 2] = sal_obs.squeeze()
dataFit[:, 3] = temp_obs.squeeze()
# np.savetxt(os.getcwd() + "/dataFit.txt", dataFit, delimiter = ",")
#%% Section IV: plot other thing
fig = plt.figure()
for i in range(len(zc)):
    l1, = plt.plot(salinity[i, :, :].reshape(-1, 1), np.ones([len(salinity[i, :, :].reshape(-1, 1)), 1]) * np.array(zc[i]).squeeze(), 'r.')
l2, = plt.plot(sal, depth, 'k.', label = "AUV data")
plt.xlabel("Salinity [psu]")
plt.ylabel("Depth [m]")
plt.title("Salinity over depth varation")
plt.legend([l1, l2], ["SINMOD", "AUV data"])
plt.gca().invert_yaxis()
fig.savefig(figpath + "SalOverDepth.pdf")
plt.show()
#%%


import scipy.spatial.distance as scdist
i = 0
xobs = x[ind_obs[i]].reshape(-1, 1)
yobs = y[ind_obs[i]].reshape(-1, 1)
zobs = z[ind_obs[i]].reshape(-1, 1)
sal_obs = sal[ind_obs[i]].reshape(-1, 1)
temp_obs = temp[ind_obs[i]].reshape(-1, 1)


# Setup the grid
n1 = 50 # number of grid points along east direction, or x, or number of columns
n2 = 50 # number of grid points along north direction, or y, or number of rows
n = n1 * n2 # total number of  grid points

XLIM = [-100, 400] # limits in the grid
YLIM = [-450, 550]


sites1 = np.linspace(YLIM[0], YLIM[1], n1)
sites2 = np.linspace(XLIM[0], XLIM[1], n2)
sites1m, sites2m = np.meshgrid(sites1, sites2)
sites1v = sites1m.reshape(-1, 1) # sites1v is the vectorised version
sites2v = sites2m.reshape(-1, 1)

# Compute the distance matrix
grid = np.hstack((sites1v, sites2v))
t = scdist.cdist(grid, grid)
plotf(t, "Distance matrix for the grid", YLIM, XLIM)

obs = np.hstack((yobs, xobs))
t_obs = scdist.cdist(obs, obs)
fig = plotf(t_obs, "Distance matrix for the observation sites", YLIM, XLIM)

t_loc_obs = scdist.cdist(grid, obs)
fig = plotf(t_loc_obs, "Distance matrix for mixed structure", YLIM, XLIM)


def extractData(SINMOD, depth, coordinates):

    xc = SINMOD['xc']
    yc = SINMOD['yc']
    zc = SINMOD['zc']
    ind_depth = np.where(np.array(zc) == depth)[0][0]
    salinity = np.mean(SINMOD['salinity'][:, ind_depth, :, :], axis=0)
    temperature = np.mean(SINMOD['temperature'][:, ind_depth, :, :], axis=0)
    plt.imshow(salinity)
    plt.show()
    lat = np.array(SINMOD['gridLats'][:,:])
    lon = np.array(SINMOD['gridLons'][:,:])
    print(lat_sinmod.shape)
    maskVector = np.zeros_like(lat_sinmod)
    print(maskVector.shape)
    nrow, ncol = lat_sinmod.shape
    origin = box[-1]
    p1 = box[0]
    p2 = box[1]
    p3 = box[2]
    ko2 = (p2[0] - origin[0]) / (p2[-1] - origin[-1])
    ko3 = (p3[0] - origin[0]) / (p3[-1] - origin[-1])
    k12 = (p1[0] - p2[0]) / (p1[-1] - p2[-1])
    k13 = (p1[0] - p3[0]) / (p1[-1] - p3[-1])
    b12 = p1[0] - k12 * p1[-1]
    b13 = p1[0] - k13 * p1[-1]

#%%
def plotf(Y, string, xlim, ylim):
    '''
    :param Y:
    :param string:
    :return:
    '''
    # plt.figure(figsize=(5,5))
    fig = plt.gcf()
    plt.imshow(Y, extent=(xlim[0], xlim[1], ylim[0], ylim[1]))
    plt.title(string)
    plt.xlabel("East")
    plt.ylabel("North")
    plt.colorbar(fraction=0.045, pad=0.04)
    # plt.gca().invert_yaxis()
    plt.show()
    # plt.savefig()
    return fig



#%%
BBox = np.round(((lon_all.min(), lon_all.max(),lat_all.min(), lat_all.max())), decimals = 4)
sal_sinmod = salinity[ind_sinmod]
temp_sinmod = temperature[ind_sinmod]
#%% Extract SINMOD data to build the prior

fp='samples_2020.05.01.nc'
nc = netCDF4.Dataset(fp)
salinity = np.mean(nc['salinity'][:,0,:,:], axis = 0).reshape(-1, 1)
temperature = np.mean(nc['temperature'][:,0,:,:], axis = 0).reshape(-1, 1)
xc = nc['xc']
yc = nc['yc']
zc = nc['zc']
lat_sinmod = np.array(nc['gridLats'][:,:]).reshape(-1, 1)
lon_sinmod = np.array(nc['gridLons'][:,:]).reshape(-1, 1)

lat_all = lat + x*np.pi*2.0/circumference
lon_all = lon + y*np.pi*2.0/(circumference*np.cos(lat_all))
lat_all = lat_all / np.pi * 180
lon_all = lon_all / np.pi * 180

ind_sinmod = (lat_sinmod >= lat_all.min()) & (lat_sinmod <= lat_all.max()) & \
             (lon_sinmod >= lon_all.min()) & (lon_sinmod <= lon_all.max())


#%%

def get_prime_factors(number):
    # create an empty list and later I will
    # run a for loop with range() function using the append() method to add elements to the list.
    prime_factors = []

    # First get the number of two's that divide number
    # i.e the number of 2's that are in the factors
    while number % 2 == 0:
        prime_factors.append(2)
        number = number / 2

    # After the above while loop, when number has been
    # divided by all the 2's - so the number must be odd at this point
    # Otherwise it would be perfectly divisible by 2 another time
    # so now that its odd I can skip 2 ( i = i + 2) for each increment
    for i in range(3, int(math.sqrt(number)) + 1, 2):
        while number % i == 0:
            prime_factors.append(int(i))
            number = number / i


    # Here is the crucial part.
    # First quick refreshment on the two key mathematical conjectures of Prime factorization of any non-Prime number
    # Which is - 1. If n is not a prime number AT-LEAST one Prime factor would be less than sqrt(n)
    # And - 2. If n is not a prime number - There can be AT-MOST 1 prime factor of n greater than sqrt(n).
    # Like 7 is a prime-factor for 14 which is greater than sqrt(14)
    # But if the above loop DOES NOT go beyond square root of the initial n.
    # Then how does that greater than sqrt(n) prime-factor
    # will be captured in my prime factorization function.
    # ANS to that is - in my first for-loop I am dividing n with the prime number if that prime is a factor of n.
    # Meaning, after this first for-loop gets executed completely, the adjusted initial n should become
    # either 1 or greater than 1
    # And if n has NOT become 1 after the previous for-loop, that means that
    # The remaining n is that prime factor which is greater that the square root of initial n.
    # And that's why in the next part of my algorithm, I need to check whether n becomes 1 or not,
    if number > 2:
        prime_factors.append(int(number))

    return prime_factors


print(get_prime_factors(len(sal_sinmod)))



#%%
