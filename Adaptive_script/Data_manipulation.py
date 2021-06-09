import numpy as np
import matplotlib.pyplot as plt
from skgstat import Variogram
from usr_func import *
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'font.style': 'oblique'})
figpath = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Nidelva/May27/fig/"
## The coordinate system is NED, north, east, down

data = np.loadtxt("data.txt", delimiter=",")
circumference = 40075000

timestamp = data[:, 0]
lat = data[:, 1]
lon = data[:, 2]
x = data[:, 3]
y = data[:, 4]
z = data[:, 5]
depth = data[:, 6]
sal = data[:, 7]
temp = data[:, 8]



#%% Find the suitable data on the surface from the surface (depth around 0) range
error = 0.5

ind_surface = depth < error + 0.25



#%%
plt.scatter(y, x, c = sal)
plt.show()


#%%

# ind_start = 1000
# ind_end = 2000
# x1 = x[ind_start:ind_end].reshape(-1, 1)
# y1 = y[ind_start:ind_end].reshape(-1, 1)
# z1 = z[ind_start:ind_end].reshape(-1, 1)
# sal1 = sal[ind_start:ind_end].reshape(-1, 1)
# temp1 = temp[ind_start:ind_end].reshape(-1, 1)

x2 = x[ind_surface].reshape(-1, 1)
y2 = y[ind_surface].reshape(-1, 1)
z2 = z[ind_surface].reshape(-1, 1)
sal2 = sal[ind_surface].reshape(-1, 1)
temp2 = temp[ind_surface].reshape(-1, 1)



# plt.figure(figsize=(5, 5))
plt.figure()
plt.scatter(y2, x2, c = sal2, cmap = 'viridis')
plt.title("Salinity variation along transect lines")
plt.colorbar()
# plt.savefig(figpath + "Sal2.pdf")
plt.show()

#%%
# origin = [eta_hat_df[" lat (rad)"][0], eta_hat_df[" lon (rad)"][0]]
# lat = eta_hat_df[" lat (rad)"] + eta_hat_df[" x (m)"]*np.pi*2.0/circ
# lon = eta_hat_df[" lon (rad)"] + eta_hat_df[" y (m)"]*np.pi*2.0/(circ*np.cos(lat))
# x_m = (lat-origin[0])*circ/(2*np.pi)
# y_m = (lon-origin[1])*circ*np.cos(lat)/(2*np.pi)

#%% Krige the 2D field with the available data
'''
Assumption:
- depth is on the surface, thus, it might fluctuate a bit, even turns to negative values, but, it is 0
'''
import scipy.spatial.distance as scdist


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


# Setup the grid
n1 = 50 # number of grid points along east direction, or x, or number of columns
n2 = 50 # number of grid points along north direction, or y, or number of rows
n = n1 * n2 # total number of  grid points

XLIM = [-400, 600] # limits in the grid
YLIM = [-200, 800]


sites1 = np.linspace(XLIM[0], XLIM[1], n1)
sites2 = np.linspace(YLIM[0], YLIM[1], n2)
sites1m, sites2m = np.meshgrid(sites1, sites2)
sites1v = sites1m.reshape(-1, 1) # sites1v is the vectorised version
sites2v = sites2m.reshape(-1, 1)

# Compute the distance matrix
grid = np.hstack((sites1v, sites2v))
t = scdist.cdist(grid, grid)
plotf(t, "Distance matrix for the grid", XLIM, YLIM)

obs = np.hstack((x2, y2))
t_obs = scdist.cdist(obs, obs)
plt.figure()
fig = plotf(t_obs, "Distance matrix for the observation sites", XLIM, YLIM)
fig.savefig(figpath + "Dist_obs.pdf")

t_loc_obs = scdist.cdist(grid, obs)
plt.figure()
fig = plotf(t_loc_obs, "Distance matrix for mixed structure", XLIM, YLIM)
fig.savefig(figpath + "Dist_obs_loc.pdf")

#%%

V_v = Variogram(coordinates = np.hstack((x2, y2)), values = sal2.squeeze())
print(V_v)
# fig = plt.figure(figsize=(5, 5))
fig = V_v.plot(hist = False)
fig.savefig(figpath + "variogram2.pdf")


#%%

# Simulate the initial random field
# alpha = 1.0 # beta as in regression model
sigma = np.sqrt(55)  # scaling coef in matern kernel
eta = 3/800 # coef in matern kernel
tau = .01 # iid noise

S_thres = 20

# only one parameter is considered for salinity
beta = [[29.0], [.25], [0.0]] # [intercept, trend along east and north

Sigma = Matern_cov(sigma, eta, t)  # matern covariance
plotf(Sigma, "Matern covariance matrix for the grid", xlim = XLIM, ylim = YLIM)

Sigma_obs = Matern_cov(sigma, eta, t_obs)
plotf(Sigma_obs, "Covariance for obs", xlim = XLIM, ylim = YLIM)
Sigma_loc_obs = Matern_cov(sigma, eta, t_loc_obs)
plotf(Sigma_loc_obs, "Covariance for obs loc", xlim = XLIM, ylim = YLIM)



#%%
# generate the prior of the field
# H = np.hstack((np.ones([n, 1]), sites1v, sites2v)) # different notation for the project
# mu_prior = mu(H, beta).reshape(n, 1)
# plt.close("all")
mu_prior = np.zeros([n, 1])

# rearrange the long vector, so it contains the information having temperature and salinity
# grouped together for each location, i.e., mu_t1, mu_s1, mu_t2, mu_s2, ...,
# fig = plotf(np.copy(mu_prior).reshape(n2, n1), "Salinity prior mean", xlim = XLIM, ylim = YLIM)
# fig.savefig(figpath + "Prior_Sal.pdf")
# plt.close("all")


mu_post = mu_prior + Sigma_loc_obs @ np.linalg.solve(Sigma_obs, sal2)
Sigma_post = Sigma - Sigma_loc_obs @ np.linalg.solve(Sigma_obs, Sigma_loc_obs.T)

mu_post = mu_post.reshape(n1, n2)
# fig = plotf(mu_post, "Salinity posterior mean", xlim = XLIM, ylim = YLIM)
estd = np.sqrt(np.diag(Sigma_post)).reshape(n1, n2)

#%% gradient
grad_x, grad_y = np.gradient(mu_post)

fig = plt.figure(figsize = (18, 5 * 4))
gs = GridSpec(nrows = 4, ncols = 1)
gs.update(wspace = 0.5)

ax0 = fig.add_subplot(gs[0])
im0 = ax0.imshow(mu_post, extent = BBox)
plt.colorbar(im0, fraction=0.04, pad=0.04)
plt.gca().invert_yaxis()
ax0.set(title="Posterior salinity", xlabel='East', ylabel='North')

ax1 = fig.add_subplot(gs[1])
im1 = ax1.imshow(estd, extent = BBox)
plt.colorbar(im1, fraction=0.04, pad=0.04)
plt.gca().invert_yaxis()
ax1.set(title="Prediction error", xlabel='East', ylabel='North')

ax2 = fig.add_subplot(gs[2])
im2 = ax2.imshow(grad_x, extent = BBox)
plt.colorbar(im2, fraction=0.04, pad=0.04)
plt.gca().invert_yaxis()
ax2.set(title="Gradient along x", xlabel='East', ylabel='North')

ax3 = fig.add_subplot(gs[3])
im3 = ax3.imshow(grad_y, extent = BBox)
plt.colorbar(im3, fraction=0.04, pad=0.04)
plt.gca().invert_yaxis()
ax3.set(title="Gradient along y", xlabel='East', ylabel='North')


plt.show()

#%% Insert map
a2 = BBox[1]
a1 = BBox[0]

b2 = BBox[-1]
b1 = BBox[-2]
print(a1, a2, b1, b2)
print(a2 - a1)
print(b2 - b1)

#%%
map = plt.imread('map.png')

plt.figure(figsize = (18, 5))
plt.imshow(mu_post, alpha = .7, extent = BBox)
plt.title("Salinity posterior mean")
plt.xlabel("East")
plt.ylabel("North")
plt.colorbar(fraction=0.045, pad=0.04)
# plt.gca().invert_yaxis()
plt.imshow(map, alpha = .4, extent = BBox)

plt.savefig(figpath + "Sal_post2.pdf")
plt.show()







#%% add map

lat_all = lat + x*np.pi*2.0/circumference
lon_all = lon + y*np.pi*2.0/(circumference*np.cos(lat_all))

lat_all = lat_all / np.pi * 180
lon_all = lon_all / np.pi * 180

BBox = np.round(((lon_all.min(), lon_all.max(),lat_all.min(), lat_all.max())), decimals = 4)
print(BBox)


a = plt.imread("map.png")
plt.imshow(a)
plt.show()

#%%
