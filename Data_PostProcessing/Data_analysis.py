import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
import os
import scipy.spatial.distance as scdist
from usr_func import *
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

circumference = 40075000
datapath = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Nidelva/July06/Data/"
figpath = '/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Nidelva/July06/fig/'
SINMOD_datapath = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Adaptive_script/"

#% Data extraction from the raw data
rawTemp = pd.read_csv(datapath + "Temperature.csv", delimiter=', ', header=0, engine='python')
rawLoc = pd.read_csv(datapath + "EstimatedState.csv", delimiter=', ', header=0, engine='python')
rawSal = pd.read_csv(datapath + "Salinity.csv", delimiter=', ', header=0, engine='python')
rawDepth = pd.read_csv(datapath + "Depth.csv", delimiter=', ', header=0, engine='python')

# To group all the time stamp together, since only second accuracy matters
rawSal.iloc[:, 0] = np.ceil(rawSal.iloc[:, 0])
rawTemp.iloc[:, 0] = np.ceil(rawTemp.iloc[:, 0])
rawCTDTemp = rawTemp[rawTemp.iloc[:, 2] == 'SmartX']
rawLoc.iloc[:, 0] = np.ceil(rawLoc.iloc[:, 0])
rawDepth.iloc[:, 0] = np.ceil(rawDepth.iloc[:, 0])
rawDepth.iloc[:, 0] = np.ceil(rawDepth.iloc[:, 0])

depth_ctd = rawDepth[rawDepth.iloc[:, 2] == 'SmartX']["value (m)"].groupby(rawDepth["timestamp"]).mean()
depth_dvl = rawDepth[rawDepth.iloc[:, 2] == 'DVL']["value (m)"].groupby(rawDepth["timestamp"]).mean()
depth_est = rawLoc["depth (m)"].groupby(rawLoc["timestamp"]).mean()

# indices used to extract data
lat_origin = rawLoc["lat (rad)"].groupby(rawLoc["timestamp"]).mean()
lon_origin = rawLoc["lon (rad)"].groupby(rawLoc["timestamp"]).mean()
x_loc = rawLoc["x (m)"].groupby(rawLoc["timestamp"]).mean()
y_loc = rawLoc["y (m)"].groupby(rawLoc["timestamp"]).mean()
z_loc = rawLoc["z (m)"].groupby(rawLoc["timestamp"]).mean()
depth = rawLoc["depth (m)"].groupby(rawLoc["timestamp"]).mean()
time_loc = rawLoc["timestamp"].groupby(rawLoc["timestamp"]).mean()
time_sal= rawSal["timestamp"].groupby(rawSal["timestamp"]).mean()
time_temp = rawCTDTemp["timestamp"].groupby(rawCTDTemp["timestamp"]).mean()
dataSal = rawSal["value (psu)"].groupby(rawSal["timestamp"]).mean()
dataTemp = rawCTDTemp.iloc[:, -1].groupby(rawCTDTemp["timestamp"]).mean()

#% Rearrange data according to their timestamp
data = []
time_mission = []
xauv = []
yauv = []
zauv = []
dauv = []
sal_auv = []
temp_auv = []
lat_auv = []
lon_auv = []

for i in range(len(time_loc)):
    if np.any(time_sal.isin([time_loc.iloc[i]])) and np.any(time_temp.isin([time_loc.iloc[i]])):
        time_mission.append(time_loc.iloc[i])
        xauv.append(x_loc.iloc[i])
        yauv.append(y_loc.iloc[i])
        zauv.append(z_loc.iloc[i])
        dauv.append(depth.iloc[i])
        lat_temp = rad2deg(lat_origin.iloc[i]) + rad2deg(x_loc.iloc[i] * np.pi * 2.0 / circumference)
        lat_auv.append(lat_temp)
        lon_auv.append(rad2deg(lon_origin.iloc[i]) + rad2deg(y_loc.iloc[i] * np.pi * 2.0 / (circumference * np.cos(deg2rad(lat_temp)))))
        sal_auv.append(dataSal[time_sal.isin([time_loc.iloc[i]])].iloc[0])
        temp_auv.append(dataTemp[time_temp.isin([time_loc.iloc[i]])].iloc[0])
    else:
        print(datetime.fromtimestamp(time_loc.iloc[i]))
        continue

lat4, lon4 = 63.446905, 10.419426  # right bottom corner
lat_auv = np.array(lat_auv).reshape(-1, 1)
lon_auv = np.array(lon_auv).reshape(-1, 1)
Dx = deg2rad(lat_auv - lat4) / 2 / np.pi * circumference
Dy = deg2rad(lon_auv - lon4) / 2 / np.pi * circumference * np.cos(deg2rad(lat_auv))

xauv = np.array(xauv).reshape(-1, 1)
yauv = np.array(yauv).reshape(-1, 1)

alpha = deg2rad(60)
Rc = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
TT = (Rc @ np.hstack((Dx, Dy)).T).T
xauv_new = TT[:, 0].reshape(-1, 1)
yauv_new = TT[:, 1].reshape(-1, 1)

zauv = np.array(zauv).reshape(-1, 1)
dauv = np.array(dauv).reshape(-1, 1)
sal_auv = np.array(sal_auv).reshape(-1, 1)
temp_auv = np.array(temp_auv).reshape(-1, 1)
time_mission = np.array(time_mission).reshape(-1, 1)

datasheet = np.hstack((time_mission, lat_auv, lon_auv, xauv, yauv, zauv, dauv, sal_auv, temp_auv))
# np.savetxt(os.getcwd() + "data.txt", datasheet, delimiter = ",")

starting_index = 1
# starting_index = 700
origin = [lat4, lon4]
distance = 1000
depth_obs = [0.5, 1.0, 1.5, 2.0, 2.5]  # planned depth to be observed
box = BBox(lat4, lon4, distance, 60)
N1 = 25 # number of grid points along north direction
N2 = 25 # number of grid points along east direction
N3 = 5 # number of layers in the depth dimension
N = N1 * N2 * N3 # total number of grid points

XLIM = [0, distance]
YLIM = [0, distance]
ZLIM = [0.5, 2.5]
x = np.linspace(XLIM[0], XLIM[1], N1)
y = np.linspace(YLIM[0], YLIM[1], N2)
z = np.array(depth_obs)
grid = []
for k in z:
    for i in x:
        for j in y:
            grid.append([i, j, k])
grid = np.array(grid)
xv = grid[:, 0].reshape(-1, 1)
yv = grid[:, 1].reshape(-1, 1)
zv = grid[:, 2].reshape(-1, 1)
dx = x[1] - x[0]
coordinates= getCoordinates(box, N1, N2, dx, 60)

mu_prior = np.loadtxt("mu_prior_sal.txt").reshape(-1, 1)
beta0 = np.loadtxt("beta0.txt", delimiter = ",")
beta1 = np.loadtxt("beta1.txt", delimiter = ",")

sigma = np.sqrt(4)
tau = np.sqrt(.3)
Threshold = 28
eta = 4.5 / 400
ksi = 1000 / 24 / 0.5

grid = []
for k in z:
    for i in x:
        for j in y:
            grid.append([i, j, k])

grid = np.array(grid)
H_grid = compute_H(grid, grid, ksi)
Sigma_prior = Matern_cov(sigma, eta, H_grid)

def myround(x, base=1.):
    return base * np.round(x/base)

dauv_new = myround(dauv, base = .5)
ind = (dauv_new > 0).squeeze()
# ind = range(starting_index, len(dauv_new))
xauv_new = xauv_new[ind].reshape(-1, 1)
yauv_new = yauv_new[ind].reshape(-1, 1)
dauv_new = dauv_new[ind].reshape(-1, 1)
Xauv_new = myround(xauv_new, base = dx)
Yauv_new = myround(yauv_new, base = dx)
sal_auv = sal_auv[ind].reshape(-1, 1)
coordinates_auv = np.hstack((lat_auv[ind], lon_auv[ind]))


SINMOD_path = SINMOD_datapath + 'samples_2020.05.01.nc'
SINMOD = netCDF4.Dataset(SINMOD_path)

# #%%
# for i in range(5):
#     sal_sinmod, temp_sinmod = GetSINMODFromCoordinates(SINMOD, coordinates, depth_obs[i])
#     plt.imshow((beta0[i, 0] + beta1[i, 1] * sal_sinmod).reshape(N1, N2), vmin = 15, vmax = 30)
#     plt.colorbar()
#     plt.show()

mu_cond = mu_prior
Sigma_cond = Sigma_prior
def rotate_z(x, y, z, theta):
    w = x+1j*y
    return np.real(np.exp(1j*theta)*w), np.imag(np.exp(1j*theta)*w), z

rdbu = cm.get_cmap('RdBu', 256)
newcolors = rdbu(np.linspace(0, 1, 50))
newcmp = ListedColormap(newcolors)

x_eye = -1.25
y_eye = -1.25
z_eye = .5
# for i in range(10):
for i in [len(xauv_new)]:
# for i in range(len(xauv_new)):
    mu_cond = mu_prior
    Sigma_cond = Sigma_prior
    print(i)
    XAUV = xauv_new[:i + 1]
    YAUV = yauv_new[:i + 1]
    DAUV = dauv_new[:i + 1]
    COORDINATES = coordinates_auv[:i + 1]

    sal_sinmod, temp_sinmod = GetSINMODFromCoordinates(SINMOD, COORDINATES, DAUV)

    mu_sal_est = []
    for j in range(len(sal_sinmod)):
        k = np.where(depth_obs == dauv_new[j])[0][0]
        mu_sal_est.append(beta0[k, 0] + beta1[k, 0] * sal_sinmod[j, 0])
    mu_sal_est = np.array(mu_sal_est).reshape(-1, 1)

    obs = np.hstack((XAUV, YAUV, DAUV))
    H_obs = compute_H(obs, obs, ksi)
    Sigma_obs = Matern_cov(sigma, eta, H_obs) + tau ** 2 * np.identity(H_obs.shape[0])

    H_grid_obs = compute_H(grid, obs, ksi)
    Sigma_grid_obs = Matern_cov(sigma, eta, H_grid_obs)

    mu_cond = mu_cond + Sigma_grid_obs @ np.linalg.solve(Sigma_obs, (sal_auv[:i + 1] - mu_sal_est))
    Sigma_cond = Sigma_cond - Sigma_grid_obs @ np.linalg.solve(Sigma_obs, Sigma_grid_obs.T)
    perr = np.diag(Sigma_cond).reshape(-1, 1)
    EP = EP_1D(mu_cond, Sigma_cond, Threshold)
    ##
    fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scene'}]])

    for j in range(len(np.unique(zv))):
        ind = (zv == np.unique(zv)[j])
        fig.add_trace(
            go.Isosurface(x=xv[ind], y=yv[ind], z=-zv[ind],
                          value=mu_cond[ind], colorscale="coloraxis"),
                          # value=EP[ind], coloraxis = "coloraxis"),
                          # value=mu_cond[ind], coloraxis="coloraxis"),
            row=1, col=1
        )

    fig.add_trace(
        go.Scatter3d(
            x=XAUV.squeeze(), y=YAUV.squeeze(), z=np.array(-DAUV.squeeze()),
            marker=dict(
                size=4,
                color="black",
                showscale=False
            ),
            line=dict(
                color='darkblue',
                width=2
            )
        ),
        row=1, col=1
    )
    fig.update_coloraxes(colorscale = "gnbu")
    # fig.update_coloraxes(colorscale = newcmp)
    xe, ye, ze = rotate_z(x_eye, y_eye, z_eye, -i * .005)
    fig.update_layout(
        scene={
            'aspectmode': 'manual',
            'aspectratio': dict(x=1, y=1, z=.5),
        },
        showlegend=False,
        scene_camera_eye=dict(x=xe, y=ye, z=ze),
        title="AUV explores the field"
    )
    plotly.offline.plot(fig)
    # fig.write_image(figpath + "3D/4/T_{:04d}.png".format(i), width=1980, height=1080)


#%%
import plotly.graph_objects as go
import plotly
plotly.io.orca.config.executable = '/Users/yaoling/anaconda3/bin/orca/'
plotly.io.orca.config.save()
from plotly.subplots import make_subplots

# fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'scene'}, {'type': 'scene'}]],
#                     subplot_titles=("SINMOD data", "Prior"))
fig = make_subplots(specs=[[{'type': 'scene'}]])

for i in range(len(np.unique(zv))):
    # sal_sinmod, temp_sinmod = GetSINMODFromCoordinates(SINMOD, coordinates, depth_obs[i])
    ind = (zv == np.unique(zv)[i])
    fig.add_trace(
        go.Isosurface(x=xv[ind], y=yv[ind], z=-zv[ind], value=mu_prior[ind], coloraxis='coloraxis'),
        # go.Isosurface(x=xv[ind], y=yv[ind], z=-zv[ind], value=sal_sinmod, coloraxis='coloraxis'),
        row=1, col=1
    )
#%%
x_eye = 1.25
y_eye = 1.25
z_eye = 1.25

fig.update_layout(
    scene={
        'aspectmode': 'manual',
        'aspectratio': dict(x=1, y=1, z=.5),
    },
    showlegend=False,
    # coloraxis=dict(colorscale="viridis"),
    scene_camera_eye = dict(x=x_eye, y=y_eye, z=z_eye),
    title="Salinity data from SINMOD"
)
fig.update_coloraxes(colorscale="jet")
print("finished")

# frames=[]
# for t in np.arange(0, 6.26, 0.1):
#     xe, ye, ze = rotate_z(x_eye, y_eye, z_eye, -t)
#     frames.append(go.Frame(layout=dict(scene_camera_eye=dict(x=xe, y=ye, z=ze))))
# fig.frames=frames

plotly.offline.plot(fig)
# fig.write_image(figpath + "3D/T_{:03d}.png".format(j), width=1980, height=1080)
# fig.write_image(figpath + "True.png", width=1980, height=1080)

# data3d = np.hstack((xv, yv, -zv, mu_cond))
# data3d = np.hstack((xv, yv, -zv, mu_cond))
# # # data3d = np.hstack((xv, yv, -zv, EP_prior))
# # # data3d = np.hstack((xv, yv, -zv, perr))
# # # data3d = np.hstack((XS, YS, ZS, Sal))
# plot3d = plot3D(data3d)
# # # plot3d.draw3DVolume(surface_count = 100)
# plot3d.draw3DSurface()



#%%
obs = np.hstack((xauv_new, yauv_new, dauv_new))
H_obs = compute_H(obs, obs, ksi)
Sigma_obs = Matern_cov(sigma, eta, H_obs) + tau ** 2 * np.identity(H_obs.shape[0])

H_grid_obs = compute_H(grid, obs, ksi)
Sigma_grid_obs = Matern_cov(sigma, eta, H_grid_obs)

SINMOD_path = SINMOD_datapath + 'samples_2020.05.01.nc'
SINMOD = netCDF4.Dataset(SINMOD_path)

sal_sinmod, temp_sinmod = GetSINMODFromCoordinates(SINMOD, coordinates_auv, dauv_new)


mu_sal_est = []
for i in range(len(sal_sinmod)):
    k = np.where(depth_obs == dauv_new[i])[0][0]
    mu_sal_est.append(beta0[k, 0] + beta1[k, 0] * sal_sinmod[i, 0])

mu_sal_est = np.array(mu_sal_est).reshape(-1, 1)

# mu_cond = mu_prior + Sigma_grid_obs @ np.linalg.solve(Sigma_obs, (sal_auv - mu_sal_est))
# Sigma_cond = Sigma_prior - Sigma_grid_obs @ np.linalg.solve(Sigma_obs, Sigma_grid_obs.T)
# perr = np.diag(Sigma_cond).reshape(-1, 1)
EP_prior = EP_1D(mu_prior, Sigma_prior, Threshold)

mu_cond = mu_prior
Sigma_cond = Sigma_prior
noise = tau ** 2
R = np.diagflat(noise)

loc = find_starting_loc(EP_prior, N1, N2, N3)
xstart, ystart, zstart = loc
xnow, ynow, znow = xstart, ystart, zstart
xpre, ypre, zpre = xnow, ynow, znow
F = np.zeros([1, N])
ind_start = ravel_index(loc, N1, N2, N3)
F[0, ind_start] = True

xloc = xnow * dx
yloc = ynow * dx
dloc = depth_obs[znow]
ind_common = np.where((Xauv_new == xloc) & (Yauv_new == yloc))[0]
sal_sampled = np.mean(sal_auv[ind_common])  # average the common samples
mu_cond, Sigma_cond = GPupd(mu_cond, Sigma_cond, R, F, sal_sampled)


#%%

N_steps = 10

for j in range(N_steps):
    xcand, ycand, zcand = find_candidates_loc(xnow, ynow, znow, N1, N2, N3)

    t1 = time.time()
    xnext, ynext, znext = find_next_EIBV_1D(xcand, ycand, zcand,
                                            xnow, ynow, znow,
                                            xpre, ypre, zpre,
                                            N1, N2, N3, Sigma_cond,
                                            mu_cond, tau, Threshold)
    t2 = time.time()
    print("It takes {:.2f} seconds to compute the next waypoint".format(t2 - t1))
    print("next is ", xnext, ynext, znext)

    ind_next = ravel_index([xnext, ynext, znext], N1, N2, N3)
    F = np.zeros([1, N])
    F[0, ind_next] = True

    xloc = xnext * dx
    yloc = ynext * dx
    dloc = depth_obs[znext]
    ind_common = np.where((Xauv_new == xloc) & (Yauv_new == yloc))[0]
    sal_sampled = np.mean(sal_auv[ind_common])  # average the common samples
    mu_cond, Sigma_cond = GPupd(mu_cond, Sigma_cond, R, F, sal_sampled)

    xpre, ypre, zpre = xnow, ynow, znow
    xnow, ynow, znow = xnext, ynext, znext



#%%




import plotly.graph_objects as go
import plotly
plotly.io.orca.config.executable = '/Users/yaoling/anaconda3/bin/orca/'
plotly.io.orca.config.save()
from plotly.subplots import make_subplots

COLORSCALE = 'jet'

class plot3D():
    def __init__(self, data3D):
        self.X = data3D[:, 0].flatten()
        self.Y = data3D[:, 1].flatten()
        self.Z = data3D[:, 2].flatten()
        self.val = data3D[:, -1].flatten()


    # fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'scene'}, {'type': 'scene'}]],
    #                     subplot_titles=("Excursion Probability", "Directional vector"))
    def draw3DVolume(self, isomin = None, isomax = None, opacity = None,
                     surface_count = None, colorbar = dict(len = .5)):
        fig = make_subplots(rows = 1, cols = 1, specs = [[{'type': 'scene'}]],
                            subplot_titles=("Salinity"))

        fig.add_trace(
            go.Volume(
                x=self.X, y=self.Y, z=self.Z,
                value=self.val,
                isomin=isomin,
                isomax=isomax,
                opacity=opacity,
                surface_count=surface_count,
                colorbar=colorbar,
            ),
            row=1, col=1
        )
        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=-1.25, y=-1.25, z=1.25)
        )
        fig.update_layout(
            scene={
                'aspectmode': 'manual',
                'aspectratio': dict(x=1, y=1, z=.5),
                'xaxis': {'range': [self.X.min(), self.X.max() + 90], 'rangemode': 'tozero', 'tickmode': "linear", 'tick0': 0, 'dtick': 100},
                'yaxis': {'range': [self.Y.min(), self.Y.max() + 90], 'rangemode': 'tozero', 'tickmode': "linear", 'tick0': 0, 'dtick': 100},
                'zaxis': {'range': [self.Z.min() - 1, self.Z.max()], 'rangemode': 'tozero', 'tickmode': "linear", 'tick0': 0, 'dtick': 0.5},
            },
            coloraxis=dict(colorscale=COLORSCALE),
            showlegend=False,
            scene_camera=camera
        )
        fig.update_coloraxes(colorscale=COLORSCALE)
        plotly.offline.plot(fig)

    def draw3DSurface(self):
        fig = make_subplots(rows=1, cols=1, specs=[[{'is_3d': True}]],
                            subplot_titles=("Salinity"))
        #
        for i in range(len(np.unique(self.Z))):
            ind = (self.Z == np.unique(self.Z)[i])
            fig.add_trace(
                go.Isosurface(x=self.X[ind], y=self.Y[ind], z=self.Z[ind], value=self.val[ind], coloraxis='coloraxis'),
                row=1, col=1
            )

        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=-4, y=-4, z=3.5)
        )
        fig.update_layout(
            scene={
                'aspectmode': 'manual',
                'aspectratio': dict(x=1, y=1, z=1),
                # 'xaxis': {'range': [self.X.min(), self.X.max() + 90], 'rangemode': 'tozero', 'tickmode': "linear", 'tick0': 0, 'dtick': 100},
                # 'yaxis': {'range': [self.Y.min(), self.Y.max() + 90], 'rangemode': 'tozero', 'tickmode': "linear", 'tick0': 0, 'dtick': 100},
                # 'zaxis': {'range': [self.Z.min() - 1, self.Z.max()], 'rangemode': 'tozero', 'tickmode': "linear", 'tick0': 0, 'dtick': 0.5},
            },
            coloraxis=dict(colorscale=COLORSCALE),
            showlegend=False,
            scene_camera=camera
        )
        fig.update_coloraxes(colorscale=COLORSCALE)
        plotly.offline.plot(fig)
        # fig.write_image(figpath + "sal.png".format(j), width=1980, height=1080)



data3d = np.hstack((xv, yv, -zv, mu_prior))
# data3d = np.hstack((xv, yv, -zv, mu_cond))
# data3d = np.hstack((xv, yv, -zv, EP_prior))
# data3d = np.hstack((xv, yv, -zv, perr))
# data3d = np.hstack((XS, YS, ZS, Sal))
plot3d = plot3D(data3d)
# plot3d.draw3DVolume(surface_count = 100)
plot3d.draw3DSurface()
    # fig.add_trace(
    #     go.Scatter3d(
    #         x=path_x, y=path_y, z=path_z,
    #         marker=dict(
    #             size=4,
    #             color=path_z,
    #             colorscale='Viridis',
    #             showscale=False
    #         ),
    #         line=dict(
    #             color='darkblue',
    #             width=2
    #         )
    #     ),
    #     row=1, col=1
    # )
    #
    # u = [xv[ind_now][0] - xv[ind_pre][0]]
    # v = [yv[ind_now][0] - yv[ind_pre][0]]
    # w = [zv[ind_now][0] - zv[ind_pre][0]]
    #
    # fig.add_trace(
    #     go.Cone(x=xv[ind_pre], y=yv[ind_pre], z=zv[ind_pre], u=u, v=v, w=w, showscale=False),
    #     row=1, col=2
    # )
    # fig.add_trace(
    #     go.Scatter3d(
    #         x=xv[ID].squeeze(), y=yv[ID].squeeze(), z=zv[ID].squeeze(),
    #         mode='markers',
    #         marker=dict(
    #             size=10,
    #             color=data_EIBV * 1000,
    #             colorscale='ylgnbu',
    #             colorbar=dict(len=0.5, x=1),
    #         ),
    #
    #     ),
    #     row=1, col=2
    # )
    # fig.update_layout(
    #     scene={
    #         # ...
    #         'aspectmode': 'cube',
    #         'xaxis': {'range': [0, 1], 'rangemode': 'tozero', 'tickmode': "linear", 'tick0': 0, 'dtick': 0.5},
    #         'yaxis': {'range': [0, 1], 'rangemode': 'tozero', 'tickmode': "linear", 'tick0': 0, 'dtick': 0.5},
    #         'zaxis': {'range': [0, 1], 'rangemode': 'tozero', 'tickmode': "linear", 'tick0': 0, 'dtick': 0.5},
    #
    #     },
    #     scene2={
    #         'aspectmode': 'cube',
    #         'xaxis': {'range': [0, 1], 'rangemode': 'tozero', 'tickmode': "linear", 'tick0': 0, 'dtick': 0.5},
    #         'yaxis': {'range': [0, 1], 'rangemode': 'tozero', 'tickmode': "linear", 'tick0': 0, 'dtick': 0.5},
    #         'zaxis': {'range': [0, 1], 'rangemode': 'tozero', 'tickmode': "linear", 'tick0': 0, 'dtick': 0.5},
    #     },
    #     margin={'autoexpand': False},
    #     autosize=False
    #     # scene=dict(
    #     #     xaxis=dict(nticks=4, range=[0, 1], ),
    #     #     yaxis=dict(nticks=4, range=[0, 1], ),
    #     #     zaxis=dict(nticks=4, range=[0, 1], ), )
    # )
    # fig.write_image(figpath + "3D/T_{:03d}.png".format(j), width=1980, height=1080)

# initial_zoom = 12
# # apikey = 'AIzaSyDkWNSq_EKnrV9qP6thJe5Y8a5kVLKEjUI'
# apikey = 'AIzaSyAZ_VZXoJULTFQ9KSPg1ClzHEFjyPbJUro'
# gmap = CustomGoogleMapPlotter(lat[0, 0], lon[0, 0], initial_zoom, map_type='satellite', apikey = apikey)
# gmap.color_scatter(lat[:, 0].tolist(), lon[:, 0].tolist(), sal.squeeze(), size = 10, colormap='hsv')
# gmap.draw(figpath + "sal.html")
#
# gmap.color_scatter(lat[:, 0].tolist(), lon[:, 0].tolist(), temp.squeeze(), size = 10, colormap='hsv')
# gmap.draw(figpath + "temp.html")
