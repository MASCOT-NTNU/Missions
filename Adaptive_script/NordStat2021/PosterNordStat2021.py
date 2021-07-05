import os

from usr_func import *

fp='samples_2020.05.01.nc'
nc = netCDF4.Dataset(fp)
SINMOD = nc

salinity = np.mean(SINMOD['salinity'][:, :, :, :], axis=0)
temperature = np.mean(SINMOD['temperature'][:, :, :, :], axis=0) - 273.15
depth_sinmod = np.array(SINMOD['zc'])
lat_sinmod = np.array(SINMOD['gridLats'][:, :]).reshape(-1, 1)
lon_sinmod = np.array(SINMOD['gridLons'][:, :]).reshape(-1, 1)
# sal_sinmod = np.zeros([coordinates.shape[0], 1])
# temp_sinmod = np.zeros([coordinates.shape[0], 1])
zc = np.array(SINMOD['zc'])

#%%
figpath = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Conferences/2021/NORDSTAT2021/fig/"

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 32})
plt.rcParams.update({'font.style': 'oblique'})

plt.figure(figsize = (20, 20))
# plt.scatter(lon_sinmod, lat_sinmod, c = salinity[0, :, :].reshape(-1, 1), cmap = "cividis")
# plt.scatter(lon_sinmod, lat_sinmod, c = salinity[0, :, :].reshape(-1, 1), cmap = "YlGnBu")
# plt.scatter(lon_sinmod, lat_sinmod, c = temperature[0, :, :].reshape(-1, 1), vmin = 0, vmax = 10, cmap = "RdBu")
plt.scatter(lon_sinmod, lat_sinmod, c = temperature[0, :, :].reshape(-1, 1), vmin = 0, vmax = 10, cmap = "bwr")
f = 1.0/np.cos(60*np.pi/180)
ax = plt.gca()
ax.set_aspect(f)
plt.title("Temperature at 0.5m depth")
plt.xlabel("Lon [deg]")
plt.ylabel("Lat [deg]")
plt.colorbar(fraction=0.045, pad=0.04)
plt.savefig(figpath + "temp_05.pdf")
plt.show()
a = salinity[0, :, :]


#%%

# Import libraries
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

# Creating dataset
x = np.arange(3)
y = np.arange(3)
z = np.arange(-2, 1)
xx, yy, zz = np.meshgrid(x, y, z)
c = np.ones_like(xx, dtype = np.float)
s = np.ones_like(xx)*25
s[0,0,0] = 100
# Creating figure
# fig = plt.figure(figsize=(10, 7))
ax = plt.axes(projection="3d")

# Creating plot
ax.scatter3D(xx, yy, zz, c=zz, cmap = "brg")
plt.title("simple 3D scatter plot")
ax.set_xlabel("East")
ax.set_ylabel('North')
ax.set_zlabel("Depth [m]")

for line in ax.xaxis.get_ticklines():
    line.set_visible(False)
for line in ax.yaxis.get_ticklines():
    line.set_visible(False)
for line in ax.zaxis.get_ticklines():
    line.set_visible(False)

# plt.tick_params(
#     axis='x',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom=False,      # ticks along the bottom edge are off
#     top=False,         # ticks along the top edge are off
#     labelbottom=False) # labels along the bottom edge are off
#
# plt.tick_params(
#     axis='y',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom=False,      # ticks along the bottom edge are off
#     top=False,         # ticks along the top edge are off
#     labelbottom=False) # labels along the bottom edge are off
#
# plt.tick_params(
#     axis='z',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom=False,      # ticks along the bottom edge are off
#     top=False,         # ticks along the top edge are off
#     labelbottom=False) # labels along the bottom edge are off

ax.view_init(-10, 80)
# show plot
ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
plt.show()

#%%
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.gca(projection='3d')
# ax.set_aspect("equal")

count = 0
x = range(0,2+1)
X,Y,Z = np.meshgrid(x,x,x)

c = np.zeros_like(X, dtype=np.float)
c[((X+Y+Z+3*count)%2) == 0] = 0.5
c[count,count,count] = 1

s = np.ones_like(X)*25
s[count,count,count] = 100
ax.scatter(X,Y,Z, c=c,s=s, cmap="brg")

plt.show()

#%%
fig = plt.figure()
plt.scatter(lon_sinmod, lat_sinmod, c = salinity[0, :, :])
plt.xlabel("East")
plt.ylabel("North")
plt.colorbar(fraction=0.045, pad=0.04)
# plt.gca().invert_yaxis()
plt.show()
# plt.savefig()

a_sal = np.array(salinity[0, :, :]).reshape(-1, 1)
# for i in range(coordinates.shape[0]):
#     lat, lon = coordinates[i]
#     ind_depth = np.where(np.array(depth_sinmod) == depth)[0][0]
#     idx = np.argmin((lat_sinmod - lat) ** 2 + (lon_sinmod - lon) ** 2)
#     sal_sinmod[i] = salinity[ind_depth].reshape(-1, 1)[idx]
#     temp_sinmod[i] = temperature[ind_depth].reshape(-1, 1)[idx]

# apikey = 'AIzaSyAZ_VZXoJULTFQ9KSPg1ClzHEFjyPbJUro' # (your API key here)
apikey = 'AIzaSyDkWNSq_EKnrV9qP6thJe5Y8a5kVLKEjUI'
gmap = gmplot.GoogleMapPlotter(lat_sinmod[0], lon_sinmod[1], 14, apikey=apikey)

# Highlight some attractions:
attractions_lats = np.array(lat_sinmod)
attractions_lngs = np.array(lon_sinmod)
# gmap.scatter(attractions_lats, attractions_lngs, color='#3B0B39', size=4, marker=False)
gmap.scatter(attractions_lats, attractions_lngs, size=1, marker=False)

# Mark a hidden gem:
# for i in range(box.shape[0]):
#     gmap.marker(box[i, 0], box[i, 1], color='cornflowerblue')

# Draw the map:
gmap.draw(os.getcwd() + '/NordStat2021/map.html')

#%%
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from gmplot import GoogleMapPlotter
from random import random


class CustomGoogleMapPlotter(GoogleMapPlotter):
    def __init__(self, center_lat, center_lng, zoom, apikey='AIzaSyAZ_VZXoJULTFQ9KSPg1ClzHEFjyPbJUro',
                 map_type='satellite'):
        if apikey == '':
            try:
                with open('apikey.txt', 'r') as apifile:
                    apikey = apifile.readline()
            except FileNotFoundError:
                pass
        super().__init__(center_lat, center_lng, zoom, apikey)

        self.map_type = map_type
        assert(self.map_type in ['roadmap', 'satellite', 'hybrid', 'terrain'])

    def write_map(self,  f):
        f.write('\t\tvar centerlatlng = new google.maps.LatLng(%f, %f);\n' %
                (self.center[0], self.center[1]))
        f.write('\t\tvar myOptions = {\n')
        f.write('\t\t\tzoom: %d,\n' % (self.zoom))
        f.write('\t\t\tcenter: centerlatlng,\n')

        # Change this line to allow different map types
        f.write('\t\t\tmapTypeId: \'{}\'\n'.format(self.map_type))

        f.write('\t\t};\n')
        f.write(
            '\t\tvar map = new google.maps.Map(document.getElementById("map_canvas"), myOptions);\n')
        f.write('\n')

    def color_scatter(self, lats, lngs, values=None, colormap='coolwarm',
                      size=None, marker=False, s=None, **kwargs):
        def rgb2hex(rgb):
            """ Convert RGBA or RGB to #RRGGBB """
            rgb = list(rgb[0:3]) # remove alpha if present
            rgb = [int(c * 255) for c in rgb]
            hexcolor = '#%02x%02x%02x' % tuple(rgb)
            return hexcolor

        if values is None:
            colors = [None for _ in lats]
        else:
            cmap = plt.get_cmap(colormap)
            norm = Normalize(vmin=min(values), vmax=max(values))
            scalar_map = ScalarMappable(norm=norm, cmap=cmap)
            colors = [rgb2hex(scalar_map.to_rgba(value)) for value in values]
        for lat, lon, c in zip(lats, lngs, colors):
            self.scatter(lats=[lat], lngs=[lon], c=c, size=size, marker=marker,
                         s=s, **kwargs)


initial_zoom = 12
# num_pts = 40
#
# lats = [37.428]
# lons = [-122.145]
# values = [random() * 20]
# for pt in range(num_pts):
#     lats.append(lats[-1] + (random() - 0.5)/100)
#     lons.append(lons[-1] + random()/100)
#     values.append(values[-1] + random())
gmap = CustomGoogleMapPlotter(lat_sinmod[0], lon_sinmod[0], initial_zoom, map_type='satellite')
gmap.color_scatter(lat_sinmod, lon_sinmod, a_sal.squeeze(), colormap='coolwarm')

# gmap.draw("mymap.html")
gmap.draw(os.getcwd() + "/NordStat2021/mymap.html")

#%%
from usr_func import *
from support_func_nordstat import *

figpath = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Conferences/2021/NORDSTAT2021/fig/EIBV/"

# Setup the grid
n1 = 10 # number of grid points along east direction
n2 = 10 # number of grid points along north direction
n3 = 10 # number of layers in the depth dimension
n = n1 * n2 * n3 # total number of grid points
N = n
x = np.linspace(0, 1, n1)
y = np.linspace(0, 1, n2)
z = np.linspace(0, 1, n3)
xx, yy, zz = np.meshgrid(x, y, z)

xv = xx.reshape(-1, 1) # sites1v is the vectorised version
yv = yy.reshape(-1, 1)
zv = zz.reshape(-1, 1)
grid = np.hstack((xv, yv, zv))
t = scdist.cdist(grid, grid)
# Simulate the initial random field
sigma = 0.5  # scaling coef in matern kernel
eta = 5.5 # coef in matern kernel
tau = .05 # iid noise
S_thres = 0.5
# only one parameter is considered
beta = [[1], [0.133], [0.15], [.15]] # [intercept, trend along east, trend along north, along depth]

Sigma = Matern_cov(sigma, eta, t)  # matern covariance

L = np.linalg.cholesky(Sigma)  # lower triangle covariance matrix
N = L.shape[0]
z = np.dot(L, np.random.randn(n).reshape(-1, 1))

H = np.hstack((np.ones([n, 1]), xv, yv, zv)) # different notation for the project
mu_prior = mu(H, beta).reshape(n, 1)
mu_real = xv ** 2 + yv ** 2 + zv ** 2  # add covariance structure
# mu_real = (xv - 0.1) ** 2 + (yv - 0.1) ** 2 + (zv - 0.1) ** 2  # add covariance structure
# mu_real = (1 - xv) ** 2 + (1 - yv) ** 2 + (1 - zv) ** 2  # add covariance structure
# S_thres = np.mean(mu_prior)

# Section II: compute its corresponding excursion probabilities
def EP_1D(mu, Sig, Thres):
    '''
    :param mu:
    :param Sig:
    :param T_thres:
    :param S_thres:
    :return:
    '''
    n = mu.shape[0]
    ES_Prob = np.zeros([n, 1])
    for i in range(n):
        ES_Prob[i] = norm.cdf(Thres, mu[i], Sig[i, i])
    return ES_Prob

ES_Prob_prior = EP_1D(mu_prior, Sigma, S_thres)
ES_Prob_m_prior = np.array(ES_Prob_prior).reshape(n2, n1, n3) # ES_Prob_m is the reshaped matrix form of ES_Prob
ES_Prob_true = EP_1D(mu_real, Sigma, S_thres)
ES_Prob_m_true = np.array(ES_Prob_true).reshape(n2, n1, n3)

X = xv
Y = yv
Z = zv

import plotly.graph_objects as go
import plotly

plotly.io.orca.config.executable = '/Users/yaoling/anaconda3/bin/orca/'
plotly.io.orca.config.save()
from plotly.subplots import make_subplots

fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'scene'}, {'type': 'scene'}]],
                    subplot_titles=("True field", "Excursion probability"))

fig.add_trace(
    go.Volume(
        x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
        value=mu_real.flatten(),
        isomin=0,
        isomax=0.5,
        opacity=0.4,
        surface_count=100,
        colorbar=dict(len=0.5, x=0.45),
    ),
    row=1, col=1
)

fig.add_trace(
    go.Volume(
        x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
        value=ES_Prob_true.flatten(),
        isomin=0.5,
        isomax=1,
        opacity=0.4,
        surface_count=100,
        colorbar=dict(len=0.5, x=1),
    ),
    row=1, col=2
)

# fig.write_image(figpath + "3D/T_{:03d}.png".format(j), width=1980, height=1080)
# fig.write_image(figpath + "True.png", width=1980, height=1080)
plotly.offline.plot(fig)
#%%
ES_true = compute_ES(mu_real, S_thres)
ES_true_m = np.array(ES_true).reshape(n2, n1, n3)
ES_prior = compute_ES(mu_prior, S_thres)
ES_prior_m = np.array(ES_prior).reshape(n2, n1, n3)


N_steps = 80
row_start, col_start, dep_start = find_starting_loc(ES_Prob_prior, n1, n2, n3)
row_start = n1 - 1
col_start = n2 - 1
dep_start = n3 - 1
row_now, col_now, dep_now = row_start, col_start, dep_start
row_pre, col_pre, dep_pre = row_now, col_now, dep_now

print("The starting point is ")
print(row_now, col_now, dep_now)

mu_cond = mu_prior
Sigma_cond = Sigma
noise = tau ** 2
R = np.diagflat(noise)

path_row = []
path_col = [] # only used for plotting
ibv = []
mse = []

mse.append(np.mean((mu_cond - mu_real) ** 2))
ibv.append(np.sum(ES_Prob_prior * (1 - ES_Prob_prior)))


path_x = []
path_y = []
path_z = []
ind_pre = np.ravel_multi_index((row_pre, col_pre, dep_pre), (n2, n1, n3))
ind_now = np.ravel_multi_index((row_now, col_now, dep_now), (n2, n1, n3))
path_x.append(xv[ind_now][0])
path_y.append(yv[ind_now][0])
path_z.append(zv[ind_now][0])


#%
import plotly.graph_objects as go
import plotly

plotly.io.orca.config.executable = '/Users/yaoling/anaconda3/bin/orca/'
plotly.io.orca.config.save()
from plotly.subplots import make_subplots



for i in range(n3):
    path_row.append([])
    path_col.append([])

for i in range(n3):
    if i == dep_now:
        path_row[i].append(row_now)
        path_col[i].append(col_now)
    else:
        path_row[i].append(np.nan)
        path_col[i].append(np.nan)
X = xv
Y = yv
Z = zv

for j in range(N_steps):

    row_cand, col_cand, dep_cand = find_candidates_loc_rule(row_now, col_now, dep_now)

    row_next, col_next, dep_next, ID, data_EIBV = find_next_EIBV_rule(row_cand, col_cand, dep_cand,
                                                       row_now, col_now, dep_now,
                                                       row_pre, col_pre, dep_pre,
                                                       Sigma_cond, mu_cond, tau, S_thres)

    ind_next = np.ravel_multi_index((row_next, col_next, dep_next), (n2, n1, n3))
    F = np.zeros([1, N])
    F[0, ind_next] = True
    y_sampled = np.dot(F, mu_real) + tau * np.random.randn(1)

    mu_cond, Sigma_cond = GPupd(mu_cond, Sigma_cond, R, F, y_sampled)

    ES_Prob = EP_1D(mu_cond, Sigma_cond, S_thres)
    mse.append(np.mean((mu_cond - mu_real) ** 2))
    ibv.append(np.sum(ES_Prob * (1 - ES_Prob)))
    ES_Prob_m = ES_Prob.reshape(n2, n1, n3)

    for i in range(n3):
        if i == dep_now:
            path_row[i].append(row_now)
            path_col[i].append(col_now)
        else:
            path_row[i].append(np.nan)
            path_col[i].append(np.nan)

    row_pre, col_pre, dep_pre = row_now, col_now, dep_now
    row_now, col_now, dep_now = row_next, col_next, dep_next
    ind_pre = np.ravel_multi_index((row_pre, col_pre, dep_pre), (n2, n1, n3))
    ind_now = np.ravel_multi_index((row_now, col_now, dep_now), (n2, n1, n3))
    path_x.append(xv[ind_now][0])
    path_y.append(yv[ind_now][0])
    path_z.append(zv[ind_now][0])

    ##
    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'scene'}, {'type': 'scene'}]],
                        subplot_titles=("Excursion Probability", "Directional vector"))

    fig.add_trace(
        go.Volume(
            x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
            value=ES_Prob.flatten(),
            isomin=0.5,
            isomax=1,
            opacity=0.4,
            surface_count=100,
            colorbar=dict(len=0.5, x=0.45),
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter3d(
            x=path_x, y=path_y, z=path_z,
            marker=dict(
                size=4,
                color=path_z,
                colorscale='Viridis',
                showscale=False
            ),
            line=dict(
                color='darkblue',
                width=2
            )
        ),
        row=1, col=1
    )

    u = [xv[ind_now][0] - xv[ind_pre][0]]
    v = [yv[ind_now][0] - yv[ind_pre][0]]
    w = [zv[ind_now][0] - zv[ind_pre][0]]

    fig.add_trace(
        go.Cone(x=xv[ind_pre], y=yv[ind_pre], z=zv[ind_pre], u=u, v=v, w=w, showscale = False),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter3d(
            x=xv[ID].squeeze(), y=yv[ID].squeeze(), z=zv[ID].squeeze(),
            mode='markers',
            marker=dict(
                size=10,
                color=data_EIBV * 1000,
                colorscale='ylgnbu',
                colorbar=dict(len=0.5, x=1),
            ),

        ),
        row=1, col=2
    )
    fig.update_layout(
        scene={
            # ...
            'aspectmode': 'cube',
            'xaxis': {'range': [0, 1], 'rangemode': 'tozero', 'tickmode': "linear", 'tick0': 0, 'dtick': 0.5},
            'yaxis': {'range': [0, 1], 'rangemode': 'tozero', 'tickmode': "linear", 'tick0': 0, 'dtick': 0.5},
            'zaxis': {'range': [0, 1], 'rangemode': 'tozero', 'tickmode': "linear", 'tick0': 0, 'dtick': 0.5},

        },
        scene2 = {
            'aspectmode': 'cube',
            'xaxis': {'range': [0, 1], 'rangemode': 'tozero', 'tickmode': "linear", 'tick0': 0, 'dtick': 0.5},
            'yaxis': {'range': [0, 1], 'rangemode': 'tozero', 'tickmode': "linear", 'tick0': 0, 'dtick': 0.5},
            'zaxis': {'range': [0, 1], 'rangemode': 'tozero', 'tickmode': "linear", 'tick0': 0, 'dtick': 0.5},
        },
        margin={'autoexpand': False},
        autosize=False
        # scene=dict(
        #     xaxis=dict(nticks=4, range=[0, 1], ),
        #     yaxis=dict(nticks=4, range=[0, 1], ),
        #     zaxis=dict(nticks=4, range=[0, 1], ), )
    )
    fig.write_image(figpath + "3D/T_{:03d}.png".format(j), width=1980, height=1080)
    ##



    print(row_now, col_now, dep_now)
    print("Step NO.", str(j))

#%%
# plt.figure(figsize=(10, 5))
plt.figure(figsize=(7, 5))
p1, = plt.plot(np.arange(len(ibv)), ibv, 'k.-')
p2, = plt.plot(np.arange(len(ibv)), np.multiply(mse, 100), 'r.-')
plt.xlabel("Number of iterations")
plt.ylabel("IBV & 100MSE")
plt.xlim([0, len(ibv)])
ax = plt.gca()
ax.legend([p1, p2], ["IBV", "100MSE"])
plt.savefig(figpath + 'ibv.pdf')
plt.show()
#
# plt.subplot(121)
# plt.plot(ibv, 'ko-')
# plt.xlabel('Number of iterations')
# plt.ylabel("IBV")
# plt.subplot(122)
# plt.plot(mse, 'ko-')
# plt.xlabel('Number of iterations')
# plt.ylabel("MSE")
# plt.show()



# Generate nicely looking random 3D-field

X, Y, Z = xv, yv, zv
#%%
cbarlocs = [.5, 1.0]

fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'scene'}, {'type': 'scene'}]],
                    subplot_titles=("Excursion Probability", "Directional vector"))

fig.add_trace(
    go.Volume(
        x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
        value=ES_Prob.flatten(),
        isomin=0.5,
        isomax=1,
        opacity=0.4,
        surface_count=100,
        colorbar=dict(len=0.5, x=0.45),
    ),
    row=1, col=1
)

fig.add_trace(
    go.Scatter3d(
        x=path_x, y=path_y, z=path_z,
        marker=dict(
            size=4,
            colorscale='Viridis',
            showscale = False
        ),
        line=dict(
            color='darkblue',
            width=2
        ),
    ),
    row=1, col=1
)

u = [xv[ind_now][0] - xv[ind_pre][0]]
v = [yv[ind_now][0] - yv[ind_pre][0]]
w = [zv[ind_now][0] - zv[ind_pre][0]]


fig.add_trace(
    go.Scatter3d(
        x=xv[ID].squeeze(), y=yv[ID].squeeze(), z=zv[ID].squeeze(),
        mode='markers',
        marker=dict(
            size=10,
            color = data_EIBV*1000,
            colorscale='ylgnbu',
            colorbar = dict(len = 0.45, x = 1),
        ),
    ),
    row=1, col=2
)

fig.add_trace(
    go.Cone(x=xv[ind_pre], y=yv[ind_pre], z=zv[ind_pre], u=u, v=v, w=w, showscale = False),
    row=1, col=2,
)

fig.update_layout(
        scene={
            # ...
            'aspectmode': 'cube',
            'xaxis': {'range': [0, 1], 'rangemode': 'tozero', 'tickmode': "linear", 'tick0': 0, 'dtick': 0.5},
            'yaxis': {'range': [0, 1], 'rangemode': 'tozero', 'tickmode': "linear", 'tick0': 0, 'dtick': 0.5},
            'zaxis': {'range': [0, 1], 'rangemode': 'tozero', 'tickmode': "linear", 'tick0': 0, 'dtick': 0.5},

        },
        scene2 = {
            'aspectmode': 'cube',
            'xaxis': {'range': [0, 1], 'rangemode': 'tozero', 'tickmode': "linear", 'tick0': 0, 'dtick': 0.5},
            'yaxis': {'range': [0, 1], 'rangemode': 'tozero', 'tickmode': "linear", 'tick0': 0, 'dtick': 0.5},
            'zaxis': {'range': [0, 1], 'rangemode': 'tozero', 'tickmode': "linear", 'tick0': 0, 'dtick': 0.5},
        },
        margin={'autoexpand': False},
        # autosize=False
        # scene=dict(
        #     xaxis=dict(nticks=4, range=[0, 1], ),
        #     yaxis=dict(nticks=4, range=[0, 1], ),
        #     zaxis=dict(nticks=4, range=[0, 1], ), )
    )

# fig.write_image(figpath + "3D/T_{:03d}.png".format(j), width=1980, height=1080)
##
# fig.update_layout(scene_xaxis_showticklabels=False,
#                   scene_yaxis_showticklabels=False,
#                   scene_zaxis_showticklabels=False)

plotly.offline.plot(fig)
# plotly.io.write_image(fig, "test.png", format="png")
# plotly.offline.plot(fig)
# plotly.offline.plot(fig)
# fig.show()

#%%


import plotly.graph_objects as go


fig.update_layout(height=600, width=800, title_text="Side By Side Subplots")
plotly.offline.plot(fig)

#%%
import plotly.graph_objects as go
import pandas as pd
import numpy as np

rs = np.random.RandomState()
rs.seed(0)

def brownian_motion(T = 1, N = 100, mu = 0.1, sigma = 0.01, S0 = 20):
    dt = float(T)/N
    t = np.linspace(0, T, N)
    W = rs.standard_normal(size = N)
    W = np.cumsum(W)*np.sqrt(dt) # standard brownian motion
    X = (mu-0.5*sigma**2)*t + sigma*W
    S = S0*np.exp(X) # geometric brownian motion
    return S

dates = pd.date_range('2012-01-01', '2013-02-22')
T = (dates.max()-dates.min()).days / 365
N = dates.size
start_price = 100
y = brownian_motion(T, N, sigma=0.1, S0=start_price)
z = brownian_motion(T, N, sigma=0.1, S0=start_price)
print(y.shape)
print(z.shape)
print(dates.shape)

#%%
fig = go.Figure(data=go.Scatter3d(
    x=dates, y=y, z=z,
    marker=dict(
        size=4,
        color=z,
        colorscale='Viridis',
    ),
    line=dict(
        color='darkblue',
        width=2
    )
))

# fig.update_layout(
#     width=800,
#     height=700,
#     autosize=False,
#     scene=dict(
#         camera=dict(
#             up=dict(
#                 x=0,
#                 y=0,
#                 z=1
#             ),
#             eye=dict(
#                 x=0,
#                 y=1.0707,
#                 z=1,
#             )
#         ),
#         aspectratio = dict( x=1, y=1, z=0.7 ),
#         aspectmode = 'manual'
#     ),
# )

# fig.show()
plotly.offline.plot(fig)

#%%
plt.figure(figsize = (8, 5))
plt.plot(ibv, 'ko-')
plt.xlabel("Number of iterations")
plt.ylabel("Integrated Bernoulli Variances")
plt.show()

#%%
from usr_func import *
data = np.loadtxt("data.txt", delimiter=",")
timestamp = data[:, 0]
lat = rad2deg(data[:, 1])
lon = rad2deg(data[:, 2])
xauv = data[:, 3]
yauv = data[:, 4]
zauv = data[:, 5]
depth = data[:, 6]
sal = data[:, 7]
temp = data[:, 8]

lat_auv = lat + rad2deg(xauv * np.pi * 2.0 / circumference)
lon_auv = lon + rad2deg(yauv * np.pi * 2.0 / (circumference * np.cos(deg2rad(lat_auv))))

depth_obs = 0.5
err_bound = 0.1
depthl = depth_obs - err_bound
depthu = depth_obs + err_bound

ind_obs = (depthl <= depth) & (depth <= depthu)
lat_obs = lat_auv[ind_obs].reshape(-1, 1)
lon_obs = lon_auv[ind_obs].reshape(-1, 1)
sal_obs = sal[ind_obs].reshape(-1, 1)
temp_obs = temp[ind_obs].reshape(-1, 1)


#%%
h1 = sal_obs[0:150]
h2 = sal_obs[700:800]
hv = np.vstack((h1, h2))
plt.figure()
# plt.hist(hv, bins =50)
plt.plot(h2)
plt.show()

#%%
# plt.figure()
# plt.scatter(rad2deg(lon_auv[ind_obs]), rad2deg(lat_auv[ind_obs]), c = sal_obs)
# plt.colorbar()
# plt.show()

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from gmplot import GoogleMapPlotter

class CustomGoogleMapPlotter(GoogleMapPlotter):
    def __init__(self, center_lat, center_lng, zoom, apikey='AIzaSyAZ_VZXoJULTFQ9KSPg1ClzHEFjyPbJUro',
                 map_type='satellite'):
        if apikey == '':
            try:
                with open('apikey.txt', 'r') as apifile:
                    apikey = apifile.readline()
            except FileNotFoundError:
                pass
        print(apikey)
        GoogleMapPlotter(center_lat, center_lng, zoom, apikey=apikey)
        super().__init__(center_lat, center_lng, zoom, apikey = apikey)

        self.map_type = map_type
        assert(self.map_type in ['roadmap', 'satellite', 'hybrid', 'terrain'])

    def write_map(self,  f):
        f.write('\t\tvar centerlatlng = new google.maps.LatLng(%f, %f);\n' %
                (self.center[0], self.center[1]))
        f.write('\t\tvar myOptions = {\n')
        f.write('\t\t\tzoom: %d,\n' % (self.zoom))
        f.write('\t\t\tcenter: centerlatlng,\n')

        # Change this line to allow different map types
        f.write('\t\t\tmapTypeId: \'{}\'\n'.format(self.map_type))

        f.write('\t\t};\n')
        f.write(
            '\t\tvar map = new google.maps.Map(document.getElementById("map_canvas"), myOptions);\n')
        f.write('\n')

    def color_scatter(self, lats, lngs, values=None, colormap='coolwarm',
                      size=None, marker=False, s=None, **kwargs):
        def rgb2hex(rgb):
            """ Convert RGBA or RGB to #RRGGBB """
            rgb = list(rgb[0:3]) # remove alpha if present
            rgb = [int(c * 255) for c in rgb]
            hexcolor = '#%02x%02x%02x' % tuple(rgb)
            return hexcolor

        if values is None:
            colors = [None for _ in lats]
        else:
            cmap = plt.get_cmap(colormap)
            norm = Normalize(vmin=min(values), vmax=max(values))
            scalar_map = ScalarMappable(norm=norm, cmap=cmap)
            colors = [rgb2hex(scalar_map.to_rgba(value)) for value in values]
        for lat, lon, c in zip(lats, lngs, colors):
            self.scatter(lats=[lat], lngs=[lon], c=c, size=size, marker=marker,
                         s=s, **kwargs)

initial_zoom = 12
# apikey = 'AIzaSyDkWNSq_EKnrV9qP6thJe5Y8a5kVLKEjUI'
apikey = 'AIzaSyAZ_VZXoJULTFQ9KSPg1ClzHEFjyPbJUro'
gmap = CustomGoogleMapPlotter(lat_auv[ind_obs][0], lon_auv[ind_obs][0], initial_zoom, apikey = apikey)
# for i in range(len(sal_obs)):
gmap.color_scatter(lat_auv[ind_obs], lon_auv[ind_obs], sal_obs.squeeze(), size = 5, colormap='hsv')

# gmap = CustomGoogleMapPlotter(lat_sinmod[0], lon_sinmod[0], initial_zoom, map_type='satellite')

# gmap.draw("mymap.html")
gmap.draw(os.getcwd() + "/NordStat2021/mymap.html")

#%%
plt.figure()
plt.plot(salinity.reshape(-1, 1), depth_sinmod.reshape(-1, 1), 'r.')
plt.plot(sal, depth, 'k.')

plt.gca().invert_yaxis()
plt.show()


#%% Convert video to images
import cv2
video_loc = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Conferences/2021/NORDSTAT2021/Poster/Materials/"
vidcap = cv2.VideoCapture(video_loc + 'm2.mov')
def getFrame(sec):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vidcap.read()
    if hasFrames:
        cv2.imwrite(video_loc + "M2/IMG_{:03d}.jpg".format(count), image)     # save frame as JPG file
    return hasFrames
sec = 0
frameRate = 0.5 #//it will capture image in each 0.5 second
count=1
success = getFrame(sec)
while success:
    count = count + 1
    sec = sec + frameRate
    sec = round(sec, 2)
    success = getFrame(sec)

print("hello world")

#%%
import os
file_loc = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Conferences/2021/NORDSTAT2021/Poster/"
filename = file_loc + "beamer2.tex"
v1 = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Conferences/2021/NORDSTAT2021/Poster/Materials/M1/"
v2 = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Conferences/2021/NORDSTAT2021/Poster/Materials/M2/"
v3 = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Conferences/2021/NORDSTAT2021/fig/EIBV/3D/"

f1 = open(filename, "w+")
# f1.writelines("\\documentclass[14pt]{beamer}\n")
# f1.writelines("\\usetheme{Singapore}\n")
# f1.writelines("\\begin{document}\n\n")
print(len([name for name in os.listdir(v2) if name.endswith(".jpg")]))

V1 = []
V2 = []
V3 = []
for vv1 in os.listdir(v1):
    if vv1.endswith('.jpg'):
        # print(file_name)
        V1.append(vv1)

for vv2 in os.listdir(v2):
    if vv2.endswith('.jpg'):
        # print(file_name)
        V2.append(vv2)

for vv3 in os.listdir(v3):
    if vv3.endswith('.png'):
        # print(file_name)
        V3.append(vv3)

V1 = sorted(V1)
V2 = sorted(V2)
V3 = sorted(V3)

for i in range(len([name for name in os.listdir(v1) if name.endswith(".jpg")])):
    f1.writelines("\\begin{frame}\n")
    f1.writelines("\\begin{figure}\n")
    f1.writelines("\\centering\n")
    s = v1 + "IMG_{:03d}.jpg".format(i + 1)
    f1.writelines("\\includegraphics[width = \\textwidth]{\"%s\"}\n" % s)
    f1.writelines("\\end{figure}\n")
    f1.writelines("\\end{frame}\n\n")


for i in range(len([name for name in os.listdir(v3) if name.endswith(".png")])):
    f1.writelines("\\begin{frame}\n")
    f1.writelines("\\begin{figure}\n")
    f1.writelines("\\centering\n")
    s = v3 + "T_{:03d}.png".format(i)
    f1.writelines("\\includegraphics[width = \\textwidth]{\"%s\"}\n" % s)
    f1.writelines("\\end{figure}\n")
    f1.writelines("\\end{frame}\n\n")

for i in range(len([name for name in os.listdir(v2) if name.endswith(".jpg")])):
    f1.writelines("\\begin{frame}\n")
    f1.writelines("\\begin{figure}\n")
    f1.writelines("\\centering\n")
    s = v2 + "IMG_{:03d}.jpg".format(i + 1)
    f1.writelines("\\includegraphics[width = \\textwidth]{\"%s\"}\n" % s)
    f1.writelines("\\end{figure}\n")
    f1.writelines("\\end{frame}\n\n")


# f1.writelines("\\end{document}\n")
f1.close()

