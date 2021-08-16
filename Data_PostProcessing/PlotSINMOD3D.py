import netCDF4
import numpy as np
import plotly.graph_objects as go
import plotly
# plotly.io.orca.config.executable = '/Users/yaoling/anaconda3/bin/orca/'
# plotly.io.orca.config.save()
from plotly.subplots import make_subplots

server = False
if server == True:
    SINMOD_datapath = "/home/ahomea/y/yaoling/MASCOT/ES_3D/"
    figpath = '/home/ahomea/y/yaoling/MASCOT/Projects_practice/PLOTSINMOD/fig/'
else:
    SINMOD_datapath = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Adaptive_script/"
    figpath = '/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Presentation/MITPortugal/fig/SINMOD_data/'

SINMOD_path = SINMOD_datapath + 'samples_2020.05.01.nc'
SINMOD = netCDF4.Dataset(SINMOD_path)





# COLORSCALE = 'jet'
COLORSCALE = 'RdBu'

class plot3D():
    def __init__(self, data3D):
        self.X = data3D[:, 0].flatten()
        self.Y = data3D[:, 1].flatten()
        self.Z = data3D[:, 2].flatten()
        self.val = data3D[:, -1].flatten()
    # fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'scene'}, {'type': 'scene'}]],
    #                     subplot_titles=("Excursion Probability", "Directional vector"))
    def draw3DVolume(self, count, isomin = None, isomax = None, opacity = None,
                     surface_count = None):
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
                # colorbar=colorbar,
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
                # 'xaxis': {'range': [self.X.min(), self.X.max() + 90], 'rangemode': 'tozero', 'tickmode': "linear", 'tick0': 0, 'dtick': 100},
                # 'yaxis': {'range': [self.Y.min(), self.Y.max() + 90], 'rangemode': 'tozero', 'tickmode': "linear", 'tick0': 0, 'dtick': 100},
                # 'zaxis': {'range': [self.Z.min() - 1, self.Z.max()], 'rangemode': 'tozero', 'tickmode': "linear", 'tick0': 0, 'dtick': 0.5},
            },
            coloraxis=dict(colorscale=COLORSCALE),
            showlegend=False,
            scene_camera=camera
        )
        fig.update_coloraxes(colorscale=COLORSCALE)
        if server == True:
            fig.write_image(figpath + "sal_{:03d}.png".format(count), width=1980, height=1080)
            print(COLORSCALE)
        else:
            print(COLORSCALE)
            plotly.offline.plot(fig)

    def draw3DSurface(self, count):
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
            eye=dict(x=-1.25, y=-1.25, z=1.25)
        )
        fig.update_layout(
            scene={
                'aspectmode': 'manual',
                'aspectratio': dict(x=1, y=1, z=.5),
                # 'xaxis': {'range': [self.X.min(), self.X.max() + 90], 'rangemode': 'tozero', 'tickmode': "linear", 'tick0': 0, 'dtick': 100},
                # 'yaxis': {'range': [self.Y.min(), self.Y.max() + 90], 'rangemode': 'tozero', 'tickmode': "linear", 'tick0': 0, 'dtick': 100},
                # 'zaxis': {'range': [self.Z.min() - 1, self.Z.max()], 'rangemode': 'tozero', 'tickmode': "linear", 'tick0': 0, 'dtick': 0.5},
            },
            coloraxis=dict(colorscale=COLORSCALE),
            showlegend=False,
            scene_camera=camera
        )
        fig.update_coloraxes(colorscale=COLORSCALE)
        if server == True:
            fig.write_image(figpath + "sal_{:03d}.png".format(count), width=1980, height=1080)
        else:
            plotly.offline.plot(fig)



x_sinmod = SINMOD['xc']
y_sinmod = SINMOD['yc']
z_sinmod = SINMOD['zc']
sal_sinmod = SINMOD['salinity']

depth_limit = 2.5
GridS = []
for k in z_sinmod[np.array(z_sinmod) <= depth_limit]:
    for i in x_sinmod:
        for j in y_sinmod:
            GridS.append([i, j, k])

GridS = np.array(GridS)
XS = GridS[:, 0].reshape(-1, 1)
YS = GridS[:, 1].reshape(-1, 1)
ZS = GridS[:, 2].reshape(-1, 1)

if server == True:
    for t in range(sal_sinmod.shape[0]):
        # t = 0
        Sal = []
        print(t)
        for k in np.where(np.array(z_sinmod) <= depth_limit)[0]:
            for i in range(len(x_sinmod)):
                for j in range(len(y_sinmod)):
                    Sal.append(sal_sinmod[t, k, j, i])
        Sal = np.array(Sal).reshape(-1, 1)
        data3d = np.hstack((XS, YS, -ZS, Sal))
        plot3d = plot3D(data3d)
        plot3d.draw3DSurface(t)
else:
    t = 0
    Sal = []
    print(t)
    for k in np.where(np.array(z_sinmod) <= depth_limit)[0]:
        for i in range(len(x_sinmod)):
            for j in range(len(y_sinmod)):
                Sal.append(sal_sinmod[t, k, j, i])
    Sal = np.array(Sal).reshape(-1, 1)
    Sal = np.ma.filled(Sal, -1)
    data3d = np.hstack((XS, YS, -ZS, Sal))
    plot3d = plot3D(data3d)
    # plot3d.draw3DSurface(t)
    plot3d.draw3DVolume(t, isomin = 0)




