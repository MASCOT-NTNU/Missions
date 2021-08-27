#! /usr/bin/env python3
__author__ = "Yaolin Ge"
__copyright__ = "Copyright 2021, The MASCOT Project, NTNU (https://wiki.math.ntnu.no/mascot)"
__credits__ = ["Yaolin Ge"]
__license__ = "MIT"
__version__ = "1.0.1"
__maintainer__ = "Yaolin Ge"
__email__ = "yaolin.ge@ntnu.no"
__status__ = "UnderDevelopment"

import numpy as np
from gmplot import GoogleMapPlotter
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.pyplot as plt
import matplotlib.path as mplPath # used to determine whether a point is inside the grid or not
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly
plotly.io.orca.config.executable = '/usr/local/bin/orca'
plotly.io.orca.config.save()

'''
Goal of the script is to make the class structure
Grid: generate the grid
GP: take care of the processes
Path planner: plan the next waypoint
'''

class Grid:
    '''
    Grid generates the waypoint grid graph
    '''
    lat_origin, lon_origin = 41.10251, -8.669811 # the right bottom corner coordinates
    origin = [lat_origin, lon_origin]  # set the origin at the right bottom corner
    distance = 6000  # distance of the edge
    depth_obs = [0.5, 1.25, 2.0]  # planned depth to be observed
    alpha = 115  # angle to be tilted
    circumference = 40075000  # circumference of the earth, [m]
    depth_tolerance = 0.25  # tolerance +/- in depth, 0.5 m == [0.25 ~ 0.75]m

    N1 = 21  # number of grid points along north direction
    N2 = 41  # number of grid points along east direction
    N3 = len(depth_obs)  # number of layers in the depth dimension
    N = N1 * N2 * N3  # total number of grid points
    XLIM = [0, distance]  # limit for the x axis
    YLIM = [0, distance]  # limit for the y axis
    ZLIM = [depth_obs[0], depth_obs[-1]]  # limit for the z axis
    x = np.linspace(XLIM[0], XLIM[-1], N1) # x coordinates
    y = np.linspace(YLIM[0], YLIM[-1], N1) # y coordinates
    z = np.array(depth_obs).reshape(-1, 1)  # z coordinates
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    R = None # rotational matrix
    box = None # bounding box
    grid = None # grid coordinates xy
    grid_coord = None

    def __init__(self):
        self.computeR()
        self.generateBox()
        self.generateGrid()
        self.generateGridCoord()
        self.print_grid()
        print("Grid is generated successfully!\n\n")

    def print_grid(self):
        print("circumference: ", self.circumference)
        print("depth tolerance: ", self.depth_tolerance)
        print("alpha: ", self.alpha)
        print("N1: ", self.N1)
        print("N2: ", self.N2)
        print("N3: ", self.N3)
        print("distance: ", self.distance)
        print("depth_obs: ", self.depth_obs)
        print("origin: ", self.origin)
        print("XLIM: ", self.XLIM)
        print("YLIM: ", self.YLIM)
        print("ZLIM: ", self.ZLIM)

    def set_distance(self, value):
        self.distance = value

    def set_depth_tolerance(self, value):
        self.depth_tolerance = value

    def set_origin(self, lat, lon):
        self.lat_origin = lat
        self.lon_origin = lon

    def set_circumference(self, value):
        self.circumference = value

    def set_depthObs(self, value):
        self.depth_obs = value

    def set_alpha(self, value):
        self.alpha = value

    def set_N1(self, value):
        self.N1 = value

    def set_N2(self, value):
        self.N2 = value

    @staticmethod
    def deg2rad(deg):
        return deg / 180 * np.pi

    @staticmethod
    def rad2deg(rad):
        return rad / np.pi * 180

    def computeR(self):
        # rotational matrix convert tilted x'y' to regual xy
        self.R = np.array([[np.cos(self.deg2rad(self.alpha)), np.sin(self.deg2rad(self.alpha))],
                          [-np.sin(self.deg2rad(self.alpha)), np.cos(self.deg2rad(self.alpha))]])
        print("Rotational matrix is computed:")
        print(self.R)

    def BBox(self):
        '''
        :return: Box of the operational regions
        '''
        lat4 = self.deg2rad(self.lat_origin)  # the right bottom corner
        lon4 = self.deg2rad(self.lon_origin)  # the right bottom corner

        lat2 = lat4 + self.distance * np.sin(self.deg2rad(self.alpha)) / self.circumference * 2 * np.pi
        lat1 = lat2 + self.distance * np.cos(self.deg2rad(self.alpha)) / self.circumference * 2 * np.pi
        lat3 = lat4 + self.distance * np.sin(np.pi / 2 - self.deg2rad(self.alpha)) / self.circumference * 2 * np.pi

        lon2 = lon4 + self.distance * np.cos(self.deg2rad(self.alpha)) / (self.circumference * np.cos(lat2)) * 2 * np.pi
        lon3 = lon4 - self.distance * np.cos(np.pi / 2 - self.deg2rad(self.alpha)) / (
                    self.circumference * np.cos(lat3)) * 2 * np.pi
        lon1 = lon3 + self.distance * np.cos(self.deg2rad(self.alpha)) / (self.circumference * np.cos(lat1)) * 2 * np.pi

        box = np.vstack((np.array([lat1, lat2, lat3, lat4]), np.array([lon1, lon2, lon3, lon4]))).T
        return self.rad2deg(box)

    def GGrid(self):
        grid = []
        for k in range(self.N3):
            for i in range(self.N1):
                for j in range(self.N2):
                    tempx, tempy = self.R @ np.vstack((self.x[i], self.y[j]))
                    grid.append([tempx[0], tempy[0], self.depth_obs[k]])
        return np.array(grid)

    def generateGridCoord(self):
        lat_grid = np.zeros([self.N1, self.N2])
        lon_grid = np.zeros([self.N1, self.N2])
        for i in range(self.N1):
            for j in range(self.N2):
                xnew, ynew = self.R @ np.vstack((self.x[i], self.y[j])) # rotate the xy into the new frame
                lat_grid[i, j] = self.lat_origin + self.rad2deg(xnew * np.pi * 2.0 / self.circumference)
                lon_grid[i, j] = self.lon_origin + self.rad2deg(ynew * np.pi * 2.0 /
                                (self.circumference * np.cos(self.deg2rad(lat_grid[i, j]))))
        self.grid_coord = np.hstack((lat_grid.reshape(-1, 1), lon_grid.reshape(-1, 1)))
        print("Grid coordinates are generated, shape is ")
        print(self.grid_coord.shape)

    def generateBox(self):
        self.box = self.BBox()

    def generateGrid(self):
        self.grid = self.GGrid()

    def checkGrid(self):
        # Make 3D plot # #
        fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scene'}]])
        fig.add_trace(
            go.Scatter3d(
                x=self.grid[:, 0], y=self.grid[:, 1], z=self.grid[:, -1],
            ),
            row=1, col=1
        )
        fig.update_layout(
            scene={
                'aspectmode': 'manual',
                'aspectratio': dict(x=1, y=1, z=.5),
            },
            showlegend=False,
            title="AUV explores the field"
        )
        plotly.offline.plot(fig, filename="grid.html", auto_open=True)

    @staticmethod
    def checkBox(lat_origin, lon_origin, box):
        initial_zoom = 12
        apikey = 'AIzaSyAZ_VZXoJULTFQ9KSPg1ClzHEFjyPbJUro'
        gmap = GoogleMapPlotter(lat_origin, lon_origin, initial_zoom, map_type='satellite', apikey=apikey)
        gmap.scatter(box[:, 0], box[:, 1], 'cornflowerblue', size=10)
        gmap.draw("/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/MapPlot/box.html")

    @staticmethod
    def checkGridCoord(lat_origin, lon_origin, lat, lon):
        initial_zoom = 12
        apikey = 'AIzaSyAZ_VZXoJULTFQ9KSPg1ClzHEFjyPbJUro'
        gmap = GoogleMapPlotter(lat_origin, lon_origin, initial_zoom, map_type='satellite', apikey=apikey)
        gmap.scatter(lat, lon, color='#99ff00', size=20, marker=False)
        gmap.draw("/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/MapPlot/map.html")


from progress.bar import Bar

class GridPoly(Grid):

    def __init__(self, coord_polygon):
        self.lat_origin, self.lon_origin = 41.061874, -8.650977
        self.reluctance = [30, 150] # distance between two points, min and max
        self.pointsPr = 1000 # points per layer
        self.coord_polygon = coord_polygon
        self.getBox()
        self.getGrid()
        self.checkBox(self.lat_origin, self.lon_origin, self.coord_polygon)
        self.getPolygonArea()
        self.plotGridonMap(self.grid)
        # self.generateBaseGrid()
        # Grid.__init__(self)
        # os.system("say initialised")

    def getBox(self):
        self.lat_min = np.amin(self.coord_polygon[:, 0])
        self.lon_min = np.amin(self.coord_polygon[:, -1])
        self.lat_max = np.amax(self.coord_polygon[:, 0])
        self.lon_max = np.amax(self.coord_polygon[:, -1])
        self.box = np.array([[self.lat_min, self.lon_min], [self.lat_max, self.lon_min],
                             [self.lat_min, self.lon_max], [self.lat_max, self.lon_max]])

    def getGrid(self):
        counter = 0
        self.grid = np.empty((0, 2))
        self.poly_path = mplPath.Path(self.coord_polygon)
        while counter <= self.pointsPr:
            lat_random = np.random.uniform(self.lat_min, self.lat_max)
            lon_random = np.random.uniform(self.lon_min, self.lon_max)
            if self.poly_path.contains_point((lat_random, lon_random)):
                if self.grid.shape[0] == 0:
                    self.grid = np.append(self.grid, np.array([lat_random, lon_random]).reshape(1, -1), axis = 0)
                    counter = counter + 1
                    print(counter)
                else:
                    ind_neighbour = ((self.grid[:, 0] - lat_random) ** 2 + (self.grid[:, 1] - lon_random) ** 2).argmin()
                    if (self.getDistance(self.grid[ind_neighbour], [lat_random, lon_random]) >= self.reluctance[0]):
                            # (self.getDistance(self.grid[ind_neighbour], [lat_random, lon_random]) <= self.reluctance[1]):
                        self.grid = np.append(self.grid, np.array([lat_random, lon_random]).reshape(1, -1), axis = 0)
                        counter = counter + 1
                        print(counter)
                    else:
                        pass

    def getPolygonArea(self):
        area = 0
        prev = self.coord_polygon[-1]
        for i in range(self.coord_polygon.shape[0]):
            now = self.coord_polygon[i]
            xnow, ynow = GridPoly.latlon2xy(now[0], now[1])
            xpre, ypre = GridPoly.latlon2xy(prev[0], prev[1])
            area += xnow * ypre - ynow * xpre
            prev = now
        self.PolyArea = area / 2
        print("Area: ", self.PolyArea / 1e6, " km2")
        os.system("say The area covered is {:.1f} squared kilometers".format(self.PolyArea / 1e6))

    def plotGridonMap(self, grid):
        def color_scatter(gmap, lats, lngs, values=None, colormap='coolwarm',
                          size=None, marker=False, s=None, **kwargs):
            def rgb2hex(rgb):
                """ Convert RGBA or RGB to #RRGGBB """
                rgb = list(rgb[0:3])  # remove alpha if present
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
                gmap.scatter(lats=[lat], lngs=[lon], c=c, size=size, marker=marker, s=s, **kwargs)
        initial_zoom = 12
        apikey = 'AIzaSyAZ_VZXoJULTFQ9KSPg1ClzHEFjyPbJUro'
        gmap = GoogleMapPlotter(grid[0, 0], grid[0, 1], initial_zoom, apikey=apikey)
        color_scatter(gmap, grid[:, 0], grid[:, 1], np.zeros_like(grid[:, 0]), size=20, colormap='hsv')
        gmap.draw("/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/MapPlot/map.html")

    @staticmethod
    def latlon2xy(lat, lon):
        x = GridPoly.deg2rad((lat - GridPoly.lat_origin)) / 2 / np.pi * GridPoly.circumference
        y = GridPoly.deg2rad((lon - GridPoly.lon_origin)) / 2 / np.pi * GridPoly.circumference * np.cos(GridPoly.deg2rad(lat))
        return x, y

    @staticmethod
    def getDistance(coord1, coord2):
        x1, y1 = GridPoly.latlon2xy(coord1[0], coord1[1])
        x2, y2 = GridPoly.latlon2xy(coord2[0], coord2[1])
        dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        return dist

# polygon = np.array([[41.119233, -8.684664],
#                     [41.124663, -8.719759],
#                     [41.154379, -8.719759],
#                     [41.174658, -8.716345],
#                     [41.184653, -8.755802],
#                     [41.152665, -8.787862],
#                     [41.124520, -8.784258],
#                     [41.097793, -8.744800],
#                     [41.106512, -8.669678]])

polygon = np.array([[41.154048,-8.690331],
                    [41.151126,-8.697998],
                    [41.146167,-8.699673],
                    [41.142020,-8.698232],
                    [41.138724,-8.695476],
                    [41.135439,-8.692878],
                    [41.134865,-8.686244],
                    [41.136944,-8.677676],
                    [41.139944,-8.679487],
                    [41.139344,-8.686413],
                    [41.140632,-8.690824],
                    [41.142870,-8.693485],
                    [41.145835,-8.694987],
                    [41.150319,-8.693925],
                    [41.151651,-8.688966]])


a = GridPoly(polygon)
import os
os.system("say finished")
# a = Grid()
# a.checkGridCoord()
# a.checkBox()
