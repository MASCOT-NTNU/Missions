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

    def deg2rad(self, deg):
        return deg / 180 * np.pi

    def rad2deg(self, rad):
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

    def checkBox(self):
        from gmplot import GoogleMapPlotter
        initial_zoom = 12
        apikey = 'AIzaSyAZ_VZXoJULTFQ9KSPg1ClzHEFjyPbJUro'
        gmap = GoogleMapPlotter(self.lat_origin, self.lon_origin, initial_zoom, map_type='satellite', apikey=apikey)
        gmap.scatter(self.box[:, 0], self.box[:, 1], 'cornflowerblue', size=10)
        gmap.draw("/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/MapPlot/map.html")

    def checkGrid(self):
        import plotly.graph_objects as go
        import plotly
        plotly.io.orca.config.executable = '/Users/yaoling/anaconda3/bin/orca/'
        plotly.io.orca.config.save()
        from plotly.subplots import make_subplots

        # Make 3D plot # #
        fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scene'}]])
        fig.add_trace(
            go.Scatter3d(
                x=self.grid[:, 0], y=self.grid[:, 1], z=self.grid[:, -1],
                # marker=dict(
                #     color="black",
                #     showscale=False
                # ),
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

    def checkGridCoord(self):
        from gmplot import GoogleMapPlotter
        initial_zoom = 12
        apikey = 'AIzaSyAZ_VZXoJULTFQ9KSPg1ClzHEFjyPbJUro'
        gmap = GoogleMapPlotter(self.lat_origin, self.lon_origin, initial_zoom, map_type='satellite', apikey=apikey)
        gmap.scatter(self.grid_coord[:, 0], self.grid_coord[:, 1], color='#99ff00', size=20, marker=False)
        gmap.draw("/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/MapPlot/map.html")

a = Grid()
a.checkGridCoord()
# a.checkBox()
