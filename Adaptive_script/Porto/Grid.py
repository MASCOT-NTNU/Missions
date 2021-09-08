#! /usr/bin/env python3
__author__ = "Yaolin Ge"
__copyright__ = "Copyright 2021, The MASCOT Project, NTNU (https://wiki.math.ntnu.no/mascot)"
__credits__ = ["Yaolin Ge"]
__license__ = "MIT"
__version__ = "1.0.1"
__maintainer__ = "Yaolin Ge"
__email__ = "yaolin.ge@ntnu.no"
__status__ = "UnderDevelopment"

import time

import numpy as np
import os
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
    Grid generates the waypoint grid graph with a square
    '''
    lat_origin, lon_origin = 41.10251, -8.669811 # the right bottom corner coordinates
    origin = [lat_origin, lon_origin]  # set the origin at the right bottom corner
    distance = 6000  # distance of the edge
    depth_obs = [0.5, 1.0, 1.5, 2.0, 2.5]  # planned depth to be observed
    alpha = 60  # angle to be tilted
    circumference = 40075000  # circumference of the earth, [m]
    depth_tolerance = 0.25  # tolerance +/- in depth, 0.5 m == [0.25 ~ 0.75]m

    N1 = 31  # number of grid points along north direction
    N2 = 31  # number of grid points along east direction
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

class WaypointNode:
    '''
    generate node for each waypoint
    '''
    waypoint_loc = None
    subwaypoint_len = 0
    subwaypoint_loc = []
    def __init__(self, subwaypoints_len, subwaypoints_loc, waypoint_loc):
        self.subwaypoint_len = subwaypoints_len
        self.subwaypoint_loc = subwaypoints_loc
        self.waypoint_loc = waypoint_loc

class GridPoly(Grid, WaypointNode):
    '''
    generate the polygon grid with equal-distance from one to another
    '''
    distance_poly = 100  # [m], distance between two neighbouring points
    depth_obs = [-.5, -1.25, -2] # [m], distance in depth, depth to be explored
    polygon = None
    loc_start = None
    counter_plot = 0  # counter for plot number
    counter_grid = 0  # counter for grid points
    debug = True
    voiceCtrl = False
    figpath = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Porto/Setup/Grid/fig/"

    def __init__(self, polygon = np.array([[41.12251, -8.707745],
                                        [41.12413, -8.713079],
                                        [41.11937, -8.715101],
                                        [41.11509, -8.717317],
                                        [41.11028, -8.716535],
                                        [41.10336, -8.716813],
                                        [41.10401, -8.711306],
                                        [41.11198, -8.710787],
                                        [41.11764, -8.710245],
                                        [41.12251, -8.707745]]), debug = True, voiceCtrl = False):
        if debug:
            self.checkFolder()
        self.lat_origin, self.lon_origin = 41.061874, -8.650977 # origin location
        self.grid_poly = []
        self.pointsPr = 1000  # points per layer
        self.polygon = polygon
        self.debug = debug
        self.voiceCtrl = voiceCtrl
        self.polygon_path = mplPath.Path(self.polygon)
        self.angle_poly = self.deg2rad(np.arange(0, 6) * 60)  # angles for polygon
        self.getPolygonArea()

        print("Grid polygon is activated!")
        print("Distance between neighbouring points: ", self.distance_poly)
        print("Depth to be observed: ", self.depth_obs)
        print("Starting location: ", self.loc_start)
        print("Polygon: ", self.polygon.shape)
        print("Points desired: ", self.pointsPr)
        print("Debug mode: ", self.debug)
        print("fig path: ", self.figpath)
        t1 = time.time()
        self.getGridPoly()
        t2 = time.time()
        print("Grid discretisation takes: {:.2f} seconds".format(t2 - t1))
        # self.checkSingular()

    # def checkSingular(self):
    #     x, y = self.latlon2xy(self.grid_poly[:, 0], self.grid_poly[:, 1], self.lat_origin, self.lon_origin)
    #     x = x.reshape(-1, 1)
    #     y = y.reshape(-1, 1)
    #     grid = np.hstack((y, x))
    #     import scipy.spatial.distance as scdist
    #     t = scdist.cdist(grid, grid)
    #     print(["Positive " if np.all(np.linalg.eigvals(t) > 0) else "Singular"])
    #     plt.figure()
    #     plt.plot(self.grid_poly[:, 1], self.grid_poly[:, 0], 'k.')
    #     plt.title("grid discretisation")
    #     plt.show()

    def checkFolder(self):
        i = 0
        while os.path.exists(self.figpath + "P%s" % i):
            i += 1
        self.figpath = self.figpath + "P%s" % i
        if not os.path.exists(self.figpath):
            print(self.figpath + " is created")
            os.mkdir(self.figpath)
        else:
            print(self.figpath + " is already existed")

    def revisit(self, loc):
        '''
        func determines whether it revisits the points it already have
        '''
        temp = np.array(self.grid_poly)
        if len(self.grid_poly) > 0:
            dist_min = np.min(np.sqrt((temp[:, 0] - loc[0]) ** 2 + (temp[:, 1] - loc[1]) ** 2))
            ind = np.argmin(np.sqrt((temp[:, 0] - loc[0]) ** 2 + (temp[:, 1] - loc[1]) ** 2))
            if dist_min <= .00001:
                return [True, ind]
            else:
                return [False, []]
        else:
            return [False, []]

    def getNewLocations(self, loc):
        '''
        get new locations around the current location 
        '''
        lat_delta, lon_delta = self.xy2latlon(self.distance_poly * np.sin(self.angle_poly),
                                              self.distance_poly * np.cos(self.angle_poly), 0, 0)
        return lat_delta + loc[0], lon_delta + loc[1]

    def getStartLocation(self):
        lat_min = np.amin(self.polygon[:, 0])
        lat_max = np.amax(self.polygon[:, 0])
        lon_min = np.amin(self.polygon[:, 1])
        lon_max = np.amax(self.polygon[:, 1])
        path_polygon = mplPath.Path(self.polygon)
        while True:
            lat_random = np.random.uniform(lat_min, lat_max)
            lon_random = np.random.uniform(lon_min, lon_max)
            if path_polygon.contains_point((lat_random, lon_random)):
                break
        print("The generated random starting location is: ")
        print([lat_random, lon_random])
        self.loc_start = [lat_random, lon_random]
    
    def getGridPoly(self):
        '''        
        get the polygon grid discretisation 
        '''
        self.getStartLocation()
        lat_new, lon_new = self.getNewLocations(self.loc_start)
        start_node = []
        for i in range(len(self.angle_poly)):
            if self.polygon_path.contains_point((lat_new[i], lon_new[i])):
                start_node.append([lat_new[i], lon_new[i]])
                self.grid_poly.append([lat_new[i], lon_new[i]])
                self.counter_grid = self.counter_grid + 1

        if self.debug:
            self.counter_plot = self.counter_plot + 1
            print(self.counter_grid)
            plt.figure(figsize=(10, 10))
            temp1 = np.array(self.grid_poly)
            plt.plot(temp1[:, 1], temp1[:, 0], 'k.')
            plt.plot(self.loc_start[1], self.loc_start[0], 'bx')
            plt.plot(self.polygon[:, 1], self.polygon[:, 0], 'r-')
            plt.xlabel("Lon [deg]")
            plt.ylabel("Lat [deg]")
            plt.title(
                "Step No. {:04d}, added {:1d} new points, {:1d} total points in the grid".format(self.counter_plot, 6, 6))

            plt.savefig(self.figpath + "/I_{:04d}.png".format(self.counter_plot))
            plt.close("all")
        WaypointNode_start = WaypointNode(len(start_node), start_node, self.loc_start)
        Allwaypoints = self.getAllWaypoints(WaypointNode_start)
        self.grid_poly = np.array(self.grid_poly)
        if len(self.grid_poly) > self.pointsPr:
            print("{:d} waypoints are generated, only {:d} waypoints are selected!".format(len(self.grid_poly), self.pointsPr))
            self.grid_poly = self.grid_poly[:self.pointsPr, :]
        else:
            print("{:d} waypoints are generated, all are selected!".format(len(self.grid_poly)))
        print("Grid: ", self.grid_poly.shape)

    def getAllWaypoints(self, waypoint_node):
        if self.counter_grid > self.pointsPr:  # stopping criterion to end the recursion
            return WaypointNode(0, [], waypoint_node.waypoint_loc)
        # print(self.counter_grid)
        for i in range(waypoint_node.subwaypoint_len):  # loop through all the subnodes
            subsubwaypoint = []
            length_new = 0
            lat_subsubwaypoint, lon_subsubwaypoint = self.getNewLocations(waypoint_node.subwaypoint_loc[i])  # generate candidates location
            for j in range(len(self.angle_poly)):
                if self.polygon_path.contains_point((lat_subsubwaypoint[j], lon_subsubwaypoint[j])):
                    testRevisit = self.revisit([lat_subsubwaypoint[j], lon_subsubwaypoint[j]])
                    if not testRevisit[0]:
                        subsubwaypoint.append([lat_subsubwaypoint[j], lon_subsubwaypoint[j]])
                        self.grid_poly.append([lat_subsubwaypoint[j], lon_subsubwaypoint[j]])
                        self.counter_grid = self.counter_grid + 1
                        length_new = length_new + 1
            if len(subsubwaypoint) > 0:
                if self.debug:
                    self.counter_plot = self.counter_plot + 1
                    print(self.counter_grid)
                    plt.figure(figsize=(10, 10))
                    temp1 = np.array(self.grid_poly)
                    plt.plot(temp1[:, 1], temp1[:, 0], 'k.')
                    plt.plot(temp1[-length_new:][:, 1], temp1[-length_new:][:, 0], 'g.')
                    plt.plot(waypoint_node.subwaypoint_loc[i][1], waypoint_node.subwaypoint_loc[i][0], 'bx')
                    plt.plot(self.polygon[:, 1], self.polygon[:, 0], 'r-')
                    plt.xlabel("Lon [deg]")
                    plt.ylabel("Lat [deg]")
                    plt.title("Step No. {:04d}, added {:1d} new points, {:1d} total points in the grid".format(self.counter_plot,
                                                                                                         length_new,
                                                                                                         self.counter_grid))
                    plt.savefig(self.figpath + "/I_{:04d}.png".format(self.counter_plot))
                    plt.close("all")
                Subwaypoint = WaypointNode(len(subsubwaypoint), subsubwaypoint, waypoint_node.subwaypoint_loc[i])
                self.getAllWaypoints(Subwaypoint)
            else:
                return WaypointNode(0, [], waypoint_node.subwaypoint_loc[i])
        return WaypointNode(0, [], waypoint_node.waypoint_loc)

    def getPolygonArea(self):
        area = 0
        prev = self.polygon[-2]
        for i in range(self.polygon.shape[0] - 1):
            now = self.polygon[i]
            xnow, ynow = GridPoly.latlon2xy(now[0], now[1], self.lat_origin, self.lon_origin)
            xpre, ypre = GridPoly.latlon2xy(prev[0], prev[1], self.lat_origin, self.lon_origin)
            area += xnow * ypre - ynow * xpre
            prev = now
        self.PolyArea = area / 2
        print("Area: ", self.PolyArea / 1e6, " km2")
        if self.voiceCtrl:
            os.system("say Area is: {:.1f} squared kilometers".format(self.PolyArea / 1e6))

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
    def latlon2xy(lat, lon, lat_origin, lon_origin):
        x = GridPoly.deg2rad((lat - lat_origin)) / 2 / np.pi * GridPoly.circumference
        y = GridPoly.deg2rad((lon - lon_origin)) / 2 / np.pi * GridPoly.circumference * np.cos(GridPoly.deg2rad(lat))
        return x, y

    @staticmethod
    def xy2latlon(x, y, lat_origin, lon_origin):
        lat = lat_origin + GridPoly.rad2deg(x * np.pi * 2.0 / GridPoly.circumference)
        lon = lon_origin + GridPoly.rad2deg(y * np.pi * 2.0 / (GridPoly.circumference * np.cos(GridPoly.deg2rad(lat))))
        return lat, lon

    @staticmethod
    def getDistance(coord1, coord2):
        x1, y1 = GridPoly.latlon2xy(coord1[0], coord1[1], GridPoly.lat_origin, GridPoly.lon_origin)
        x2, y2 = GridPoly.latlon2xy(coord2[0], coord2[1], GridPoly.lat_origin, GridPoly.lon_origin)
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

# polygon = np.array([[41.154048,-8.690331],
#                     [41.151126,-8.697998],
#                     [41.146167,-8.699673],
#                     [41.142020,-8.698232],
#                     [41.138724,-8.695476],
#                     [41.135439,-8.692878],
#                     [41.134865,-8.686244],
#                     [41.136944,-8.677676],
#                     [41.139944,-8.679487],
#                     [41.139344,-8.686413],
#                     [41.140632,-8.690824],
#                     [41.142870,-8.693485],
#                     [41.145835,-8.694987],
#                     [41.150319,-8.693925],
#                     [41.151651,-8.688966]])

# polygon = np.array([[41.07765, -8.718977],
#                     [41.07814, -8.705016],
#                     [41.11243, -8.706682],
#                     [41.11566, -8.718444],
#                     [41.07765, -8.718977]])

# polygon = np.array([[41.12251, -8.707745],
#                     [41.12413, -8.713079],
#                     [41.11937, -8.715101],
#                     [41.11509, -8.717317],
#                     [41.11028, -8.716535],
#                     [41.10336, -8.716813],
#                     [41.10401, -8.711306],
#                     [41.11198, -8.710787],
#                     [41.11764, -8.710245],
#                     [41.12251, -8.707745]])

# polygon = np.array([[0, 0],
#                     [.001, .005],
#                     [.005, .01],
#                     [.01, .01],
#                     [.007, .003],
#                     [.011, -.003],
#                     [.013, -.002],
#                     [.013, -.005],
#                     [.01, -.006],
#                     [.004, -.004],
#                     [0, 0]])

# polygon = np.array([[41.142246, -8.689333],
#                     [41.140181, -8.689711],
#                     [41.148664, -8.703608],
#                     [41.136460, -8.702222],
#                     [41.142246, -8.689333]])

if __name__ == "__main__":
    a = GridPoly(polygon = np.array([[41.12902, -8.69901],
                        [41.12382, -8.68799],
                        [41.12642, -8.67469],
                        [41.12071, -8.67189],
                        [41.11743, -8.68336],
                        [41.11644, -8.69869],
                        [41.12295, -8.70283],
                        [41.12902, -8.69901]]), debug = False)
    lat = a.grid_poly[:, 0].reshape(-1, 1)
    lon = a.grid_poly[:, 1].reshape(-1, 1)
    grid_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Porto/Setup/Grid/grid.txt"
    grid = np.hstack((lat, lon))
    np.savetxt(grid_path, grid, delimiter = ",")

    x, y = a.latlon2xy(lat, lon, a.lat_origin, a.lon_origin)
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    grid = np.hstack((y, x))
    import scipy.spatial.distance as scdist
    t = scdist.cdist(grid, grid)
    # print(np.linalg.cholesky(t))
    from Adaptive_script.Porto.usr_func import *
    Sigma = Matern_cov(2, 4.5/600, t)

    plt.figure()
    plt.plot(a.grid_poly[:, 1], a.grid_poly[:, 0], 'k.')
    plt.title("grid discretisation")
    plt.show()

    plt.figure()
    plt.imshow(Sigma, cmap = "Paired")
    plt.colorbar()
    plt.title("Covariance matrix is " + "Positive definite" if np.all(np.linalg.eigvals(Sigma) > 0) else "Singular")
    plt.xlabel("y")
    plt.ylabel("x")
    # plt.savefig(a.figpath + "Sigma.pdf")
    plt.show()

    print(np.linalg.cholesky(Sigma))
    # a = GridPoly(debug = False)
    # t1 = time.time()
    # a.getGridPoly()
    # t2 = time.time()
    # print("Time consumed: ", t2 - t1)
    # a.plotGridonMap(a.grid_poly)
    # os.system("say finished")
    # a = Grid()
    # a.checkGridCoord()
    # a.checkBox()




