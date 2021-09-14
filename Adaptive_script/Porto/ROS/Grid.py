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
import matplotlib.pyplot as plt
import matplotlib.path as mplPath # used to determine whether a point is inside the grid or not

'''
Goal of the script is to make the class structure
Grid: generate the grid
GP: take care of the processes
Path planner: plan the next waypoint

This grid generation will generate the polygon grid as desired, using non-binary tree with recursion, it is very efficient
'''

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

class GridPoly(WaypointNode):
    '''
    generate the polygon grid with equal-distance from one to another
    '''
    lat_origin, lon_origin = 41.10251, -8.669811  # the right bottom corner coordinates
    circumference = 40075000  # circumference of the earth, [m]
    distance_poly = 60  # [m], distance between two neighbouring points
    depth_obs = [-.5, -1.25, -2] # [m], distance in depth, depth to be explored
    distanceTolerance = .1 # [m], distance tolerance for the neighbouring points
    pointsPr = 1000  # points per layer
    polygon = None
    loc_start = None
    counter_plot = 0  # counter for plot number
    counter_grid = 0  # counter for grid points
    debug = True
    voiceCtrl = False
    figpath = None

    def __init__(self, polygon = np.array([[41.12251, -8.707745], # default polygon, needs to be updated
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
        self.polygon = polygon
        self.debug = debug
        self.voiceCtrl = voiceCtrl
        self.polygon_path = mplPath.Path(self.polygon)
        self.angle_poly = self.deg2rad(np.arange(0, 6) * 60)  # angles for polygon, can be changed to have more points
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

    def checkFolder(self):
        self.figpath = os.getcwd() + "/fig/"
        print("figpath: ", self.figpath)
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
        prev = self.polygon[-1]
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

    @staticmethod
    def deg2rad(deg):
        return deg / 180 * np.pi

    @staticmethod
    def rad2deg(rad):
        return rad / np.pi * 180

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

