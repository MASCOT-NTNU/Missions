#! /usr/bin/env python3
__author__ = "Yaolin Ge"
__copyright__ = "Copyright 2021, The MASCOT Project, NTNU (https://wiki.math.ntnu.no/mascot)"
__credits__ = ["Yaolin Ge"]
__license__ = "MIT"
__version__ = "1.0.1"
__maintainer__ = "Yaolin Ge"
__email__ = "yaolin.ge@ntnu.no"
__status__ = "UnderDevelopment"

from usr_func import *
import time
import os
import matplotlib.pyplot as plt
import matplotlib.path as mplPath  # used to determine whether a point is inside the grid or not

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
    distance_poly = 100  # [m], distance between two neighbouring points
    depth_obs = [-.5, -1.25, -2]  # [m], distance in depth, depth to be explored
    pointsPr = 1000  # points per layer
    polygon = None
    loc_start = None
    counter_plot = 0  # counter for plot number
    counter_grid = 0  # counter for grid points

    def __init__(self):
        self.lat_origin, self.lon_origin = 41.061874, -8.650977  # origin location
        self.grid_poly = []
        self.load_global_path()
        self.load_polygon()
        self.polygon_path = mplPath.Path(self.polygon)
        self.angle_poly = deg2rad(np.arange(0, 6) * 60) + 30  # angles for polygon
        self.getPolygonArea()
        print("Grid polygon is activated!")
        print("Distance between neighbouring points: ", self.distance_poly)
        print("Depth to be observed: ", self.depth_obs)
        print("Starting location: ", self.loc_start)
        print("Polygon: ", self.polygon.shape)
        print("Points desired: ", self.pointsPr)
        t1 = time.time()
        self.getGridPoly()
        self.savegrid()
        t2 = time.time()
        print("Grid discretisation takes: {:.2f} seconds".format(t2 - t1))

    def load_global_path(self):
        print("Now it will load the global path.")
        self.path_global = open("path_global.txt", 'r').read()
        print("global path is set up successfully!")
        print(self.path_global)

    def load_polygon(self):
        print("Loading the polygon...")
        self.polygon = np.loadtxt(self.path_global + '/Config/polygon.txt', delimiter=", ")
        print("Polygon is loaded successfully!")

    def savegrid(self):
        grid = []
        for i in range(len(self.grid_poly)):
            for j in range(len(self.depth_obs)):
                grid.append([self.grid_poly[i, 0], self.grid_poly[i, 1], self.depth_obs[j]])
        np.savetxt(self.path_global + "/Config/grid.txt", grid, delimiter=", ")
        print("Grid is created correctly, it is saved to grid.txt")

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
        lat_delta, lon_delta = xy2latlon(self.distance_poly * np.sin(self.angle_poly),
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

        WaypointNode_start = WaypointNode(len(start_node), start_node, self.loc_start)
        Allwaypoints = self.getAllWaypoints(WaypointNode_start)
        self.grid_poly = np.array(self.grid_poly)
        if len(self.grid_poly) > self.pointsPr:
            print("{:d} waypoints are generated, only {:d} waypoints are selected!".format(len(self.grid_poly),self.pointsPr))
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
            lat_subsubwaypoint, lon_subsubwaypoint = self.getNewLocations(
                waypoint_node.subwaypoint_loc[i])  # generate candidates location
            for j in range(len(self.angle_poly)):
                if self.polygon_path.contains_point((lat_subsubwaypoint[j], lon_subsubwaypoint[j])):
                    testRevisit = self.revisit([lat_subsubwaypoint[j], lon_subsubwaypoint[j]])
                    if not testRevisit[0]:
                        subsubwaypoint.append([lat_subsubwaypoint[j], lon_subsubwaypoint[j]])
                        self.grid_poly.append([lat_subsubwaypoint[j], lon_subsubwaypoint[j]])
                        self.counter_grid = self.counter_grid + 1
                        length_new = length_new + 1
            if len(subsubwaypoint) > 0:
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
            xnow, ynow = latlon2xy(now[0], now[1], self.lat_origin, self.lon_origin)
            xpre, ypre = latlon2xy(prev[0], prev[1], self.lat_origin, self.lon_origin)
            area += xnow * ypre - ynow * xpre
            prev = now
        self.PolyArea = area / 2
        print("Area: ", self.PolyArea / 1e6, " km2")

    @staticmethod
    def getDistance(coord1, coord2):
        x1, y1 = latlon2xy(coord1[0], coord1[1], GridPoly.lat_origin, GridPoly.lon_origin)
        x2, y2 = latlon2xy(coord2[0], coord2[1], GridPoly.lat_origin, GridPoly.lon_origin)
        dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        return dist


if __name__ == "__main__":
    # polygon = np.array([[0, 0]])
    grid = GridPoly()






