
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
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'font.style': 'oblique'})

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


from progress.bar import Bar


class Node:
    node_loc = None
    subnode_len = 0
    subnode_loc = []
    def __init__(self, subnodes_len, subnodes_loc, node_loc):
        self.subnode_len = subnodes_len
        self.subnode_loc = subnodes_loc
        self.node_loc = node_loc

class GridPoly(Grid, Node):
    dist_poly = 60 # [m], distance between two neighbouring points
    polygon = None
    grid_poly = []
    counter_plot = 0 # counter for plot number
    counter_grid = 0 # counter for grid points
    figpath = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Porto/Setup/Grid/fig/"

    def __init__(self, polygon):
        self.lat_origin, self.lon_origin = 41.061874, -8.650977
        self.pointsPr = 1000 # points per layer
        self.polygon = polygon
        self.polygon_path = mplPath.Path(self.polygon)
        self.angle_poly = self.deg2rad(np.arange(0, 6) * 60) # angles for polygon

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

    def getNewPoints(self, loc):
        lat_delta, lon_delta = self.xy2latlon(self.dist_poly * np.cos(self.angle_poly),
                                              self.dist_poly * np.sin(self.angle_poly), 0, 0)
        return lat_delta + loc[0], lon_delta + loc[1]

    def getStart(self, loc):
        lat_new, lon_new = self.getNewPoints(loc)
        start_node = []
        for i in range(len(self.angle_poly)):
            if self.polygon_path.contains_point((lat_new[i], lon_new[i])):
                start_node.append([lat_new[i], lon_new[i]])
                self.grid_poly.append([lat_new[i], lon_new[i]])
                self.counter_grid = self.counter_grid + 1

        self.counter_plot = self.counter_plot + 1
        plt.figure(figsize = (10, 10))
        temp1 = np.array(self.grid_poly)
        plt.plot(temp1[:, 1], temp1[:, 0], 'k.')
        plt.plot(loc[1], loc[0], 'bx')
        plt.plot(self.polygon[:, 1], self.polygon[:, 0], 'r-')
        plt.xlabel("Lon [deg]")
        plt.ylabel("Lat [deg]")
        plt.title("Step No. {:04d}, added {:1d} new points".format(self.counter_plot, 6))
        plt.savefig(self.figpath + "I_{:04d}.png".format(self.counter_plot))
        plt.close("all")
        NODE_start = Node(len(start_node), start_node, loc)
        Subnodes = self.getSubnode(NODE_start)

    def getSubnode(self, node):

        if self.counter_grid > self.pointsPr: # stopping criterion to end the recursion
            return Node(0, [], node.node_loc)

        print(self.counter_grid)

        for i in range(node.subnode_len): # loop through all the subnodes
            subsubnode = []
            length_new = 0
            lat_subsubnode, lon_subsubnode = self.getNewPoints(node.subnode_loc[i]) # generate candidates location

            for j in range(len(self.angle_poly)):
                if self.polygon_path.contains_point((lat_subsubnode[j], lon_subsubnode[j])):
                    testRevisit = self.revisit([lat_subsubnode[j], lon_subsubnode[j]])
                    if not testRevisit[0]:
                        subsubnode.append([lat_subsubnode[j], lon_subsubnode[j]])
                        self.grid_poly.append([lat_subsubnode[j], lon_subsubnode[j]])
                        self.counter_grid = self.counter_grid + 1
                        length_new = length_new + 1
            if len(subsubnode) > 0:
                self.counter_plot = self.counter_plot + 1
                plt.figure(figsize = (10, 10))
                temp1 = np.array(self.grid_poly)
                plt.plot(temp1[:, 1], temp1[:, 0], 'k.')
                plt.plot(temp1[-length_new:][:, 1], temp1[-length_new:][:, 0], 'g.')
                plt.plot(node.node_loc[1], node.node_loc[0], 'bx')
                plt.plot(self.polygon[:, 1], self.polygon[:, 0], 'r-')
                plt.xlabel("Lon [deg]")
                plt.ylabel("Lat [deg]")
                plt.title("Step No. {:04d}, added {:1d} new points".format(self.counter_plot, length_new))
                plt.savefig(self.figpath + "I_{:04d}.png".format(self.counter_plot))
                plt.close("all")
                SUBNODE = Node(len(subsubnode), subsubnode, node.subnode_loc[i])
                self.getSubnode(SUBNODE)
            else:
                return Node(0, [], node.subnode_loc[i])
        return Node(0, [], node.node_loc)

    def getPolygonArea(self):
        area = 0
        prev = self.polygon[-1]
        for i in range(self.polygon.shape[0]):
            now = self.polygon[i]
            xnow, ynow = GridPoly.latlon2xy(now[0], now[1], self.lat_origin, self.lon_origin)
            xpre, ypre = GridPoly.latlon2xy(prev[0], prev[1], self.lat_origin, self.lon_origin)
            area += xnow * ypre - ynow * xpre
            prev = now
        self.PolyArea = area / 2
        print("Area: ", self.PolyArea / 1e6, " km2")
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
                    [41.151651,-8.688966],
                    [41.154048,-8.690331]])

a = GridPoly(polygon)
a.getStart([41.1375, -8.6875])
os.system("say finished")
# a = Grid()
# a.checkGridCoord()
# a.checkBox()


#%%
plt.plot(a.polygon[:, 1], a.polygon[:, 0], 'k-.')
plt.plot(41.1375, -86875, )
plt.show()

#%%
class GridTest(Node):

    distance = 50 # distance between
    polygon = None
    grid = []
    counter = 0
    cnt = 0
    figpath = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Porto/Setup/Grid/fig/"
    x = []
    y = []

    def __init__(self):
        pass

    def setPolyGon(self, polygon):
        self.polygon = polygon

    def setup(self):
        self.polygon_path = mplPath.Path(self.polygon)
        self.theta = self.deg2rad(np.arange(0, 6) * 60)

    def deg2rad(self, deg):
        return deg / 180 * np.pi

    def rad2deg(self, rad):
        return rad / np.pi * 180

    def revisit(self, loc):
        temp = np.array(self.grid)
        if len(self.grid):
            dist = np.min(np.sqrt((temp[:, 0] - loc[0]) ** 2 + (temp[:, 1] - loc[1]) ** 2))
            ind = np.argmin(np.sqrt((temp[:, 0] - loc[0]) ** 2 + (temp[:, 1] - loc[1]) ** 2))
            if dist <= 10:
                return [True, ind]
            else:
                return [False, []]
        else:
            return [False, []]

    def getSubnode(self, node):
        loc_self = node.loc_self
        if self.cnt > 500:
            return Node(0, [], loc_self)
        print(self.cnt)
        for ii in range(node.len_subNode):
            subnode = []
            length = 0
            xnew = node.loc_subNode[ii][0] + self.distance * np.cos(self.theta)
            ynew = node.loc_subNode[ii][1] + self.distance * np.sin(self.theta)
            for j in range(len(xnew)):
                if self.polygon_path.contains_point((xnew[j], ynew[j])):
                    testRevisit = self.revisit([xnew[j], ynew[j]])
                    if not testRevisit[0]:
                        subnode.append([xnew[j], ynew[j]])
                        self.grid.append([xnew[j], ynew[j]])
                        self.cnt = self.cnt + 1
                        length = length + 1
            if len(subnode) > 0:
                self.counter = self.counter + 1
                plt.figure()
                temp1 = np.array(self.grid)
                plt.plot(temp1[:, 0], temp1[:, 1], 'k.')
                plt.plot(temp1[-length:][:, 0], temp1[-length:][:, 1], 'g.')
                plt.plot(node.loc_subNode[ii][0], node.loc_subNode[ii][1], 'bx')
                plt.plot(self.polygon[:, 0], self.polygon[:, 1], 'r-')
                plt.xlim([-10, 1010])
                plt.ylim([-10, 1010])
                plt.savefig(self.figpath + "I_{:04d}.png".format(self.counter))
                plt.close("all")
                Subnode = Node(len(subnode), subnode, node.loc_subNode[ii])
                self.getSubnode(Subnode)
            else:
                return Node(0, [], node.loc_subNode[ii])
        return Node(0, [], loc_self)

    def getStart(self, loc):
        xnew = loc[0] + self.distance * np.cos(self.theta)
        ynew = loc[1] + self.distance * np.sin(self.theta)
        node = []
        for i in range(len(xnew)):
            if self.polygon_path.contains_point((xnew[i], ynew[i])):
                testRevisit = self.revisit([xnew[i], ynew[i]])
                if not testRevisit[0]:
                    node.append([xnew[i], ynew[i]])
                    self.grid.append([xnew[i], ynew[i]])
                    self.cnt = self.cnt + 1

        self.counter = self.counter + 1
        plt.figure()
        temp1 = np.array(self.grid)
        plt.plot(temp1[:, 0], temp1[:, 1], 'k.')
        plt.plot(loc[0], loc[1], 'bx')
        plt.plot(self.polygon[:, 0], self.polygon[:, 1], 'r-')
        plt.xlim([-10, 1010])
        plt.ylim([-10, 1010])
        plt.savefig(self.figpath + "I_{:04d}.png".format(self.counter))
        plt.close("all")
        NODE = Node(len(node), node, loc)
        subnode = self.getSubnode(NODE)
        print(NODE.loc_subNode)
        print(NODE.len_subNode)

    def getCandidates(self, loc):
        polygon_path = mplPath.Path(self.polygon)
        theta = self.deg2rad(np.arange(0, 6) * 60)
        xnew = loc[0] + self.distance * np.cos(theta)
        ynew = loc[1] + self.distance * np.sin(theta)
        length = 0
        self.New = False
        for i in range(len(xnew)):
            if polygon_path.contains_point((xnew[i], ynew[i])):
                testRevisit = self.revisit([xnew[i], ynew[i]])
                if testRevisit[0]:
                    print(xnew[i], ynew[i])
                    for i in range(len(testRevisit[1])):
                        print(self.grid[testRevisit[1][i]])
                else:
                    self.grid.append([xnew[i], ynew[i]])
                    self.cnt = self.cnt + 1
                    length = length + 1
                    self.New = True
            else:
                pass

        if self.cnt>=1000:
            return self.grid

        if self.New:
            self.counter = self.counter + 1
            plt.figure()
            temp1 = np.array(self.grid)
            plt.plot(temp1[:, 0], temp1[:, 1], 'k.')
            plt.plot(temp1[-length:][:, 0], temp1[-length:][:, 1], 'g.')
            plt.plot(self.polygon[:, 0], self.polygon[:, 1], 'r-')
            plt.xlim([-10, 1010])
            plt.ylim([-10, 1010])
            plt.savefig(self.figpath + "I_{:04d}.png".format(self.counter))
            plt.close("all")
            ind = np.random.randint(0, length)
            return self.getCandidates(self.grid[-length:][ind])
        else:
            return self.getCandidates(self.grid[np.random.randint(0, len(self.grid))])


    def getGrid(self, loc):
        self.grid = np.append(self.grid, np.array(self.getCandidates(loc)).reshape(-1, 2), axis = 0)
        # while True:
            # xnew, ynew = self.getCandidates(loc)

coord_polygon = np.array([[0, 0],
                          [0, 1000],
                          [1000, 1000],
                          [1000, 0]])

m = [0, 0]
a = GridTest()
a.setPolyGon(coord_polygon)
a.setup()
a.getStart(m)
# a.getGrid(m)
print(a.polygon)
# os.system("say finished")

# plt.plot(a.grid[:, 0], a.grid[:, 1], 'k.')
# plt.show()

#%%
a = [[1.5, 2.5], [3.523, 4.25], [5.66, 7.5]]

def revisit(loc):
    temp = np.array(a)
    b = np.where((temp[:, 0] == loc[0]) & (temp[:, 1] == loc[1]))[0]
    if len(b):
        return [True, b]
    else:
        return [False, b]

t = [3.523, 4.25]
b = revisit(t)

print(a[b[1][0]])

def int_all(val):
    return int(val)
a = np.random.rand(10, 1)
b = list(map(int_all, a))
print(a)
print(b)
#%%
a = np.random.rand(10, 2)
b = [.5, .5]
t = np.min((a[:, 0] - b[0]) ** 2 + (a[:, 1] - b[1]) ** 2)
print(a)
ind = np.argmin((a[:, 0] - b[0]) ** 2 + (a[:, 1] - b[1]) ** 2)
plt.plot(a[:, 0], a[:, 1], 'k.')
plt.plot(b[0], b[1], 'bx')
plt.plot(a[ind, 0], a[ind, 1], 'ro')
plt.show()


#%%
def xy2latlon(x, y, lat_origin, lon_origin):
    lat = lat_origin + GridPoly.rad2deg(x * np.pi * 2.0 / GridPoly.circumference)
    lon = lon_origin + GridPoly.rad2deg(y * np.pi * 2.0 / (GridPoly.circumference * np.cos(GridPoly.deg2rad(lat))))
    return lat, lon
x = np.arange(10)
y = np.arange(10)
lat_origin = 10
lon_origin = 10
lat, lon = xy2latlon(x, y, lat_origin, lon_origin)
print(lat, lon)

