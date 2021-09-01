
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


class GridTest:

    distance = 45 / 1000 # distance between
    polygon = None
    grid = np.empty((0, 2))
    counter = 0
    cnt = 0
    figpath = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Porto/Setup/Grid/fig/"
    x = []
    y = []

    def __init__(self):
        pass

    def setPolyGon(self, polygon):
        self.polygon = polygon

    def deg2rad(self, deg):
        return deg / 180 * np.pi

    def rad2deg(self, rad):
        return rad / np.pi * 180

    def isInGrid(self, loc):
        x, y = loc
        for i in range(self.grid.shape[0]):
            if x == np.around(self.grid[i, 0], decimals=4) and y == np.around(self.grid[i, 1], decimals=4):
                return False
            else:
                return True

    def getCandidates(self, xnow, ynow):

        polygon_path = mplPath.Path(self.polygon)
        theta = self.deg2rad(np.arange(0, 6) * 60)
        xnew = np.around(xnow + self.distance * np.cos(theta), decimals=4)
        ynew = np.around(ynow + self.distance * np.sin(theta), decimals=4)
        print(xnew, ynew)

        for i in range(len(xnew)):

            if polygon_path.contains_point((xnew[i], ynew[i])):
                # if
                    self.grid = np.append(self.grid, np.array([xnew[i], ynew[i]]).reshape(1, -1), axis = 0)
                    counter_in = counter_in + 1
                    self.cnt = self.cnt + 1
                # else:
                #     pass
            else:
                counter_out = counter_out + 1

        if self.cnt>=1000:
            return self.grid

        print(self.cnt)
        # print(self.isInGrid(loc))
        if grid_temp:
            g = np.array(grid_temp).reshape(-1, 2)
            self.counter = self.counter + 1
            plt.figure()
            plt.plot(self.grid[:, 0], self.grid[:, 1], 'k.')
            plt.plot(g[:, 0], g[:, 1], 'r.')
            plt.plot(self.polygon[:, 0], self.polygon[:, 1], 'r-')
            plt.xlim([-0.2, 1.2])
            plt.ylim([-0.2, 1.2])
            plt.savefig(self.figpath + "I_{:04d}.png".format(self.counter))
            plt.close("all")
            return self.getCandidates(grid_temp[np.random.randint(0, len(grid_temp))])
        else:
            return self.getCandidates(self.grid[np.random.randint(0, self.grid.shape[0])])


    def getGrid(self, loc):
        self.grid = np.append(self.grid, np.array(self.getCandidates(loc)).reshape(-1, 2), axis = 0)
        # while True:
            # xnew, ynew = self.getCandidates(loc)

coord_polygon = np.array([[0, 0],
                          [0, 1],
                          [1, 1],
                          [1, 0]])

poly_path = mplPath.Path(coord_polygon)
m = [0.95, 0.95]
a = GridTest()
a.setPolyGon(coord_polygon)
a.getGrid(m)
print(a.polygon)
# os.system("say finished")

# plt.plot(a.grid[:, 0], a.grid[:, 1], 'k.')
# plt.show()



