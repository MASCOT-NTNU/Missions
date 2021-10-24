#! /usr/bin/env python3
__author__ = "Yaolin Ge"
__copyright__ = "Copyright 2021, The MASCOT Project, NTNU (https://wiki.math.ntnu.no/mascot)"
__credits__ = ["Yaolin Ge"]
__license__ = "MIT"
__version__ = "1.0.1"
__maintainer__ = "Yaolin Ge"
__email__ = "yaolin.ge@ntnu.no"
__status__ = "UnderDevelopment"

'''
Objective: generate the desired circular polygon based on the corrected prior
'''
from usr_func import *
import matplotlib.pyplot as plt

class PolygonCircle:

    lat_center, lon_center = 41.061874, -8.650977 # center of the circular polygon
    radius = 1000 # radius of the polygon
    npoints = 200 # boundary points

    def __init__(self):
        print("Circular Polygon Generator is initialised successfully! ")
        # self.getxy()
        # self.getCircle()
        # self.checkCircle()

    def load_global_path(self):
        print("Now it will load the global path.")
        self.path_global = open("path_global.txt", 'r').read()
        print("global path is set up successfully!")
        print(self.path_global)

    def setCentre(self, lat, lon):
        print("The previous centre is: ", self.lat_center, self.lon_center)
        self.lat_center, self.lon_center = lat, lon
        print("The updated circular polygon center is: ", self.lat_center, self.lon_center)

    def getCircularPolygon(self, lat, lon, radius, npoints = 200):
        self.setCentre(lat, lon)
        self.radius = radius
        self.npoints = npoints
        print("Polygon will be generated based on the following parameters!")
        print("Polygon Centre: ", self.lat_center, self.lon_center)
        print("Polygon radius: ", self.radius)
        print("Polygon shape: ", self.npoints)
        self.getxy()
        self.getCircle()
        print("Circular polygon is generated successfully!")
        print("Now I will save it...")
        np.savetxt(self.path_global + "/Config/polygon.txt", np.hstack((self.lat_circle.reshape(-1, 1),
                                                                        self.lon_circle.reshape(-1, 1))), delimiter=", ")
        print("polygon.txt is saved successfully! ", self.path_global + '/Config/polygon.txt')

    def getxy(self):
        self.theta = np.linspace(0, np.pi * 2, self.npoints)
        self.x = self.radius * np.sin(self.theta)
        self.y = self.radius * np.cos(self.theta)

    def getCircle(self):
        self.lat_circle, self.lon_circle = xy2latlon(self.x, self.y, self.lat_center, self.lon_center)

    def checkCircle(self):
        plt.figure(figsize = (5, 5))
        plt.plot(self.lon_circle, self.lat_circle, 'k.')
        plt.show()

if __name__ == "__main__":
    a = PolygonCircle()
    a.getCircularPolygon(0, 0, 1500)
    a.checkCircle()







