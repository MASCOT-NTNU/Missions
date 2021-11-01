#!/usr/bin/env python3
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
    radius = 1400 # radius of the polygon
    npoints = 200 # boundary points

    def __init__(self):
        print("Circular Polygon Generator is initialised successfully! ")
        self.load_global_path()
        self.load_prior()
        self.find_polygon_centre()
        self.get_polygon_centre()
        self.getCircularPolygon()
        # self.checkCircle()

    def load_global_path(self):
        print("Now it will load the global path.")
        self.path_global = open("path_global.txt", 'r').read()
        print("global path is set up successfully!")
        print(self.path_global)

    def load_prior(self):
        self.prior_data_corrected = np.loadtxt(self.path_global + "/Data/Corrected/Prior_corrected.txt", delimiter=", ")
        self.lat_prior_corrected = self.prior_data_corrected[:, 0]
        self.lon_prior_corrected = self.prior_data_corrected[:, 1]
        self.depth_prior_corrected = self.prior_data_corrected[:, 2]
        self.salinity_prior_corrected = self.prior_data_corrected[:, -1]
        print("Loading prior successfully.")
        print("lat_prior: ", self.lat_prior_corrected.shape)
        print("lon_prior: ", self.lon_prior_corrected.shape)
        print("depth_prior: ", self.depth_prior_corrected.shape)
        print("salinity_prior: ", self.salinity_prior_corrected.shape)

    def getPriorIndAtLoc(self, loc):
        '''
        return the index in the prior data which corresponds to the location
        '''
        lat, lon, depth = loc
        distDepth = self.depth_prior_corrected - depth
        distLat = self.lat_prior_corrected - lat
        distLon = self.lon_prior_corrected - lon
        dist = np.sqrt(distLat ** 2 + distLon ** 2 + distDepth ** 2)
        ind_loc = np.where(dist == np.nanmin(dist))[0][0]
        return ind_loc

    def find_polygon_centre(self):
        self.path_initial_survey = np.loadtxt(self.path_global + "/Config/path_initial_survey.txt", delimiter=", ")
        self.sal_path_initial_survey = []
        for i in range(10, self.path_initial_survey.shape[0] - 10):
            self.sal_path_initial_survey.append(self.getPriorIndAtLoc([self.path_initial_survey[i, 0],
                                                                       self.path_initial_survey[i, 1], self.path_initial_survey[i, 2]]))
        self.sal_gradient = np.gradient(np.array(self.sal_path_initial_survey))
        self.ind_optimal = np.where(self.sal_gradient == np.nanmax(self.sal_gradient))[0][0]
        self.lat_centre = self.path_initial_survey[self.ind_optimal, 0] # optimal index is from the maximum gradient
        self.lon_centre = self.path_initial_survey[self.ind_optimal, 1]
        print("Saving polygon centre...")
        np.savetxt(self.path_global + "/Config/polygon_centre.txt", np.array([[self.lat_centre, self.lon_centre]]), delimiter=", ")
        print("Polygon centre is saved successfully!")

    def get_polygon_centre(self):
        print("Loading the polygon centre...")
        self.lat_centre, self.lon_centre = np.loadtxt(self.path_global + "/Config/polygon_centre.txt", delimiter = ", ")
        print("lat_centre: ", self.lat_centre)
        print("lon_centre: ", self.lon_centre)

    def getCircularPolygon(self):
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
        self.lat_circle, self.lon_circle = xy2latlon(self.x, self.y, self.lat_centre, self.lon_centre)

    def checkCircle(self):
        plt.figure(figsize = (5, 5))
        plt.plot(self.lon_circle, self.lat_circle, 'k.')
        plt.show()

if __name__ == "__main__":
    a = PolygonCircle()








