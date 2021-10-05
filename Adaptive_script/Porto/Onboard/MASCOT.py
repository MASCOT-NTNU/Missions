# ! /usr/bin/env python3
__author__ = "Yaolin Ge"
__copyright__ = "Copyright 2021, The MASCOT Project, NTNU (https://wiki.math.ntnu.no/mascot)"
__credits__ = ["Yaolin Ge"]
__license__ = "MIT"
__version__ = "1.0.1"
__maintainer__ = "Yaolin Ge"
__email__ = "yaolin.ge@ntnu.no"
__status__ = "UnderDevelopment"

import rospy
import numpy as np
from scipy.stats import mvn, norm
from auv_handler import AuvHandler
import imc_ros_interface
from imc_ros_interface.msg import Temperature, Salinity, EstimatedState, Sms
import time
import os
from pathlib import Path
from datetime import datetime


class AUV:
    circumference = 40075000  # circumference of the earth, [m]
    lat_origin, lon_origin = 41.061874, -8.650977  # origin location
    distance_poly = 100  # [m], distance between two neighbouring points
    depth_obs = [-.5, -1.25, -2] # [m], distance in depth, depth to be explored
    distanceTolerance = .1  # [m], distance tolerance for the neighbouring points
    def __init__(self):
        self.node_name = 'MASCOT'
        rospy.init_node(self.node_name, anonymous=True)
        self.rate = rospy.Rate(1)  # 1Hz
        self.auv_handler = AuvHandler(self.node_name, "MASCOT")

        rospy.Subscriber("/Vehicle/Out/Temperature_filtered", Temperature, self.TemperatureCB)
        rospy.Subscriber("/Vehicle/Out/Salinity_filtered", Salinity, self.SalinityCB)
        rospy.Subscriber("/Vehicle/Out/EstimatedState_filtered", EstimatedState, self.EstimatedStateCB)

        self.speed = 1.2  # m/s
        self.depth = 0.0  # meters
        self.last_state = "unavailable"
        self.rate.sleep()
        self.init = True
        self.currentTemperature = 0.0
        self.currentSalinity = 0.0
        self.vehicle_pos = [0, 0, 0]

        self.sms_pub_ = rospy.Publisher("/IMC/In/Sms", Sms, queue_size = 10)
        self.phone_number = "+351969459285"

    def TemperatureCB(self, msg):
        self.currentTemperature = msg.value.data

    def SalinityCB(self, msg):
        self.currentSalinity = msg.value.data

    def EstimatedStateCB(self, msg):
        offset_north = msg.lat.data - AUV.deg2rad(self.lat_origin)
        offset_east = msg.lon.data - AUV.deg2rad(self.lon_origin)
        N = offset_north * self.circumference / (2.0 * np.pi)
        E = offset_east * self.circumference * np.cos(AUV.deg2rad(self.lat_origin)) / (2.0 * np.pi)
        D = msg.z.data
        self.vehicle_pos = [N, E, D]

    @staticmethod
    def deg2rad(deg):
        return deg / 180 * np.pi

    @staticmethod
    def rad2deg(rad):
        return rad / np.pi * 180

    @staticmethod
    def latlon2xy(lat, lon, lat_origin, lon_origin):
        x = AUV.deg2rad((lat - lat_origin)) / 2 / np.pi * AUV.circumference
        y = AUV.deg2rad((lon - lon_origin)) / 2 / np.pi * AUV.circumference * np.cos(AUV.deg2rad(lat))
        return x, y

    @staticmethod
    def xy2latlon(x, y, lat_origin, lon_origin):
        lat = lat_origin + AUV.rad2deg(x * np.pi * 2.0 / AUV.circumference)
        lon = lon_origin + AUV.rad2deg(y * np.pi * 2.0 / (AUV.circumference * np.cos(AUV.deg2rad(lat))))
        return lat, lon

class DataAssimilator(AUV):
    data_salinity = []
    data_temperature = []
    data_path_waypoints = []
    data_timestamp = []

    def __init__(self):
        AUV.__init__(self)
        print("Data collector is initialised correctly")

    def append_salinity(self, value):
        DataAssimilator.data_salinity.append(value)

    def append_temperature(self, value):
        DataAssimilator.data_temperature.append(value)

    def append_path(self, value):
        DataAssimilator.data_path_waypoints.append(value)

    def append_timestamp(self, value):
        DataAssimilator.data_timestamp.append(value)

    def createDataPath(self):
        self.date_string = datetime.now().strftime("%Y_%m%d_%H%M")
        self.data_path_mission = os.getcwd() + "/Data/MissionData_on" + self.date_string
        f_pre = open("filepath_missiondata.txt", 'w')
        f_pre.write(self.data_path_mission)
        f_pre.close()
        if not os.path.exists(self.data_path_mission):
            print("New data path is created: ", self.data_path_mission)
            path = Path(self.data_path_mission)
            path.mkdir(parents = True, exist_ok=True)
        else:
            print("Folder is already existing, no need to create! ")

    def save_data(self):
        self.data_salinity_saved = np.array(self.data_salinity).reshape(-1, 1)
        self.data_temperature_saved = np.array(self.data_temperature).reshape(-1, 1)
        self.data_path_waypoints_saved = np.array(self.data_path_waypoints).reshape(-1, 3)
        self.data_timestamp_saved = np.array(self.data_timestamp).reshape(-1, 1)
        np.savetxt(self.data_path_mission + "/data_salinity.txt", self.data_salinity_saved, delimiter=", ")
        np.savetxt(self.data_path_mission + "/data_temperature.txt", self.data_temperature_saved, delimiter=", ")
        np.savetxt(self.data_path_mission + "/data_path.txt", self.data_path_waypoints_saved, delimiter=", ")
        np.savetxt(self.data_path_mission + "/data_timestamp.txt", self.data_timestamp_saved, delimiter=", ")

    def vehpos2latlon(self, x, y, lat_origin, lon_origin):
        if lat_origin <= 10:
            lat_origin = AUV.rad2deg(lat_origin)
            lon_origin = AUV.rad2deg(lon_origin)
        lat = lat_origin + AUV.rad2deg(x * np.pi * 2.0 / self.circumference)
        lon = lon_origin + AUV.rad2deg(y * np.pi * 2.0 / (self.circumference * np.cos(AUV.deg2rad(lat))))
        return lat, lon


class PathPlanner_Polygon(DataAssimilator):
    ind_start, ind_now, ind_pre, ind_cand, ind_next = [0, 0, 0, 0, 0]  # only use index to make sure it is working properly
    mu_cond, Sigma_cond, F = [None, None, None]  # conditional mean
    mu_prior, Sigma_prior = [None, None]  # prior mean and covariance matrix
    travelled_waypoints = None  # track how many waypoins have been explored
    data_path_waypoint = []  # save the waypoint to compare
    data_path_lat = []  # save the waypoint lat
    data_path_lon = []  # waypoint lon
    data_path_depth = []  # waypoint depth
    Total_waypoints = 60  # total number of waypoints to be explored
    counter_plot_simulation = 0  # track the plot, will be deleted
    distance_neighbours = np.sqrt(AUV.distance_poly ** 2 + (AUV.depth_obs[1] - AUV.depth_obs[0]) ** 2)

    def __init__(self):
        DataAssimilator.__init__(self)
        print("range of neighbours: ", self.distance_neighbours)
        self.load_prior()
        self.travelled_waypoints = 0
        self.move_to_starting_loc()
        self.run()

    def load_prior(self):
        self.prior_corrected_path = "prior_corrected.txt"
        self.sigma_path = "Sigma_sal.txt"
        self.threshold_path = "Threshold_S.txt"
        self.R_sal_path = "R_sal.txt"
        self.data_prior = np.loadtxt(self.prior_corrected_path, delimiter=", ")
        self.lat_loc = self.data_prior[:, 0]
        self.lon_loc = self.data_prior[:, 1]
        self.depth_loc = self.data_prior[:, 2]
        self.salinity_prior_corrected = self.data_prior[:, -1]
        self.mu_prior = self.salinity_prior_corrected
        self.Sigma_prior = np.loadtxt(self.sigma_path, delimiter=", ")
        self.Threshold_S = np.loadtxt(self.threshold_path, delimiter=", ")
        self.R_sal = np.loadtxt(self.R_sal_path, delimiter=", ")
        self.mu_cond = self.mu_prior
        self.Sigma_cond = self.Sigma_prior
        self.N = len(self.mu_prior)
        print("Prior for salinity is loaded correctly!!!")
        print("mu_prior shape is: ", self.mu_prior.shape)
        print("Sigma_prior shape is: ", self.Sigma_prior.shape)
        print("mu_cond shape is: ", self.mu_cond.shape)
        print("Sigma_cond shape is: ", self.Sigma_cond.shape)
        print("Threshold: ", self.Threshold_S)
        print("R_sal: ", self.R_sal)
        print("N: ", self.N)

    def updateF(self, ind):
        self.F = np.zeros([1, self.N])
        self.F[0, ind] = True

    def EP_1D(self, mu, Sigma, Threshold):
        EP = np.zeros_like(mu)
        for i in range(EP.shape[0]):
            EP[i] = norm.cdf(Threshold, mu[i], Sigma[i, i])
        return EP

    def move_to_starting_loc(self):
        EP_Prior = self.EP_1D(self.mu_prior, self.Sigma_prior, self.Threshold_S)
        ep_criterion = 0.5  # excursion probability close to 0.5
        self.ind_start = (np.abs(EP_Prior - ep_criterion)).argmin()
        if self.ind_start == 0:
            self.ind_start = np.random.randint(self.N)
        self.ind_next = self.ind_start
        self.updateF(self.ind_next)
        self.data_path_lat.append(self.lat_loc[self.ind_start])
        self.data_path_lon.append(self.lon_loc[self.ind_start])
        self.auv_handler.setWaypoint(self.deg2rad(self.lat_loc[self.ind_start]), self.deg2rad(self.lon_loc[self.ind_start]), -self.depth_loc[self.ind_start])
        self.updateWaypoint()

    def find_candidates_loc(self):
        '''
        find the candidates location based on distance coverage
        '''
        delta_x, delta_y = self.latlon2xy(self.lat_loc, self.lon_loc, self.lat_loc[self.ind_now],
                                          self.lon_loc[self.ind_now])  # using the distance
        delta_z = self.depth_loc - self.depth_loc[self.ind_now]  # depth distance in z-direction
        self.distance_vector = np.sqrt(delta_x ** 2 + delta_y ** 2 + delta_z ** 2)
        self.ind_cand = np.where(self.distance_vector <= self.distance_neighbours + self.distanceTolerance)[0]

    def GPupd(self, y_sampled):
        t1 = time.time()
        C = self.F @ self.Sigma_cond @ self.F.T + self.R_sal
        self.mu_cond = self.mu_cond + self.Sigma_cond @ self.F.T @ np.linalg.solve(C,
                                                                                   (y_sampled - self.F @ self.mu_cond))
        self.Sigma_cond = self.Sigma_cond - self.Sigma_cond @ self.F.T @ np.linalg.solve(C, self.F @ self.Sigma_cond)
        t2 = time.time()

    def EIBV_1D(self, threshold, mu, Sig, F, R):
        Sigxi = Sig @ F.T @ np.linalg.solve(F @ Sig @ F.T + R, F @ Sig)
        V = Sig - Sigxi
        sa2 = np.diag(V).reshape(-1, 1)  # the corresponding variance term for each location
        IntA = 0.0
        for i in range(len(mu)):
            sn2 = sa2[i]
            m = mu[i]
            IntA = IntA + mvn.mvnun(-np.inf, threshold, m, sn2)[0] - mvn.mvnun(-np.inf, threshold, m, sn2)[0] ** 2
        return IntA

    def find_next_EIBV_1D(self, mu, Sig):
        id = []  # ind vector for containing the filtered desired candidate location
        t1 = time.time()
        dx1, dy1 = self.latlon2xy(self.lat_loc[self.ind_now], self.lon_loc[self.ind_now], self.lat_loc[self.ind_pre],
                                  self.lon_loc[self.ind_pre])
        dz1 = self.depth_loc[self.ind_now] - self.depth_loc[self.ind_pre]
        lat_cand_plot = []
        lon_cand_plot = []
        depth_cand_plot = []
        vec1 = np.array([dx1, dy1, dz1]).squeeze()
        for i in range(len(self.ind_cand)):
            if self.ind_cand[i] != self.ind_now:
                dx2, dy2 = self.latlon2xy(self.lat_loc[self.ind_cand[i]], self.lon_loc[self.ind_cand[i]],
                                          self.lat_loc[self.ind_now], self.lon_loc[self.ind_now])
                dz2 = self.depth_loc[self.ind_cand[i]] - self.depth_loc[self.ind_now]
                vec2 = np.array([dx2, dy2, dz2]).squeeze()
                if np.dot(vec1, vec2) > 0:
                    if dx2 == 0 and dy2 == 0:
                        print("Sorry, I cannot dive or float directly")
                        pass
                    else:
                        id.append(self.ind_cand[i])
                        lat_cand_plot.append(self.lat_loc[self.ind_cand[i]])
                        lon_cand_plot.append(self.lon_loc[self.ind_cand[i]])
                        depth_cand_plot.append(self.depth_loc[self.ind_cand[i]])
        id = np.unique(np.array(id))
        self.ind_cand = id
        M = len(id)
        eibv = []
        for k in range(M):
            F = np.zeros([1, self.N])
            F[0, id[k]] = True
            eibv.append(self.EIBV_1D(self.Threshold_S, mu, Sig, F, self.R_sal))
        t2 = time.time()

        if len(eibv) == 0:  # in case it is in the corner and not found any valid candidate locations
            self.ind_next = np.abs(
                self.EP_1D(mu, Sig, self.Threshold_S) - .5).argmin()  # if not found next, use the other one
        else:
            self.ind_next = self.ind_cand[np.argmin(np.array(eibv))]
        self.data_path_lat.append(self.lat_loc[self.ind_next])
        self.data_path_lon.append(self.lon_loc[self.ind_next])
        self.data_path_depth.append(self.depth_loc[self.ind_next])
        print("Finding next waypoint takes: ", t2 - t1)
        self.updateF(self.ind_next)

    def updateWaypoint(self):
        # Since the accurate location of lat lon might have numerical problem for selecting the candidate location
        self.ind_pre = self.ind_now
        self.ind_now = self.ind_next

    def send_SMS(self):
        print("Message has been sent to: ", self.phone_number)
        SMS = Sms()
        SMS.number.data = self.phone_number
        SMS.timeout.data = 60
        x_auv = self.vehicle_pos[0]
        y_auv = self.vehicle_pos[1]
        lat_auv, lon_auv = self.vehpos2latlon(x_auv, y_auv, self.lat_origin, self.lon_origin)
        SMS.contents.data = "LAUV-Xplore-1 location: " + str(lat_auv) + ", " + str(lon_auv)
        self.sms_pub_.publish(SMS)

    def send_SMS_starting_loc(self):
        print("Message has been sent to: ", self.phone_number)
        SMS = Sms()
        SMS.number.data = self.phone_number
        SMS.timeout.data = 60
        x_auv = self.vehicle_pos[0]
        y_auv = self.vehicle_pos[1]
        lat_auv, lon_auv = self.vehpos2latlon(x_auv, y_auv, self.lat_origin, self.lon_origin)
        SMS.contents.data = "I am moving to the starting location. LAUV-Xplore-1 location: " + str(lat_auv) + ", " + str(lon_auv)
        self.sms_pub_.publish(SMS)

    def send_SMS_mission_complete(self):
        print("Mission complete Message has been sent to: ", self.phone_number)
        SMS = Sms()
        SMS.number.data = self.phone_number
        SMS.timeout.data = 60
        x_auv = self.vehicle_pos[0]
        y_auv = self.vehicle_pos[1]
        lat_auv, lon_auv = self.vehpos2latlon(x_auv, y_auv, self.lat_origin, self.lon_origin)
        SMS.contents.data = "Congrats, Mission complete. LAUV-Xplore-1 location: " + str(lat_auv) + ", " + str(lon_auv)
        self.sms_pub_.publish(SMS)

    def surfacing(self, time_length):
        for i in range(time_length):
            if (i + 1) % 15 == 0:
                self.send_SMS()
            self.append_mission_data()
            self.save_mission_data()
            print("Sleep {:d} seconds".format(i))
            self.auv_handler.setWaypoint(self.deg2rad(self.lat_loc[self.ind_pre]),
                                         self.deg2rad(self.lon_loc[self.ind_pre]),
                                         0)
            self.auv_handler.spin()  # publishes the reference, stay on the surface
            self.rate.sleep()  #

    def save_mission_data(self):
        if np.around((self.t2 - self.t1), 0) % 10 == 0:
            print("Data is saved: ", self.counter_data_saved, " times")
            self.save_data()
            self.counter_data_saved = self.counter_data_saved + 1

    def append_mission_data(self):
        self.append_salinity(self.currentSalinity)
        self.append_temperature(self.currentTemperature)
        lat_temp, lon_temp = self.vehpos2latlon(self.vehicle_pos[0], self.vehicle_pos[1], self.lat_origin,
                                                self.lon_origin)
        self.append_path([lat_temp, lon_temp, self.vehicle_pos[2]])
        self.append_timestamp(datetime.now().timestamp())

    def send_starting_waypoint(self):
        if (self.t2 - self.t1) / 300 >= 1 and (self.t2 - self.t1) % 300 >= 0:
            print("Longer than 10 mins, need a long break")
            time_length = 90
            for i in range(time_length):
                if (i + 1) % 15 == 0:
                    self.send_SMS_starting_loc()
                self.append_mission_data()
                self.save_mission_data()
                print("Sleep {:d} seconds".format(i))
                self.auv_handler.setWaypoint(self.deg2rad(self.lat_loc[self.ind_start]),
                                             self.deg2rad(self.lon_loc[self.ind_start]),
                                             0)
                self.auv_handler.spin()  # publishes the reference, stay on the surface
                self.rate.sleep()  #
            self.t1 = self.t2
        self.auv_handler.setWaypoint(self.deg2rad(self.lat_loc[self.ind_start]),
                                     self.deg2rad(self.lon_loc[self.ind_start]), -self.depth_loc[self.ind_start])
        print("moving to the starting location")

    def send_next_waypoint(self):
        # Move to the next waypoint
        self.counter_waypoint = self.counter_waypoint + 1  # needs to be updated before

        if self.counter_waypoint % 3 == 0: # check whether it needs to surface
            if (self.t2 - self.t1) / 600 >= 1 and (self.t2 - self.t1) % 600 >= 0:
                print("Longer than 10 mins, need a long break")
                self.surfacing(90)  # surfacing 90 seconds after 10 mins of travelling
                self.t1 = self.t2
            else:
                print("Less than 10 mins, need a shorter break")
                self.surfacing(30) # surfacing 30 seconds

        self.auv_handler.setWaypoint(self.deg2rad(self.lat_loc[self.ind_next]),
                                     self.deg2rad(self.lon_loc[self.ind_next]),
                                     -self.depth_loc[self.ind_next])
        print("next waypoint", self.deg2rad(self.lat_loc[self.ind_next]),
                               self.deg2rad(self.lon_loc[self.ind_next]),
                               -self.depth_loc[self.ind_next])

    def run(self):
        self.createDataPath()
        self.t1 = time.time()
        self.counter_waypoint = 0
        self.counter_data_saved = 0
        while not rospy.is_shutdown():
            if self.init:
                self.t2 = time.time()
                self.append_mission_data()
                self.save_mission_data()
                print(self.auv_handler.getState())
                print("Counter waypoint: ", self.counter_waypoint)
                print("Elapsed time after surfacing: ", self.t2 - self.t1)
                if self.counter_waypoint == 0:
                    self.send_starting_waypoint()
                if self.auv_handler.getState() == "waiting" and self.last_state != "waiting":
                    print("Arrived the current location")
                    self.sal_sampled = np.mean(self.data_salinity[-10:])  # take the past ten samples and average
                    print("Sampled salinity: ", self.sal_sampled)
                    self.GPupd(self.sal_sampled)  # update the field when it arrives the specified location
                    self.find_candidates_loc()
                    self.find_next_EIBV_1D(self.mu_cond, self.Sigma_cond)
                    self.updateWaypoint()
                    self.travelled_waypoints += 1
                    if self.travelled_waypoints >= self.Total_waypoints:
                        self.send_SMS_mission_complete()
                        rospy.signal_shutdown("Mission completed!!!")
                        break
                    self.send_next_waypoint()

                self.last_state = self.auv_handler.getState()
                self.auv_handler.spin()
            self.rate.sleep()

if __name__ == "__main__":
    a = PathPlanner_Polygon()
    print("Mission complete!!!")








