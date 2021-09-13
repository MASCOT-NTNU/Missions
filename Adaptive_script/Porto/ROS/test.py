import rospy
import numpy as np
from scipy.stats import mvn, norm
from auv_handler import AuvHandler
import imc_ros_interface
from imc_ros_interface.msg import Temperature, Salinity, EstimatedState
from Grid import GridPoly
from GP import GP_Poly
import time
import os
from datetime import datetime

class AUV(GridPoly):
    def __init__(self):
        GridPoly.__init__(self)
        self.node_name = 'MASCOT'
        rospy.init_node(self.node_name, anonymous=True)
        self.rate = rospy.Rate(1)  # 1Hz
        self.auv_handler = AuvHandler(self.node_name, "MASCOT")

        rospy.Subscriber("/Vehicle/Out/Temperature_filtered", Temperature, self.TemperatureCB)
        rospy.Subscriber("/Vehicle/Out/Salinity_filtered", Salinity, self.SalinityCB)
        rospy.Subscriber("/Vehicle/Out/EstimatedState_filtered", EstimatedState, self.EstimatedStateCB)

        self.speed = 1.6  # m/s
        self.depth = 0.0  # meters
        self.last_state = "unavailable"
        self.rate.sleep()
        self.init = True
        self.currentTemperature = 0.0
        self.currentSalinity = 0.0
        self.vehicle_pos = [0, 0, 0]
        self.surfacing = False
        self.surfacing_time = 25 # surface time, [sec]

    def TemperatureCB(self, msg):
        self.currentTemperature = msg.value.data

    def SalinityCB(self, msg):
        self.currentSalinity = msg.value.data

    def EstimatedStateCB(self, msg):
        offset_north = msg.lat.data - self.deg2rad(self.lat_origin)
        offset_east = msg.lon.data - self.deg2rad(self.lon_origin)
        N = offset_north * self.circumference / (2.0 * np.pi)
        E = offset_east * self.circumference * np.cos(self.deg2rad(self.lat_origin)) / (2.0 * np.pi)
        D = msg.z.data
        self.vehicle_pos = [N, E, D]

class DataAssimilator(GP_Poly):
    salinity = []
    temperature = []
    path = []
    timestamp = []
    def __init__(self):
        GP_Poly.__init__(self)
        print("Data collector is initialised correctly")

    def append_salinity(self, value):
        DataAssimilator.salinity.append(value)

    def append_temperature(self, value):
        DataAssimilator.temperature.append(value)

    def append_path(self, value):
        DataAssimilator.path.append(value)

    def append_timestamp(self, value):
        DataAssimilator.timestamp.append(value)

    def initialiseDataPath(self):
        self.date_string = datetime.now().strftime("%Y_%m%d_%H%M")
        self.data_path_mission = "Data/" + self.date_string + "/"
        if not os.path.exists(self.data_path_mission):
            print("New data path is created: ", self.data_path_mission)
            os.mkdir(self.data_path_mission)
        else:
            print("Folder is already existing, no need to create! ")

    def save_data(self):
        self.salinity = np.array(self.salinity).reshape(-1, 1)
        self.temperature = np.array(self.temperature).reshape(-1, 1)
        self.path = np.array(self.path).reshape(-1, 3)
        self.timestamp = np.array(self.timestamp).reshape(-1, 1)

        np.savetxt(self.data_path_mission + "data_salinity.txt", self.salinity, delimiter=",")
        np.savetxt(self.data_path_mission + "data_temperature.txt", self.temperature, delimiter=",")
        np.savetxt(self.data_path_mission + "data_path.txt", self.path, delimiter=",")
        np.savetxt(self.data_path_mission + "data_timestamp.txt", self.timestamp, delimiter=",")

    def vehpos2latlon(self, x, y, lat_origin, lon_origin):
        if lat_origin <= 10:
            lat_origin = self.rad2deg(lat_origin)
            lon_origin = self.rad2deg(lon_origin)
        lat = lat_origin + self.rad2deg(x * np.pi * 2.0 / self.circumference)
        lon = lon_origin + self.rad2deg(y * np.pi * 2.0 / (self.circumference * np.cos(self.deg2rad(lat))))
        return lat, lon

class PathPlanner_Polygon(AUV, DataAssimilator):
    ind_start, ind_now, ind_pre, ind_cand, ind_next = [0, 0, 0, 0, 0]  # only use index to make sure it is working properly
    mu_cond, Sigma_cond, F = [None, None, None]  # conditional mean
    mu_prior, Sigma_prior = [None, None]  # prior mean and covariance matrix
    travelled_waypoints = None  # track how many waypoins have been explored
    data_path_waypoint = []  # save the waypoint to compare
    data_path_lat = []  # save the waypoint lat
    data_path_lon = []  # waypoint lon
    data_path_depth = []  # waypoint depth
    Total_waypoints = 100  # total number of waypoints to be explored
    counter_plot_simulation = 0  # track the plot, will be deleted
    distance_neighbours = np.sqrt(GP_Poly.distance_poly ** 2 + (GP_Poly.depth_obs[1] - GP_Poly.depth_obs[0]) ** 2)

    def __init__(self):
        AUV.__init__(self)
        DataAssimilator.__init__(self)
        print("range of neighbours: ", self.distance_neighbours)
        self.travelled_waypoints = 0
        self.load_prior()
        # self.move_to_starting_loc()
        # self.run()

    def load_prior(self):
        self.lat_loc = self.lat_selected.reshape(-1, 1)  # select the top layer for simulation
        self.lon_loc = self.lon_selected.reshape(-1, 1)
        self.depth_loc = self.depth_selected.reshape(-1, 1)
        self.salinity_loc = self.salinity_selected.reshape(-1, 1)
        self.mu_prior = self.salinity_loc
        self.Sigma_prior = self.Sigma_sal
        print("Sigma_prior: ", self.Sigma_prior.shape)
        print("Sigma sal: ", self.Sigma_sal.shape)
        self.mu_real = self.mu_prior + np.linalg.cholesky(self.Sigma_prior) @ np.random.randn(
            len(self.mu_prior)).reshape(-1, 1)
        self.mu_cond = self.mu_prior
        self.Sigma_cond = self.Sigma_prior
        self.N = len(self.mu_prior)
        print("Prior for salinity is loaded correctly!!!")
        print("mu_prior shape is: ", self.mu_prior.shape)
        print("Sigma_prior shape is: ", self.Sigma_prior.shape)
        print("mu_cond shape is: ", self.mu_cond.shape)
        print("Sigma_cond shape is: ", self.Sigma_cond.shape)
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
        self.auv_handler.setWaypoint(self.deg2rad(self.lat_loc[self.ind_start]), self.deg2rad(self.lon_loc[self.ind_start]), self.depth_loc[self.ind_start])
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
        print("vec1 :", vec1)
        for i in range(len(self.ind_cand)):
            if self.ind_cand[i] != self.ind_now:
                dx2, dy2 = self.latlon2xy(self.lat_loc[self.ind_cand[i]], self.lon_loc[self.ind_cand[i]],
                                          self.lat_loc[self.ind_now], self.lon_loc[self.ind_now])
                dz2 = self.depth_loc[self.ind_cand[i]] - self.depth_loc[self.ind_now]
                vec2 = np.array([dx2, dy2, dz2]).squeeze()
                print("vec2: ", vec2)
                if np.dot(vec1, vec2) > 0:
                    print("Product: ", np.dot(vec1, vec2))
                    id.append(self.ind_cand[i])
                    lat_cand_plot.append(self.lat_loc[self.ind_cand[i]])
                    lon_cand_plot.append(self.lon_loc[self.ind_cand[i]])
                    depth_cand_plot.append(self.depth_loc[self.ind_cand[i]])
                    print("The candidate location: ", self.ind_cand[i], self.ind_now)
                    print("Candloc: ", [self.lat_loc[self.ind_cand[i]], self.lon_loc[self.ind_cand[i]]])
                    print("NowLoc: ", [self.lat_loc[self.ind_now], self.lon_loc[self.ind_now]])
        print("Before uniquing: ", id)
        id = np.unique(np.array(id))
        print("After uniquing: ", id)
        self.ind_cand = id
        M = len(id)
        eibv = []
        for k in range(M):
            F = np.zeros([1, self.N])
            F[0, id[k]] = True
            eibv.append(self.EIBV_1D(self.Threshold_S, mu, Sig, F, self.R_sal))
        t2 = time.time()

        if len(eibv) == 0:  # in case it is in the corner and not found any valid candidate locations
            print("No valid candidates found: ")
            self.ind_next = np.abs(
                self.EP_1D(mu, Sig, self.Threshold_S) - .5).argmin()  # if not found next, use the other one
        else:
            print("The EIBV for the candidate location: ", np.array(eibv))
            self.ind_next = self.ind_cand[np.argmin(np.array(eibv))]
        print("ind_next: ", self.ind_next)

        self.data_path_lat.append(self.lat_loc[self.ind_next])
        self.data_path_lon.append(self.lon_loc[self.ind_next])
        self.data_path_depth.append(self.depth_loc[self.ind_next])
        print("Finding next waypoint takes: ", t2 - t1)
        self.updateF(self.ind_next)

    def updateWaypoint(self):
        # Since the accurate location of lat lon might have numerical problem for selecting the candidate location
        # print("Before updating: ")
        # print("ind pre: ", self.ind_pre)
        # print("ind now: ", self.ind_now)
        # print("ind next: ", self.ind_next)
        self.ind_pre = self.ind_now
        self.ind_now = self.ind_next
        # print("After updating: ")
        # print("ind pre: ", self.ind_pre)
        # print("ind now: ", self.ind_now)
        # print("ind next: ", self.ind_next)

    def run(self):
        while not rospy.is_shutdown():
            if self.init:
                self.append_salinity(self.currentSalinity)
                self.append_temperature(self.currentTemperature)
                lat_temp, lon_temp = self.vehpos2latlon(self.vehicle_pos[0], self.vehicle_pos[1], self.lat_origin,
                                                        self.lon_origin)
                self.append_path([lat_temp, lon_temp, self.vehicle_pos[2]])
                print(self.auv_handler.getState())
                if self.auv_handler.getState() == "waiting" and self.last_state != "waiting":
                    print("Arrived the current location")
                    print(self.salinity[-10:])
                    sal_sampled = np.mean(self.salinity[-10:])  # take the past ten samples and average
                    print(sal_sampled)
                    self.GPupd(sal_sampled)  # update the field when it arrives the specified location
                    self.find_candidates_loc()
                    self.find_next_EIBV_1D(self.mu_cond, self.Sigma_cond)
                    self.updateWaypoint()
                    self.travelled_waypoints += 1

                    # Move to the next waypoint
                    self.auv_handler.setWaypoint(self.deg2rad(self.lat_loc[self.ind_next]), self.deg2rad(self.lon_loc[self.ind_next]),
                                                 self.depth_loc[self.ind_next])

                    if self.travelled_waypoints >= self.Total_waypoints:
                        rospy.signal_shutdown("Mission completed!!!")
                self.last_state = self.auv_handler.getState()
                self.auv_handler.spin()
            self.rate.sleep()

if __name__ == "__main__":
    # a = PathPlanner()
    a = PathPlanner_Polygon()
    print("Mission complete!!!")








