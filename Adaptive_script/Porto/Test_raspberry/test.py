import rospy
import numpy as np
from scipy.stats import mvn, norm
from auv_handler import AuvHandler

import imc_ros_interface
from imc_ros_interface.msg import Temperature, Salinity, EstimatedState
from Grid import GridPoly
from GP import GP_Poly
import time

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
    data_salinity = []
    data_temperature = []
    data_path_waypoint = []
    data_timestamp = []
    def __init__(self):
        GP_Poly.__init__(self)
        print("Data collector is initialised correctly")

    def append_salinity(self, value):
        DataAssimilator.data_salinity.append(value)

    def append_temperature(self, value):
        DataAssimilator.data_temperature.append(value)

    def append_path(self, value):
        DataAssimilator.data_path_waypoint.append(value)

    def append_timestamp(self, value):
        DataAssimilator.data_timestamp.append(value)

    def save_data(self):
        self.data_salinity = np.array(self.data_salinity).reshape(-1, 1)
        self.data_temperature = np.array(self.data_temperature).reshape(-1, 1)
        self.data_path_waypoint = np.array(self.data_path_waypoint).reshape(-1, 3)
        self.data_timestamp = np.array(self.data_timestamp).reshape(-1, 1)
        np.savetxt("data_salinity.txt", self.data_salinity, delimiter=",")
        np.savetxt("data_temperature.txt", self.data_temperature, delimiter=",")
        np.savetxt("data_path.txt", self.data_path, delimiter=",")
        np.savetxt("data_timestamp.txt", self.data_timestamp, delimiter=",")

    def vehpos2latlon(self, x, y, lat_origin, lon_origin):
        if lat_origin <= 10:
            lat_origin = self.rad2deg(lat_origin)
            lon_origin = self.rad2deg(lon_origin)
        lat = lat_origin + self.rad2deg(x * np.pi * 2.0 / self.circumference)
        lon = lon_origin + self.rad2deg(y * np.pi * 2.0 / (self.circumference * np.cos(self.deg2rad(lat))))
        return lat, lon


class PathPlanner_Polygon(AUV, DataAssimilator):
    lat_start, lon_start, depth_start = [None, None, None]
    lat_now, lon_now, depth_now = [None, None, None]
    lat_pre, lon_pre, depth_pre = [None, None, None]
    lat_cand, lon_cand, depth_cand = [None, None, None]
    lat_next, lon_next, depth_next = [None, None, None]
    mu_cond, Sigma_cond, F = [None, None, None]
    mu_prior, Sigma_prior = [None, None]
    travelled_waypoints = None
    Total_waypoints = 10
    distance_neighbours = np.sqrt(GP_Poly.distance_poly ** 2 + (GP_Poly.depth_obs[1] - GP_Poly.depth_obs[0]) ** 2)

    def __init__(self):
        AUV.__init__(self)
        DataAssimilator.__init__(self)
        print("AUV is set up correctly")
        print("range of neighbours: ", self.distance_neighbours)
        self.travelled_waypoints = 0
        self.load_prior()
        self.move_to_starting_loc()
        self.run()

    def load_prior(self):
        print("Prior for salinity is loaded correctly!!!")
        self.mu_prior = self.salinity_layers_ave.reshape(-1, 1)
        self.Sigma_prior = self.Sigma_sal
        self.mu_cond = self.mu_prior
        self.Sigma_cond = self.Sigma_prior
        self.lat_loc = self.lat_layers.reshape(-1, 1)
        self.lon_loc = self.lon_layers.reshape(-1, 1)
        self.depth_loc = self.depth_layers_ave.reshape(-1, 1)
        self.N = len(self.lat_loc)
        print("mu_prior shape is: ", self.mu_prior.shape)
        print("Sigma_prior shape is: ", self.Sigma_prior.shape)
        print("mu_cond shape is: ", self.mu_cond.shape)
        print("Sigma_cond shape is: ", self.Sigma_cond.shape)
        print("lat loc: ", self.lat_loc.shape)
        print("lon loc: ", self.lon_loc.shape)
        print("depth loc: ", self.depth_loc.shape)
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
        ep_criterion = 0.5 # excursion probability close to 0.5
        ind = (np.abs(EP_Prior - ep_criterion)).argmin()
        self.lat_start = self.lat_loc[ind]
        self.lon_start = self.lon_loc[ind]
        self.depth_start = self.depth_loc[ind]
        self.updateF(ind)
        self.lat_next, self.lon_next, self.depth_next = self.lat_start, self.lon_start, self.depth_start
        self.auv_handler.setWaypoint(self.deg2rad(self.lat_next), self.deg2rad(self.lon_next), self.depth_next)
        print("Next waypoint:", self.lat_next, self.lon_next, self.depth_next)

        self.updateWaypoint()
        self.updateWaypoint()

    def find_candidates_loc(self):
        '''
        find the candidates location based on distance coverage
        '''
        delta_x, delta_y = self.latlon2xy(self.lat_loc, self.lon_loc, self.lat_now, self.lon_now)
        delta_depth = self.depth_loc - self.depth_now
        self.distance_total = np.sqrt(delta_x ** 2 + delta_y ** 2 + delta_depth ** 2)
        self.ind_cand = np.where(self.distance_total <= self.distance_neighbours + 1)[0] # 1 here refers to the tolerance
        print("number of candidates: ", len(self.ind_cand))
        self.lat_cand = self.lat_loc[self.ind_cand]
        self.lon_cand = self.lon_loc[self.ind_cand]
        self.depth_cand = self.depth_loc[self.ind_cand]

    def GPupd(self, y_sampled):
        print("Updating the field...")
        t1 = time.time()
        C = self.F @ self.Sigma_cond @ self.F.T + self.R_sal
        self.mu_cond = self.mu_cond + self.Sigma_cond @ self.F.T @ np.linalg.solve(C, (y_sampled - self.F @ self.mu_cond))
        self.Sigma_cond = self.Sigma_cond - self.Sigma_cond @ self.F.T @ np.linalg.solve(C, self.F @ self.Sigma_cond)
        t2 = time.time()
        print("Updating takes: ", t2 - t1)

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
        # filter the candidates to smooth the AUV path planning
        print("Trying to find the next waypoint...")
        t1 = time.time()
        id = []
        dlat1 = self.lat_now - self.lat_pre
        dlon1 = self.lon_now - self.lon_pre
        ddepth1 = self.depth_now - self.depth_pre
        vec1 = np.array([dlat1, dlon1, ddepth1]).reshape(1, -1)
        for i in range(len(self.lat_cand)):
            if self.lat_cand[i] == self.lat_now and self.lon_cand[i] == self.lon_now and self.depth_cand[i] == self.depth_now:
                continue
            dlat2 = self.lat_cand[i] - self.lat_now
            dlon2 = self.lon_cand[i] - self.lon_now
            ddepth2 = self.depth_cand[i] - self.depth_now
            vec2 = np.array([dlat2, dlon2, ddepth2]).reshape(-1, 1)
            if np.dot(vec1, vec2) >= 0:
                id.append(self.ind_cand[i])
            else:
                continue
        id = np.unique(np.array(id))
        print(id)

        M = len(id)
        eibv = []
        for k in range(M):
            F = np.zeros([1, self.N])
            F[0, id[k]] = True
            eibv.append(self.EIBV_1D(self.Threshold_S, mu, Sig, F, self.R_sal))
            print(eibv)
        ind_desired = np.argmin(np.array(eibv))
        print(ind_desired)
        t2 = time.time()
        self.lat_next = self.lat_cand[ind_desired]
        self.lon_next = self.lon_cand[ind_desired]
        self.depth_next = self.depth_cand[ind_desired]
        print("Found next waypoint: ", self.lat_next, self.lon_next, self.depth_next)
        print("Finding next waypoint takes: ", t2 - t1)
        self.updateF(id[ind_desired])

    def updateWaypoint(self):
        self.lat_pre, self.lon_pre, self.depth_pre = self.lat_now, self.lon_now, self.depth_now
        self.lat_now, self.lon_now, self.depth_now = self.lat_next, self.lon_next, self.depth_next

    def run(self):
        while not rospy.is_shutdown():
            if self.init:
                self.append_salinity(self.currentSalinity)
                self.append_temperature(self.currentTemperature)
                lat_temp, lon_temp = self.vehpos2latlon(self.vehicle_pos[0], self.vehicle_pos[1], self.lat_origin, self.lon_origin)
                self.append_path([lat_temp, lon_temp, self.vehicle_pos[2]])
                print(self.auv_handler.getState())
                print(self.last_state)
                if self.auv_handler.getState() == "waiting" and self.last_state != "waiting":
                    print("Arrived the current location")
                    print(self.data_salinity[-10:])
                    sal_sampled = np.mean(self.data_salinity[-10:])  # take the past ten samples and average
                    print(sal_sampled)
                    self.GPupd(sal_sampled) # update the field when it arrives the specified location
                    self.find_candidates_loc()
                    self.find_next_EIBV_1D(self.mu_cond, self.Sigma_cond)
                    self.updateWaypoint()
                    self.travelled_waypoints += 1
                    
                    print("Next waypoint:", self.lat_next, self.lon_next, self.depth_next)

                    # Move to the next waypoint
                    self.auv_handler.setWaypoint(self.deg2rad(self.lat_next), self.deg2rad(self.lon_next), self.depth_next)
                    print("sey waypoint correctly!")

                    if self.travelled_waypoints >= self.Total_waypoints:
                        rospy.signal_shutdown("Mission completed!!!")
                self.last_state = self.auv_handler.getState()
                self.auv_handler.spin()
            self.rate.sleep()
    pass


if __name__ == "__main__":
    # a = PathPlanner()
    a = PathPlanner_Polygon()
    print("Mission complete!!!")








