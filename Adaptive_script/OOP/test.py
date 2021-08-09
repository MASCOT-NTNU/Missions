import rospy
import numpy as np
from scipy.stats import mvn, norm
from auv_handler import AuvHandler

import imc_ros_interface
from imc_ros_interface.msg import Temperature, Salinity, EstimatedState
from Grid import Grid

class AUV(GP):
    def __init__(self):
        Grid.__init__(self)
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

class DataAssimilator(GP):
    salinity = []
    temperature = []
    path = []
    timestamp = []
    def __init__(self):
        print("Data collector is initialised correctly")

    def append_salinity(self, value):
        DataAssimilator.salinity.append(value)

    def append_temperature(self, value):
        DataAssimilator.temperature.append(value)

    def append_path(self, value):
        DataAssimilator.path.append(value)

    def append_timestamp(self, value):
        DataAssimilator.timestamp.append(value)

    def save_data(self):
        self.salinity = np.array(self.salinity).reshape(-1, 1)
        self.temperature = np.array(self.temperature).reshape(-1, 1)
        self.path = np.array(self.path).reshape(-1, 3)
        self.timestamp = np.array(self.timestamp).reshape(-1, 1)
        np.savetxt("data_salinity.txt", self.salinity, delimiter=",")
        np.savetxt("data_temperature.txt", self.temperature, delimiter=",")
        np.savetxt("data_path.txt", self.path, delimiter=",")
        np.savetxt("data_timestamp.txt", self.timestamp, delimiter=",")

    def vehpos2latlon(self, x, y, lat_origin, lon_origin):
        if lat_origin <= 10:
            lat_origin = self.rad2deg(lat_origin)
            lon_origin = self.rad2deg(lon_origin)
        lat = lat_origin + self.rad2deg(x * np.pi * 2.0 / self.circumference)
        lon = lon_origin + self.rad2deg(y * np.pi * 2.0 / (self.circumference * np.cos(self.deg2rad(lat))))
        return lat, lon


class PathPlanner(AUV, DataAssimilator):

    xstart, ystart, zstart = [None, None, None]
    xnow, ynow, znow = [None, None, None]
    xpre, ypre, zpre = [None, None, None]
    xcand, ycand, zcand = [None, None, None]
    mu_cond, Sigma_cond, F = [None, None, None]
    mu_prior, Sigma_prior = [None, None]
    lat_next, lon_next, depth_next = [None, None, None]
    travelled_waypoints = None
    Total_waypoints = 10

    def __init__(self):
        AUV.__init__(self)
        DataAssimilator.__init__(self)
        print("AUV is set up correctly")
        self.travelled_waypoints = 0
        self.load_prior()
        self.find_starting_loc()
        # self.auv_handler.setWaypoint(self.deg2rad(self.lat_origin), self.deg2rad(self.lon_origin), 0)
        self.run()

    def load_prior(self):
        mu_prior_sal = np.loadtxt('mu_prior_sal.txt', delimiter=",")
        mu_prior_temp = np.loadtxt('mu_prior_temp.txt', delimiter=",")
        print("Prior for salinity is loaded correctly!!!")
        self.mu_prior = mu_prior_sal
        self.Sigma_prior = self.Sigma_sal
        print("mu_prior shape is: ", self.mu_prior.shape)
        print("Sigma_prior shape is: ", self.Sigma_prior.shape)

    def ravel_index(self, loc):
        x, y, z = loc
        ind = int(z * self.N1 * self.N2 + y * self.N1 + x)
        return ind

    def unravel_index(self, ind):
        zind = np.floor(ind / (self.N1 * self.N2))
        residual = ind - zind * (self.N1 * self.N2)
        yind = np.floor(residual / self.N1)
        xind = residual - yind * self.N1
        loc = [int(xind), int(yind), int(zind)]
        return loc

    def updateF(self, ind):
        self.F = np.zeros([1, self.N])
        self.F[0, ind] = True

    def EP_1D(self, mu, Sigma, Threshold):
        EP = np.zeros_like(mu)
        for i in range(EP.shape[0]):
            EP[i] = norm.cdf(Threshold, mu[i], Sigma[i, i])
        return EP

    def find_starting_loc(self):
        EP_Prior = self.EP_1D(self.mu_prior, self.Sigma_prior, self.Threshold_S)
        ep_criterion = 0.5 # excursion probability close to 0.5
        ind = (np.abs(EP_Prior - ep_criterion)).argmin()
        loc = self.unravel_index(ind)
        self.xstart, self.ystart, self.zstart = loc
        print(self.xstart, self.ystart, self.zstart)
        print(loc)
        print(np.where(self.F == 1))
        self.updateF(ind)
        print(np.where(self.F == 1))

    def find_candidates_loc(self):
        x_ind_l = [self.xnow - 1 if self.xnow > 0 else self.xnow]
        x_ind_u = [self.xnow + 1 if self.xnow < self.N1 - 1 else self.xnow]
        y_ind_l = [self.ynow - 1 if self.ynow > 0 else self.ynow]
        y_ind_u = [self.ynow + 1 if self.ynow < self.N2 - 1 else self.ynow]
        z_ind_l = [self.znow - 1 if self.znow > 0 else self.znow]
        z_ind_u = [self.znow + 1 if self.znow < self.N3 - 1 else self.znow]

        x_ind_v = np.unique(np.vstack((x_ind_l, self.xnow, x_ind_u)))
        y_ind_v = np.unique(np.vstack((y_ind_l, self.ynow, y_ind_u)))
        z_ind_v = np.unique(np.vstack((z_ind_l, self.znow, z_ind_u)))

        self.xcand, self.ycand, self.zcand = np.meshgrid(x_ind_v, y_ind_v, z_ind_v)
        self.xcand.reshape(-1, 1)
        self.ycand.reshape(-1, 1)
        self.zcand.reshape(-1, 1)

    def GPupd(self, y_sampled):
        C = self.F @ self.Sigma_cond @ self.F.T + self.R_sal
        self.mu_cond = self.mu_cond + self.Sigma_cond @ self.F.T @ np.linalg.solve(C, (y_sampled - self.F @ self.mu_cond))
        self.Sigma_cond = self.Sigma_cond - self.Sigma_cond @ self.F.T @ np.linalg.solve(C, self.F @ self.Sigma_cond)

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
        id = []
        dx1 = self.xnow - self.xpre
        dy1 = self.ynow - self.ypre
        dz1 = self.znow - self.zpre
        vec1 = np.array([dx1, dy1, dz1])
        for i in self.xcand:
            for j in self.ycand:
                for z in self.zcand:
                    if i == self.xnow and j == self.ynow and z == self.znow:
                        continue
                    dx2 = i - self.xnow
                    dy2 = j - self.ynow
                    dz2 = z - self.znow
                    vec2 = np.array([dx2, dy2, dz2])
                    if np.dot(vec1, vec2) >= 0:
                        id.append(self.ravel_index([i, j, z]))
                    else:
                        continue
        id = np.unique(np.array(id))

        M = len(id)
        eibv = []
        for k in range(M):
            F = np.zeros([1, self.N])
            F[0, id[k]] = True
            eibv.append(self.EIBV_1D(self.Threshold_S, mu, Sig, F, self.R_sal))
        ind_desired = np.argmin(np.array(eibv))
        self.x_next, self.y_next, self.z_next = self.unravel_index(id[ind_desired])

    def getNextWaypoint(self):
        x_loc, y_loc = self.R @ np.vstack((self.xnext * self.dx, self.ynext * self.dy))  # converted xp/yp with distance inside
        self.lat_next = self.lat_origin + self.rad2deg(x_loc * np.pi * 2.0 / self.circumference)
        self.lon_next = self.lon_origin + self.rad2deg(y_loc * np.pi * 2.0 / (self.circumference * np.cos(self.deg2rad(lat))))
        self.depth_next = self.depth_obs[self.znext]
        print("Next way point: ", self.lat_next, self.lon_next, self.depth_next)

    def updateWaypoint(self):
        self.xpre, self.ypre, self.zpre = self.xnow, self.ynow, self.znow
        self.xnow, self.ynow, self.znow = self.xnext, self.ynext, self.znext

    def run(self):
        while not rospy.is_shutdown():
            if self.init:
                self.append_salinity(self.currentSalinity)
                self.append_temperature(self.currentTemperature)
                lat_temp, lon_temp = self.vehpos2latlon(self.vehicle_pos[0], self.vehicle_pos[1], self.lat_origin, self.lon_origin)
                print(lat_temp, lon_temp, self.vehicle_pos[2])
                print([lat_temp, lon_temp, self.vehicle_pos[2]])
                self.append_path([lat_temp, lon_temp, self.vehicle_pos[2]])

                print(self.auv_handler.getState())
                if self.auv_handler.getState() == "waiting" and self.last_state != "waiting":
                    # if self.surfacing:
                    #     for i in range(self.surfacing_time):
                    #         print("Sleep {:d} seconds".format(i))
                    #         self.auv_handler.spin()  # publishes the reference, stay on the surface
                    #         self.rate.sleep()  #
                    #     self.surfacing = False

                    print("Arrived the current location")
                    # self.auv_handler.setWaypoint(self.deg2rad(self.lat_origin), self.deg2rad(self.lon_origin), 2)

                    sal_sampled = np.mean(self.salinity[-10:])  # take the past ten samples and average
                    self.GPupd(sal_sampled)
                    self.find_candidates_loc()
                    self.find_next_EIBV_1D(self.mu_cond, self.Sigma_cond)
                    self.getNextWaypoint()

                    ind_next = self.ravel_index([self.xnext, self.ynext, self.znext])
                    self.updateF(ind_next)
                    self.updateWaypoint()
                    self.travelled_waypoints += 1
                    # Move to the next waypoint
                    self.auv_handler.setWaypoint(self.deg2rad(self.lat_next), self.deg2rad(self.lon_next), self.depth_next)
                    if self.travelled_waypoints >= self.Total_steps:
            #             logfile.write("Mission completed!!!\n")
                        rospy.signal_shutdown("Mission completed!!!")
                self.last_state = self.auv_handler.getState()
                self.auv_handler.spin()
            self.rate.sleep()
    pass

a = PathPlanner()








