
# a = Grid()
# a.print_var()
# a.checkGrid()
# a.generateBox()
# a.checkBox()

# b = GaussianProcess()
# from Prior import Prior
# a = Prior()

# class PathPlanner:


import rospy
import numpy as np
from scipy.stats import mvn
from auv_handler import AuvHandler

import imc_ros_interface
from imc_ros_interface.msg import Temperature, Salinity, EstimatedState
from Grid import Grid

class AUV(Prior):
    def __init__(self):
        Prior.__init__(self)
        self.node_name = 'LAUV-Roald'
        rospy.init_node(self.node_name, anonymous=True)
        self.rate = rospy.Rate(1)  # 1Hz
        self.auv_handler = AuvHandler(self.node_name, "LAUV-Roald")

        rospy.Subscriber("/Vehicle/Out/Temperature_filtered", Temperature, self.TemperatureCB)
        rospy.Subscriber("/Vehicle/Out/Salinity_filtered", Salinity, self.SalinityCB)
        rospy.Subscriber("/Vehicle/Out/EstimatedState_filtered", EstimatedState, self.EstimatedStateCB)

        self.speed = 1.6  # m/s
        self.depth = 0.0  # meters
        self.last_state = "unavailable"
        self.rate.sleep()
        # self.auv_handler.setWaypoint(deg2rad(lat4), deg2rad(lon4))
        # self.auv_handler.setWaypoint(deg2rad(lat_start), deg2rad(lon_start))
        self.init = True
        self.currentTemperature = 0.0
        self.currentSalinity = 0.0
        self.vehicle_pos = [0, 0, 0]
        self.surfacing = False
        self.surfacing_time = 25 # surface time, [sec]
        print("finished initialisation")

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

class PathPlanner(AUV):

    xstart, ystart, zstart = [None, None, None]
    xnow, ynow, znow = [None, None, None]
    xpre, ypre, zpre = [None, None, None]
    xcand, ycand, zcand = [None, None, None]
    mu_cond, Sigma_cond, F = [None, None, None]


    def __init__(self):
        AUV.__init__(self)
        print("AUV is set up correctly")
        self.auv_handler.setWaypoint(self.deg2rad(self.lat_origin), self.deg2rad(self.lon_origin), 0)
        self.run()

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
        x_next, y_next, z_next = self.unravel_index(id[ind_desired])

        return x_next, y_next, z_next

    def run(self):
        print("get into run")
        while not rospy.is_shutdown():
            print("not shut down")
            if self.init:
                print(self.auv_handler.getState())
                if self.auv_handler.getState() == "waiting" and self.last_state != "waiting":
                    # if self.surfacing:
                    #     for i in range(self.surfacing_time):
                    #         print("Sleep {:d} seconds".format(i))
                    #         self.auv_handler.spin()  # publishes the reference, stay on the surface
                    #         self.rate.sleep()  #
                    #     self.surfacing = False

                    print("Arrived the current location")
                    self.auv_handler.setWaypoint(self.deg2rad(self.lat_origin), self.deg2rad(self.lon_origin), 2)

            #         sal_sampled = np.mean(data_salinity[-10:])  # take the past ten samples and average
            #
            #         mu_cond, Sigma_cond = GPupd(mu_cond, Sigma_cond, R, F, sal_sampled)
            #
            #         xcand, ycand, zcand = find_candidates_loc(xnow, ynow, znow, N1, N2, N3)
            #
            #         t1 = time.time()
            #         xnext, ynext, znext = find_next_EIBV_1D(xcand, ycand, zcand,
            #                                                 xnow, ynow, znow,
            #                                                 xpre, ypre, zpre,
            #                                                 N1, N2, N3, Sigma_cond,
            #                                                 mu_cond, tau_sal, Threshold_S)
            #         t2 = time.time()
            #         t_elapsed.append(t2 - t1)
            #         print("It takes {:.2f} seconds to compute the next waypoint".format(t2 - t1))
            #         logfile.write("It takes {:.2f} seconds to compute the next waypoint\n".format(t2 - t1))
            #
            #         print("next is ", xnext, ynext, znext)
            #         lat_next, lon_next = xy2latlon(xnext, ynext, origin, distance, alpha)
            #         depth_next = depth_obs[znext]
            #         ind_next = ravel_index([xnext, ynext, znext], N1, N2, N3)
            #
            #         F = np.zeros([1, N])
            #         F[0, ind_next] = True
            #
            #         xpre, ypre, zpre = xnow, ynow, znow
            #         xnow, ynow, znow = xnext, ynext, znext
            #
            #         path.append([xnow, ynow, znow])
            #         print(xcand.shape)
            #
            #         # Move to the next waypoint
            #         self.auv_handler.setWaypoint(deg2rad(lat_next), deg2rad(lon_next), depth_next)
            #
            #         if counter_waypoint >= N_steps:
            #             logfile.write("Mission completed!!!\n")
            #             rospy.signal_shutdown("Mission completed!!!")
                self.last_state = self.auv_handler.getState()
                self.auv_handler.spin()
            self.rate.sleep()
    pass

a = PathPlanner()



#%%
# class PathPlanner(Grid, GP):
#     def __init__(self, Grid, GP):
#         # super().__init__(Grid)
#         super().__init__(GP)
#         print("hello world")
#         print(self.DistanceMatrix.shape)
#         # self.EP_prior = self.EP_1D()
#         # self.starting_loc = self.find_starting_loc()
#         # self.mu_cond = self.mu_prior
#         # self.Sigma_cond = self.Sigma_prior
#         # self.noise = self.tau_sal ** 2
#         # self.R = np.diagflat(self.noise)
#
#     def xy2latlon(x, y, origin, distance, alpha):
#         '''
#         :param x: index from origin along left line
#         :param y: index from origin along right line
#         :param origin:
#         :param distance:
#         :param alpha:
#         :return:
#         '''
#         R = np.array([[np.cos(self.deg2rad(alpha)), -np.sin(self.deg2rad(alpha))],
#                       [np.sin(self.deg2rad(alpha)), np.cos(self.deg2rad(alpha))]])
#         x_loc, y_loc = R @ np.vstack((x * distance, y * distance))  # converted xp/yp with distance inside
#         lat_origin, lon_origin = origin
#         lat = lat_origin + self.rad2deg(x_loc * np.pi * 2.0 / self.circumference)
#         lon = lon_origin + self.rad2deg(y_loc * np.pi * 2.0 / (self.circumference * np.cos(self.deg2rad(lat))))
#         return np.hstack((lat, lon))
#
#     def ravel_index(loc, n1, n2, n3):
#         '''
#         :param loc:
#         :param n1:
#         :param n2:
#         :param n3:
#         :return:
#         '''
#         x, y, z = loc
#         ind = int(z * n1 * n2 + y * n1 + x)
#         return ind
#
#     def unravel_index(ind, n1, n2, n3):
#         '''
#         :param ind:
#         :param n1:
#         :param n2:
#         :param n3:
#         :return:
#         '''
#         zind = np.floor(ind / (n1 * n2))
#         residual = ind - zind * (n1 * n2)
#         yind = np.floor(residual / n1)
#         xind = residual - yind * n1
#         loc = [int(xind), int(yind), int(zind)]
#         return loc
#
#     def find_starting_loc(EP_Prior, n1, n2, n3):
#         '''
#         :param EP_Prior:
#         :param n1:
#         :param n2:
#         :param n3:
#         :return:
#         '''
#         ep_criterion = 0.5
#         ind = (np.abs(EP_Prior - ep_criterion)).argmin()
#         loc = unravel_index(ind, n1, n2, n3)
#         return loc
#
#     def find_candidates_loc(x_ind, y_ind, z_ind, N1, N2, N3):
#         '''
#         :param x_ind:
#         :param y_ind:
#         :param z_ind:
#         :param N1: number of grid along x direction
#         :param N2:
#         :param N3:
#         :return:
#         '''
#
#         x_ind_l = [x_ind - 1 if x_ind > 0 else x_ind]
#         x_ind_u = [x_ind + 1 if x_ind < N1 - 1 else x_ind]
#         y_ind_l = [y_ind - 1 if y_ind > 0 else y_ind]
#         y_ind_u = [y_ind + 1 if y_ind < N2 - 1 else y_ind]
#         z_ind_l = [z_ind - 1 if z_ind > 0 else z_ind]
#         z_ind_u = [z_ind + 1 if z_ind < N3 - 1 else z_ind]
#
#         x_ind_v = np.unique(np.vstack((x_ind_l, x_ind, x_ind_u)))
#         y_ind_v = np.unique(np.vstack((y_ind_l, y_ind, y_ind_u)))
#         z_ind_v = np.unique(np.vstack((z_ind_l, z_ind, z_ind_u)))
#
#         x_ind, y_ind, z_ind = np.meshgrid(x_ind_v, y_ind_v, z_ind_v)
#
#         return x_ind.reshape(-1, 1), y_ind.reshape(-1, 1), z_ind.reshape(-1, 1)
#
#     def EP_1D(mu, Sigma, Threshold):
#         '''
#         This function computes the excursion probability
#         :param mu:
#         :param Sigma:
#         :param Threshold:
#         :return:
#         '''
#         EP = np.zeros_like(mu)
#         for i in range(EP.shape[0]):
#             EP[i] = norm.cdf(Threshold, mu[i], Sigma[i, i])
#         return EP
#
#     def find_next_EIBV_1D(x_cand, y_cand, z_cand, x_now, y_now, z_now,
#                           x_pre, y_pre, z_pre, N1, N2, N3, Sig, mu, tau, Threshold):
#
#         id = []
#         dx1 = x_now - x_pre
#         dy1 = y_now - y_pre
#         dz1 = z_now - z_pre
#         vec1 = np.array([dx1, dy1, dz1])
#         for i in x_cand:
#             for j in y_cand:
#                 for z in z_cand:
#                     if i == x_now and j == y_now and z == z_now:
#                         continue
#                     dx2 = i - x_now
#                     dy2 = j - y_now
#                     dz2 = z - z_now
#                     vec2 = np.array([dx2, dy2, dz2])
#                     if np.dot(vec1, vec2) >= 0:
#                         id.append(ravel_index([i, j, z], N1, N2, N3))
#                     else:
#                         continue
#         id = np.unique(np.array(id))
#
#         M = len(id)
#         noise = tau ** 2
#         R = np.diagflat(noise)  # diag not anymore support constructing matrix from vector
#         N = N1 * N2 * N3
#         eibv = []
#         for k in range(M):
#             F = np.zeros([1, N])
#             F[0, id[k]] = True
#             eibv.append(EIBV_1D(Threshold, mu, Sig, F, R))
#         ind_desired = np.argmin(np.array(eibv))
#         x_next, y_next, z_next = unravel_index(id[ind_desired], N1, N2, N3)

#         return x_next, y_next, z_next


# if __name__ == "__main__":
#     A = Grid()
#     B = B()
#     B.test(A)
#     B.lat_origin = 10
#     print(B.lat_origin)
#     print(B.alpha)
#     print(A.lat_origin)
#     C = C()
#     C.te()
#     print(C.lat_origin)
#     print(A.__dir__())
#     print(B.__dir__())
    # print(A.lon_origin)

    # B = GP(A)
    # C = PathPlanner(A, B)


# #%%
# class A:
#     A = 'nothing'
#
# class B(A):
#     def method_b(self, value):
#         A.A = value
#
# class C(A):
#     pass
#
# b = B()
# c = C()
# print(c.A)
# b.method_b(13)
# print(c.A)
#
# class T1:
#     t = 10
# class T2(T1):
#     b = T1.t
#     print(b)
#     def __init__(self, value):
#         T1.t = value
# class T3(T2):
#     def __init__(self):
#         print("good")
#     pass
#
# a = T1()
# c = T3()
# # print(a.t)
# # print(c.t)
# b = T2(12)
# # print(c.t)
# # print(a.t)








