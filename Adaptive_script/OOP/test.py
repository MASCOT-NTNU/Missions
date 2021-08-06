

from GP import GaussianProcess
from sklearn.linear_model import LinearRegression
# a = Grid()
# a.print_var()
# a.checkGrid()
# a.generateBox()
# a.checkBox()

# b = GaussianProcess()
import numpy as np
class Prior(GaussianProcess):
    AUVdata = None
    mu_sal_prior = None
    mu_temp_prior = None
    Sigma_sal_prior = None
    Sigma_temp_prior = None
    lat_grid = np.zeros([GaussianProcess.N1, GaussianProcess.N2])
    lon_grid = np.zeros([GaussianProcess.N1, GaussianProcess.N2])
    coord_grid = np.zeros([GaussianProcess.N1 * GaussianProcess.N2, 2])
    depth_tolerance = 0.25 # tolerance +/- in depth, 0.5 m == [0.25 ~ 0.75]m

    SINMOD_datapath = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Adaptive_script/samples_2020.05.01.nc"
    SINMOD_Data = None

    beta0 = None
    beta1 = None
    mu_prior_sal = None
    mu_prior_temp = None

    def __init__(self):
        GaussianProcess.__init__(self)
        self.generateCoordinates()
        self.getAUVData()
        self.getSINMODData()
        self.getCoefficients()
        self.saveCoef()

        # print(Prior.beta0)
        # print(Prior.beta1)
        print(Prior.mu_prior_sal.shape)
        # print(Prior.AUVdata.shape)
        # self.print_var()

    # def print_var(self):
    #     print("lat_grid: ", Prior.lat_grid.shape)
    #     print("lon_grid: ", Prior.lon_grid.shape)
    #     print("coord_grid: ", Prior.coord_grid.shape)
    #     # print("mu_sal_prior: ", Prior.mu_sal_prior.shape)
    #     # print("mu_temp_prior: ", Prior.mu_temp_prior.shape)
    #     # print(GaussianProcess.N2)
    #     # print(GaussianProcess.grid.shape)
    #     # print(Prior.coord_grid)
    #     print("data is ", Prior.AUVdata)

    def generateCoordinates(self):
        for i in range(GaussianProcess.N1):
            for j in range(GaussianProcess.N2):
                xnew, ynew = GaussianProcess.R @ np.vstack((GaussianProcess.x[i], GaussianProcess.y[j]))

                Prior.lat_grid[i, j] = GaussianProcess.lat_origin + \
                            GaussianProcess.rad2deg(self, xnew * np.pi * 2.0 /
                                                    GaussianProcess.circumference)
                Prior.lon_grid[i, j] = GaussianProcess.lon_origin + \
                            GaussianProcess.rad2deg(self, ynew * np.pi * 2.0 /
                                                    (GaussianProcess.circumference *
                                                     np.cos(GaussianProcess.deg2rad(self, Prior.lat_grid[i, j]))))
        Prior.coord_grid = np.hstack((Prior.lat_grid.reshape(-1, 1), Prior.lon_grid.reshape(-1, 1)))

    def getAUVData(self):
        Prior.AUVdata = np.loadtxt('data.txt', delimiter = ",")

    def getSINMODData(self):
        import netCDF4
        Prior.SINMOD_Data = netCDF4.Dataset(Prior.SINMOD_datapath)

    def getCoefficients(self):
        # timestamp = Prior.AUVdata[:, 0].reshape(-1, 1)
        lat_auv_origin = self.rad2deg(Prior.AUVdata[:, 1]).reshape(-1, 1)
        lon_auv_origin = self.rad2deg(Prior.AUVdata[:, 2]).reshape(-1, 1)
        xauv = Prior.AUVdata[:, 3].reshape(-1, 1)
        yauv = Prior.AUVdata[:, 4].reshape(-1, 1)
        # zauv = Prior.AUVdata[:, 5].reshape(-1, 1)
        depth_auv = Prior.AUVdata[:, 6].reshape(-1, 1)
        sal_auv = Prior.AUVdata[:, 7].reshape(-1, 1)
        temp_auv = Prior.AUVdata[:, 8].reshape(-1, 1)
        lat_auv = lat_auv_origin + self.rad2deg(xauv * np.pi * 2.0 / self.circumference)
        lon_auv = lon_auv_origin + self.rad2deg(yauv * np.pi * 2.0 / (self.circumference * np.cos(self.deg2rad(lat_auv))))

        depthl = np.array(self.depth_obs) - self.depth_tolerance
        depthu = np.array(self.depth_obs) + self.depth_tolerance

        Prior.beta0 = np.zeros([len(self.depth_obs), 2])
        Prior.beta1 = np.zeros([len(self.depth_obs), 2])
        sal_residual = []
        temp_residual = []

        Prior.mu_prior_sal = []
        Prior.mu_prior_temp = []
        for i in range(len(self.depth_obs)):
            # sort out AUV data
            ind_obs = (depthl[i] <= depth_auv) & (depth_auv <= depthu[i])
            lat_obs = lat_auv[ind_obs].reshape(-1, 1)
            lon_obs = lon_auv[ind_obs].reshape(-1, 1)
            sal_obs = sal_auv[ind_obs].reshape(-1, 1)
            temp_obs = temp_auv[ind_obs].reshape(-1, 1)

            # sort out SINMOD data
            salinity = np.mean(Prior.SINMOD_Data['salinity'][:, :, :, :], axis=0)  # time averaging of salinity
            temperature = np.mean(Prior.SINMOD_Data['temperature'][:, :, :, :], axis=0) - 273.15  # time averaging of temperature
            depth_sinmod = np.array(Prior.SINMOD_Data['zc'])  # depth from SINMOD
            lat_sinmod = np.array(Prior.SINMOD_Data['gridLats'][:, :]).reshape(-1, 1)  # lat from SINMOD
            lon_sinmod = np.array(Prior.SINMOD_Data['gridLons'][:, :]).reshape(-1, 1)  # lon from SINMOD
            sal_sinmod = np.zeros([sal_obs.shape[0], 1])
            temp_sinmod = np.zeros([temp_obs.shape[0], 1])

            for j in range(sal_obs.shape[0]):
                ind_depth = np.where(np.array(depth_sinmod) == self.depth_obs[i])[0][0]
                idx = np.argmin((lat_sinmod - lat_obs[j]) ** 2 + (lon_sinmod - lon_obs[j]) ** 2)
                sal_sinmod[j] = salinity[ind_depth].reshape(-1, 1)[idx]
                temp_sinmod[j] = temperature[ind_depth].reshape(-1, 1)[idx]

            # compute the coef for salinity
            sal_modelX = np.hstack((np.ones_like(sal_sinmod), sal_sinmod))
            sal_modelY = sal_obs
            Beta_sal = np.linalg.solve((sal_modelX.T @ sal_modelX), (sal_modelX.T @ sal_modelY))
            # compute the coef for temperature
            temp_modelX = np.hstack((np.ones_like(temp_sinmod), temp_sinmod))
            temp_modelY = temp_obs
            Beta_temp = np.linalg.solve((temp_modelX.T @ temp_modelX), (temp_modelX.T @ temp_modelY))

            Prior.beta0[i, :] = np.hstack((Beta_sal[0], Beta_temp[0]))
            Prior.beta1[i, :] = np.hstack((Beta_sal[1], Beta_temp[1]))
            Prior.mu_prior_sal.append(Prior.beta0[i, 0] + Prior.beta1[i, 0] * sal_sinmod)
            Prior.mu_prior_temp.append(Prior.beta0[i, 1] + Prior.beta1[i, 1] * temp_sinmod)

            sal_residual.append(sal_obs - Beta_sal[0] - Beta_sal[1] * sal_sinmod)
            temp_residual.append(temp_obs - Beta_temp[0] - Beta_temp[1] * temp_sinmod)

        Prior.mu_prior_sal = np.array(Prior.mu_prior_sal).reshape(-1, 1)
        Prior.mu_prior_temp = np.array(Prior.mu_prior_temp).reshape(-1, 1)

    def saveCoef(self):
        np.savetxt("beta0.txt", Prior.beta0, delimiter=",")
        np.savetxt("beta1.txt", Prior.beta1, delimiter=",")
        np.savetxt("mu_prior_sal.txt", Prior.mu_prior_sal, delimiter=",")
        np.savetxt("mu_prior_temp.txt", Prior.mu_prior_temp, delimiter=",")

    def checkCoords(self):
        from gmplot import GoogleMapPlotter
        initial_zoom = 12
        apikey = 'AIzaSyAZ_VZXoJULTFQ9KSPg1ClzHEFjyPbJUro'
        gmap = GoogleMapPlotter(self.lat_origin, self.lon_origin, initial_zoom, map_type='satellite', apikey=apikey)
        # gmap.scatter(self.coord_grid[:, 0], self.coord_grid[:, 1])
        gmap.scatter(self.coord_grid[:, 0], self.coord_grid[:, 1], color='#99ff00', size=20, marker=False)
        # print(self.coo)
        gmap.draw("/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/MapPlot/map.html")

c = Prior()
# c.checkCoords()

#%%
import numpy as np
from sklearn.linear_model import LinearRegression
print("hello")
x = np.arange(10).reshape(-1, 1)
y = (np.arange(10) + 2 * np.random.rand(10)).reshape(-1, 1)

import matplotlib.pyplot as plt
 X = np.hstack((np.ones_like(x), x))

beta = np.linalg.solve((X.T @ X), X.T @ y)
Y = beta[0] + beta[1] * x
model = LinearRegression()
model.fit(x, y)
print(model.intercept_, model.coef_)
plt.plot(x, y, 'k.')
plt.plot(x, Y, 'r-')

plt.show()
print(beta)

#%%
a = np.loadtxt("data.txt", delimiter = ",")

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








