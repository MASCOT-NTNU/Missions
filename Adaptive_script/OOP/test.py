import numpy as np
from gmplot import GoogleMapPlotter
import matplotlib.pyplot as plt

class Grid:
    def __init__(self):
        self.lat_origin, self.lon_origin = 63.446905, 10.419426  # right bottom corner
        self.origin = [self.lat_origin, self.lon_origin]
        self.distance = 1000 # distance of the edge
        self.depth_obs = [0.5, 1.0, 1.5, 2.0, 2.5]  # planned depth to be observed
        self.alpha = 60 # angle to be tilted

        self.N1 = 25  # number of grid points along north direction
        self.N2 = 25  # number of grid points along east direction
        self.N3 = 5  # number of layers in the depth dimension
        self.N = self.N1 * self.N2 * self.N3  # total number of grid points
        self.Surfacing_time = 25  # surfacing time, [seconds]
        self.circumference = 40075000 # circumference of the earth, [m]
        self.box = self.BBox() # box of the field
        print(self.circumference)
        print(self.box)
        print(self.lat_origin)
        print(self.box[:, 0].shape)

        self.XLIM = [0, self.distance] # limit for the x axis
        self.YLIM = [0, self.distance] # limit for the y axis
        self.ZLIM = [self.depth_obs[0], self.depth_obs[-1]] # limit for the z axis
        self.x = np.linspace(self.XLIM[0], self.XLIM[-1], self.N1) # x coordinates
        self.y = np.linspace(self.YLIM[0], self.YLIM[-1], self.N1) # y coordinates
        self.z = np.array(self.depth_obs).reshape(-1, 1) # z coordinates

        self.xm, self.ym, self.zm = np.meshgrid(self.x, self.y, self.z)
        self.xv = self.xm.reshape(-1, 1)
        self.yv = self.ym.reshape(-1, 1)
        self.zv = self.zm.reshape(-1, 1)
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        self.dz = self.z[1] - self.z[0]

        self.grid = np.array(self.generageGrid())

        # print(b[:100, :])

        # initial_zoom = 12
        # apikey = 'AIzaSyAZ_VZXoJULTFQ9KSPg1ClzHEFjyPbJUro'
        # gmap = GoogleMapPlotter(self.lat_origin, self.lon_origin, initial_zoom, map_type='satellite', apikey=apikey)
        # gmap.scatter(self.box[:, 0], self.box[:, 1], 'cornflowerblue', size=10)
        # gmap.draw("/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/MapPlot/map.html")


    def deg2rad(self, deg):
        '''
        :param deg:
        :return:
        '''
        return deg / 180 * np.pi

    def rad2deg(self, rad):
        '''
        :param rad:
        :return:
        '''
        return rad / np.pi * 180

    def BBox(self):
        '''
        :return: Box of the operational regions
        '''
        lat4 = self.deg2rad(self.lat_origin) # the right bottom corner
        lon4 = self.deg2rad(self.lon_origin) # the right bottom corner

        lat2 = lat4 + self.distance * np.sin(self.deg2rad(self.alpha)) / self.circumference * 2 * np.pi
        lat1 = lat2 + self.distance * np.cos(self.deg2rad(self.alpha)) / self.circumference * 2 * np.pi
        lat3 = lat4 + self.distance * np.sin(np.pi / 2 - self.deg2rad(self.alpha)) / self.circumference * 2 * np.pi

        lon2 = lon4 + self.distance * np.cos(self.deg2rad(self.alpha)) / (self.circumference * np.cos(lat2)) * 2 * np.pi
        lon3 = lon4 - self.distance * np.cos(np.pi / 2 - self.deg2rad(self.alpha)) / (self.circumference * np.cos(lat3)) * 2 * np.pi
        lon1 = lon3 + self.distance * np.cos(self.deg2rad(self.alpha)) / (self.circumference * np.cos(lat1)) * 2 * np.pi

        box = np.vstack((np.array([lat1, lat2, lat3, lat4]), np.array([lon1, lon2, lon3, lon4]))).T

        return self.rad2deg(box)


    def generageGrid(self):
        R = np.array([[np.cos(self.deg2rad(self.alpha)), np.sin(self.deg2rad(self.alpha))],
                      [-np.sin(self.deg2rad(self.alpha)), np.cos(self.deg2rad(self.alpha))]])
        grid = []
        for k in range(self.N3):
            for i in range(self.N1):
                for j in range(self.N2):
                    tempx, tempy = R @ np.vstack((self.xm[i, j, k], self.ym[i, j, k]))
                    grid.append([tempx.squeeze(), tempy.squeeze(), self.depth_obs[k]])
        return grid

class GP(Grid):
    def __init__(self, Grid):
        super().__init__()
        self.sigma_sal = np.sqrt(4) # scaling coef in matern kernel for salinity
        self.tau_sal = np.sqrt(.3) # iid noise
        self.Threshold_S = 28 # threshold for salinity

        # self.sigma_temp = np.sqrt(.5) # scaling coef in matern kernel for temperature
        # self.tau_temp = np.sqrt(.1) # iid noise
        # self.Threshold_T = 10.5 # threshold for temperature

        self.eta = 4.5 / 400 # coef in matern kernel
        self.ksi = 1000 / 24 / .5 # scaling factor in 3D

        self.DistanceMatrix = self.compute_DistanceMatrix()
        self.Sigma_prior = self.Matern_cov()

    def compute_DistanceMatrix(self):
        '''
        :return: Distance matrix with scaling the depth direction
        '''
        X = self.grid[:, 0].reshape(-1, 1)
        Y = self.grid[:, 1].reshape(-1, 1)
        Z = self.grid[:, -1].reshape(-1, 1)

        distX = X @ np.ones([1, X.shape[0]]) - np.ones([X.shape[0], 1]) @ X.T
        distY = Y @ np.ones([1, Y.shape[0]]) - np.ones([Y.shape[0], 1]) @ Y.T
        distXY = distX ** 2 + distY ** 2
        distZ = Z @ np.ones([1, Z.shape[0]]) - np.ones([Z.shape[0], 1]) @ Z.T
        dist = np.sqrt(distXY + (self.ksi * distZ) ** 2)
        return dist

    def Matern_cov(self):
        '''
        :return: Covariance matrix
        '''
        return self.sigma_sal ** 2 * (1 + self.eta * self.DistanceMatrix) * np.exp(-self.eta * self.DistanceMatrix)



class PathPlanner(GP):
    def __init__(self, GP):
        super().__init__(Grid)
        print("hello world")
        print(self.DistanceMatrix.shape)
        self.EP_prior = self.EP_1D()
        self.starting_loc = self.find_starting_loc()
        self.mu_cond = self.mu_prior
        self.Sigma_cond = self.Sigma_prior
        self.noise = self.tau_sal ** 2
        self.R = np.diagflat(self.noise)

    def xy2latlon(x, y, origin, distance, alpha):
        '''
        :param x: index from origin along left line
        :param y: index from origin along right line
        :param origin:
        :param distance:
        :param alpha:
        :return:
        '''
        R = np.array([[np.cos(self.deg2rad(alpha)), -np.sin(self.deg2rad(alpha))],
                      [np.sin(self.deg2rad(alpha)), np.cos(self.deg2rad(alpha))]])
        x_loc, y_loc = R @ np.vstack((x * distance, y * distance))  # converted xp/yp with distance inside
        lat_origin, lon_origin = origin
        lat = lat_origin + self.rad2deg(x_loc * np.pi * 2.0 / self.circumference)
        lon = lon_origin + self.rad2deg(y_loc * np.pi * 2.0 / (self.circumference * np.cos(self.deg2rad(lat))))
        return np.hstack((lat, lon))

    def ravel_index(loc, n1, n2, n3):
        '''
        :param loc:
        :param n1:
        :param n2:
        :param n3:
        :return:
        '''
        x, y, z = loc
        ind = int(z * n1 * n2 + y * n1 + x)
        return ind

    def unravel_index(ind, n1, n2, n3):
        '''
        :param ind:
        :param n1:
        :param n2:
        :param n3:
        :return:
        '''
        zind = np.floor(ind / (n1 * n2))
        residual = ind - zind * (n1 * n2)
        yind = np.floor(residual / n1)
        xind = residual - yind * n1
        loc = [int(xind), int(yind), int(zind)]
        return loc

    def find_starting_loc(EP_Prior, n1, n2, n3):
        '''
        :param EP_Prior:
        :param n1:
        :param n2:
        :param n3:
        :return:
        '''
        ep_criterion = 0.5
        ind = (np.abs(EP_Prior - ep_criterion)).argmin()
        loc = unravel_index(ind, n1, n2, n3)
        return loc

    def find_candidates_loc(x_ind, y_ind, z_ind, N1, N2, N3):
        '''
        :param x_ind:
        :param y_ind:
        :param z_ind:
        :param N1: number of grid along x direction
        :param N2:
        :param N3:
        :return:
        '''

        x_ind_l = [x_ind - 1 if x_ind > 0 else x_ind]
        x_ind_u = [x_ind + 1 if x_ind < N1 - 1 else x_ind]
        y_ind_l = [y_ind - 1 if y_ind > 0 else y_ind]
        y_ind_u = [y_ind + 1 if y_ind < N2 - 1 else y_ind]
        z_ind_l = [z_ind - 1 if z_ind > 0 else z_ind]
        z_ind_u = [z_ind + 1 if z_ind < N3 - 1 else z_ind]

        x_ind_v = np.unique(np.vstack((x_ind_l, x_ind, x_ind_u)))
        y_ind_v = np.unique(np.vstack((y_ind_l, y_ind, y_ind_u)))
        z_ind_v = np.unique(np.vstack((z_ind_l, z_ind, z_ind_u)))

        x_ind, y_ind, z_ind = np.meshgrid(x_ind_v, y_ind_v, z_ind_v)

        return x_ind.reshape(-1, 1), y_ind.reshape(-1, 1), z_ind.reshape(-1, 1)

    def EP_1D(mu, Sigma, Threshold):
        '''
        This function computes the excursion probability
        :param mu:
        :param Sigma:
        :param Threshold:
        :return:
        '''
        EP = np.zeros_like(mu)
        for i in range(EP.shape[0]):
            EP[i] = norm.cdf(Threshold, mu[i], Sigma[i, i])
        return EP

    def find_next_EIBV_1D(x_cand, y_cand, z_cand, x_now, y_now, z_now,
                          x_pre, y_pre, z_pre, N1, N2, N3, Sig, mu, tau, Threshold):

        id = []
        dx1 = x_now - x_pre
        dy1 = y_now - y_pre
        dz1 = z_now - z_pre
        vec1 = np.array([dx1, dy1, dz1])
        for i in x_cand:
            for j in y_cand:
                for z in z_cand:
                    if i == x_now and j == y_now and z == z_now:
                        continue
                    dx2 = i - x_now
                    dy2 = j - y_now
                    dz2 = z - z_now
                    vec2 = np.array([dx2, dy2, dz2])
                    if np.dot(vec1, vec2) >= 0:
                        id.append(ravel_index([i, j, z], N1, N2, N3))
                    else:
                        continue
        id = np.unique(np.array(id))

        M = len(id)
        noise = tau ** 2
        R = np.diagflat(noise)  # diag not anymore support constructing matrix from vector
        N = N1 * N2 * N3
        eibv = []
        for k in range(M):
            F = np.zeros([1, N])
            F[0, id[k]] = True
            eibv.append(EIBV_1D(Threshold, mu, Sig, F, R))
        ind_desired = np.argmin(np.array(eibv))
        x_next, y_next, z_next = unravel_index(id[ind_desired], N1, N2, N3)

        return x_next, y_next, z_next



if __name__ == "__main__":
    A = Grid()
    B = GP(A)
    C = PathPlanner(B)






