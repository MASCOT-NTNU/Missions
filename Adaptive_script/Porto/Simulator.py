import time
import matplotlib.pyplot as plt
import h5py
import numpy as np
from scipy.stats import mvn, norm
# from Adaptive_script.Porto.Grid import GridPoly
from Adaptive_script.Porto.GP import GP_Poly


class Simulator(GP_Poly):
    lat_start, lon_start, depth_start = [None, None, None]
    lat_now, lon_now, depth_now = [None, None, None]
    lat_pre, lon_pre, depth_pre = [None, None, None]
    lat_cand, lon_cand, depth_cand = [None, None, None]
    lat_next, lon_next, depth_next = [None, None, None]
    mu_cond, Sigma_cond, F = [None, None, None]
    mu_prior, Sigma_prior = [None, None]
    travelled_waypoints = None
    data_path_waypoint = []
    Total_waypoints = 10
    data_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/D2_HDF/Merged/Merged_09_North_Calm.h5"

    def __init__(self, Simulation = False):
        self.SIMULATION = Simulation
        print("Simulation mode: ", self.SIMULATION)
        GP_Poly.__init__(self, debug = False)
        self.distance_neighbours = np.sqrt(GP_Poly.distance_poly ** 2 + (GP_Poly.depth_obs[1] - GP_Poly.depth_obs[0]) ** 2)
        print("range of neighbours: ", self.distance_neighbours)
        self.travelled_waypoints = 0
        self.checkSingularity()
        self.load_prior()
        self.plot_prior()
        # self.move_to_starting_loc()
        # self.run()

    def checkSingularity(self):
        import numpy as np
        x, y = self.latlon2xy(self.lat, self.lon, self.lat_origin, self.lon_origin)
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        grid = np.hstack((y, x))
        import scipy.spatial.distance as scdist
        t = scdist.cdist(grid, grid)
        from Adaptive_script.Porto.usr_func import Matern_cov
        S = Matern_cov(4, 4.5/600, t)
        print(["The covariance matrix is " + "Positive " if np.all(np.linalg.eigvals(S) > 0) else "Singular"])

    def load_prior(self):
        self.mu_prior = self.salinity[0, :].reshape(-1, 1)
        self.Sigma_prior = self.Sigma_sal
        self.mu_real = self.mu_prior + np.linalg.cholesky(self.Sigma_prior) @ np.random.randn(len(self.mu_prior)).reshape(-1, 1)
        self.mu_cond = self.mu_prior
        self.Sigma_cond = self.Sigma_prior
        self.N = len(self.mu_prior)
        print("Prior for salinity is loaded correctly!!!")
        print("mu_prior shape is: ", self.mu_prior.shape)
        print("Sigma_prior shape is: ", self.Sigma_prior.shape)
        print("mu_cond shape is: ", self.mu_cond.shape)
        print("Sigma_cond shape is: ", self.Sigma_cond.shape)
        print("N: ", self.N)

    def plot_prior(self):
        plt.figure(figsize = (30, 15))
        plt.subplot(131)
        plt.scatter(self.lon, self.lat, c = self.mu_prior, cmap = "Paired")
        plt.colorbar()
        plt.title("Prior mean")
        plt.xlabel("Lon [deg]")
        plt.ylabel("Lat [deg]")
        plt.subplot(132)
        plt.imshow(self.Sigma_prior)
        plt.colorbar()
        plt.subplot(133)
        plt.scatter(self.lon, self.lat, c=self.mu_real, cmap="Paired")
        plt.colorbar()
        plt.title("Prior mean")
        plt.xlabel("Lon [deg]")
        plt.ylabel("Lat [deg]")
        plt.show()

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
        self.ind_desired = ind
        self.updateF(ind)
        self.data_path_waypoint.append(ind)
        self.lat_next, self.lon_next, self.depth_next = self.lat_start, self.lon_start, self.depth_start
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

        M = len(id)
        eibv = []
        for k in range(M):
            F = np.zeros([1, self.N])
            F[0, id[k]] = True
            eibv.append(self.EIBV_1D(self.Threshold_S, mu, Sig, F, self.R_sal))
        ind_desired = np.argmin(np.array(eibv))
        t2 = time.time()

        if self.SIMULATION:
            pass
        self.lat_next = self.lat_cand[ind_desired]
        self.lon_next = self.lon_cand[ind_desired]
        self.depth_next = self.depth_cand[ind_desired]
        print("Found next waypoint: ", self.lat_next, self.lon_next, self.depth_next)
        print("Finding next waypoint takes: ", t2 - t1)
        self.ind_desired = self.ind_cand[ind_desired]
        self.updateF(self.ind_desired)
        self.data_path_waypoint.append(self.ind_desired)

    def updateWaypoint(self):
        self.lat_pre, self.lon_pre, self.depth_pre = self.lat_now, self.lon_now, self.depth_now
        self.lat_now, self.lon_now, self.depth_now = self.lat_next, self.lon_next, self.depth_next

    def run(self):
        print("Arrived the current location")
        print(self.salinity_loc[self.ind_desired])
        sal_sampled = self.salinity_loc[self.ind_desired] + np.random.rand()  # take the past ten samples and average
        print(sal_sampled)
        self.GPupd(sal_sampled) # update the field when it arrives the specified location
        self.find_candidates_loc()
        self.find_next_EIBV_1D(self.mu_cond, self.Sigma_cond)
        self.updateWaypoint()
        self.travelled_waypoints += 1

        if self.travelled_waypoints >= self.Total_waypoints:
            print("Mission completed!!!")
            return False
        else:
            self.run()



# if __name__ == "__main__":
    # a = PathPlanner()
a = Simulator()
print("Mission complete!!!")




