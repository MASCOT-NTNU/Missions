import time
import matplotlib.pyplot as plt
import h5py
import numpy as np
from scipy.stats import mvn, norm
# from Adaptive_script.Porto.Grid import GridPoly
from Adaptive_script.Porto.GP import GP_Poly
import os

class Simulator(GP_Poly):
    ind_start, ind_now, ind_pre, ind_cand, ind_next = [0, 0, 0, 0, 0] # only use index to make sure it is working properly
    mu_cond, Sigma_cond, F = [None, None, None] # conditional mean
    mu_prior, Sigma_prior = [None, None] # prior mean and covariance matrix
    travelled_waypoints = None # track how many waypoins have been explored
    data_path_waypoint = [] # save the waypoint to compare
    data_path_lat = [] # save the waypoint lat
    data_path_lon = [] # waypoint lon
    data_path_depth = [] # waypoint depth
    Total_waypoints = 100 # total number of waypoints to be explored
    figpath = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Porto/Setup/Simulation/fig/"
    # data_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/D2_HDF/Merged/Merged_09_North_Calm.h5"
    counter_plot_simulation = 0 # track the plot, will be deleted

    def __init__(self, Simulation = False):
        self.SIMULATION = Simulation
        print("Simulation mode: ", self.SIMULATION)
        if self.SIMULATION:
            self.checkFolder()
        GP_Poly.__init__(self, debug = False)
        self.distance_neighbours = np.sqrt(GP_Poly.distance_poly ** 2 + (GP_Poly.depth_obs[1] - GP_Poly.depth_obs[0]) ** 2)
        print("range of neighbours: ", self.distance_neighbours)
        self.travelled_waypoints = 0
        self.load_prior()
        self.plot_prior()
        if self.SIMULATION:
            self.move_to_starting_loc()
            self.run()

    def checkFolder(self):
        i = 0
        while os.path.exists(self.figpath + "P%s" % i):
            i += 1
        self.figpath = self.figpath + "P%s/" % i
        if not os.path.exists(self.figpath):
            print(self.figpath + " is created")
            os.mkdir(self.figpath)
        else:
            print(self.figpath + " is already existed")

    def load_prior(self):
        self.lat_loc = self.lat_selected[:, 0] # select the top layer for simulation
        self.lon_loc = self.lon_selected[:, 0]
        # self.lat_loc = self.lat_selected[:, 0]
        self.salinity_loc = self.salinity_selected[:, 0]
        self.mu_prior = self.salinity_loc.reshape(-1, 1)
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
        plt.figure(figsize = (40, 10))
        plt.subplot(121)
        plt.scatter(self.lon_loc, self.lat_loc, c = self.mu_prior, s = 500, cmap = "Paired")
        plt.colorbar()
        plt.title("Prior mean")
        plt.xlabel("Lon [deg]")
        plt.ylabel("Lat [deg]")
        # plt.subplot(132)
        # plt.imshow(self.Sigma_prior)
        # plt.colorbar()
        plt.subplot(122)
        plt.scatter(self.lon_loc, self.lat_loc, c=self.mu_real, s = 500, cmap="Paired")
        plt.colorbar()
        plt.title("true mean")
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
        # self.ind_start = (np.abs(EP_Prior - ep_criterion)).argmin()
        self.ind_start = 10
        self.ind_next = self.ind_start
        self.updateF(self.ind_next)
        self.data_path_lat.append(self.lat_loc[self.ind_start])
        self.data_path_lon.append(self.lon_loc[self.ind_start])
        self.updateWaypoint()
        # self.updateWaypoint()

    def find_candidates_loc(self):
        '''
        find the candidates location based on distance coverage
        '''
        delta_x, delta_y = self.latlon2xy(self.lat_loc, self.lon_loc, self.lat_loc[self.ind_now], self.lon_loc[self.ind_now]) # using the distance
        self.distance_vector = np.sqrt(delta_x ** 2 + delta_y ** 2)
        self.ind_cand = np.where(self.distance_vector <= self.distance_neighbours + self.distanceTolerance)[0]

    def GPupd(self, y_sampled):
        t1 = time.time()
        C = self.F @ self.Sigma_cond @ self.F.T + self.R_sal
        self.mu_cond = self.mu_cond + self.Sigma_cond @ self.F.T @ np.linalg.solve(C, (y_sampled - self.F @ self.mu_cond))
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
        id = [] # ind vector for containing the filtered desired candidate location
        t1 = time.time()
        dx1, dy1 = self.latlon2xy(self.lat_loc[self.ind_now], self.lon_loc[self.ind_now], self.lat_loc[self.ind_pre], self.lon_loc[self.ind_pre])
        lat_cand_plot = []
        lon_cand_plot = []
        vec1 = np.array([dx1, dy1])
        print("vec1 :", vec1)
        for i in range(len(self.ind_cand)):
            if self.ind_cand[i] != self.ind_now:
                dx2, dy2 = self.latlon2xy(self.lat_loc[self.ind_cand[i]], self.lon_loc[self.ind_cand[i]], self.lat_loc[self.ind_now], self.lon_loc[self.ind_now])
                vec2 = np.array([dx2, dy2])
                print("vec2: ", vec2)
                if np.dot(vec1, vec2) > 0:
                    print("Product: ", np.dot(vec1, vec2))
                    id.append(self.ind_cand[i])
                    lat_cand_plot.append(self.lat_loc[self.ind_cand[i]])
                    lon_cand_plot.append(self.lon_loc[self.ind_cand[i]])
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

        if len(eibv) == 0: # in case it is in the corner and not found any valid candidate locations
            print("No valid candidates found: ")
            self.ind_next = np.where(self.EP_1D(mu, Sig, self.Threshold_S) == .5)[0][0]
        else:
            print("The EIBV for the candidate location: ", np.array(eibv))
            self.ind_next = self.ind_cand[np.argmin(np.array(eibv))]
        print("ind_next: ", self.ind_next)

        if self.SIMULATION:
            plt.figure(figsize=(40, 40))
            plt.subplot(221)
            plt.scatter(self.lon_loc, self.lat_loc, c=self.mu_cond, vmin=15, vmax=36, s = 500, alpha = 1, cmap="Paired")
            plt.plot(self.lon_loc[self.ind_pre], self.lat_loc[self.ind_pre], 'kx', markersize=10)
            plt.plot(self.lon_loc[self.ind_cand], self.lat_loc[self.ind_cand], 'go', markersize=10)
            plt.plot(self.lon_loc[self.ind_next], self.lat_loc[self.ind_next], 'rs', markersize=10)
            plt.plot(self.data_path_lon, self.data_path_lat, 'b-', linewidth = 5)
            plt.colorbar()
            plt.xlabel("Lon [deg]")
            plt.ylabel("Lat [deg]")
            plt.title("Mu cond")

            plt.subplot(222)
            err = np.sqrt(np.diag(self.Sigma_cond))
            plt.scatter(self.lon_loc, self.lat_loc, c=err, s = 500, alpha = .5)
            plt.plot(self.lon_loc[self.ind_pre], self.lat_loc[self.ind_pre], 'kx', markersize=10)
            plt.plot(self.lon_loc[self.ind_cand], self.lat_loc[self.ind_cand], 'go', markersize=10)
            plt.plot(self.lon_loc[self.ind_next], self.lat_loc[self.ind_next], 'rs', markersize=10)
            plt.plot(self.data_path_lon, self.data_path_lat, 'b-', linewidth = 5)
            plt.colorbar()
            plt.xlabel("Lon [deg]")
            plt.ylabel("Lat [deg]")
            plt.title("Prediction error")

            plt.subplot(223)
            plt.scatter(self.lon_loc, self.lat_loc, c=self.mu_real, vmin=15, vmax=36, s = 500, alpha = 1, cmap="Paired")
            plt.plot(self.lon_loc[self.ind_pre], self.lat_loc[self.ind_pre], 'kx', markersize=10)
            plt.plot(self.lon_loc[self.ind_cand], self.lat_loc[self.ind_cand], 'go', markersize=10)
            plt.plot(self.lon_loc[self.ind_next], self.lat_loc[self.ind_next], 'rs', markersize=10)
            plt.plot(self.data_path_lon, self.data_path_lat, 'b-', linewidth = 5)
            plt.colorbar()
            plt.xlabel("Lon [deg]")
            plt.ylabel("Lat [deg]")
            plt.title("True field")

            plt.subplot(224)
            plt.scatter(lon_cand_plot, lat_cand_plot, c=eibv, s=500, cmap="Paired")
            plt.plot(self.lon_loc[self.ind_pre], self.lat_loc[self.ind_pre], 'kx', markersize=10)
            plt.plot(self.lon_loc[self.ind_now], self.lat_loc[self.ind_now], 'bx', markersize=30)
            plt.plot(self.lon_loc[self.ind_cand], self.lat_loc[self.ind_cand], 'go', markersize=10)
            plt.plot(self.lon_loc[self.ind_next], self.lat_loc[self.ind_next], 'r*', markersize=30)
            plt.plot(self.data_path_lon, self.data_path_lat, 'b-', linewidth = 5)
            plt.colorbar()
            plt.xlabel("Lon [deg]")
            plt.ylabel("Lat [deg]")
            plt.title("EIBV path planning")

            plt.savefig(self.figpath + "P_{:04d}.png".format(self.counter_plot_simulation))
            self.counter_plot_simulation = self.counter_plot_simulation + 1
            plt.close("all")
            pass
        self.data_path_lat.append(self.lat_loc[self.ind_next])
        self.data_path_lon.append(self.lon_loc[self.ind_next])
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
        # print("Arrived the current location")
        sal_sampled = self.mu_real[self.ind_next]  # take the past ten samples and average
        self.GPupd(sal_sampled) # update the field when it arrives the specified location
        self.find_candidates_loc()
        self.find_next_EIBV_1D(self.mu_cond, self.Sigma_cond)
        self.updateWaypoint()
        self.travelled_waypoints += 1
        print(self.counter_plot_simulation)
        if self.travelled_waypoints >= self.Total_waypoints:
            print("Mission completed!!!")
            return False
        else:
            self.run()

# a = Simulator(Simulation=False)
a = Simulator(Simulation=True)
print("Mission complete!!!")




