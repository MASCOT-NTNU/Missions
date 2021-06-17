#!/usr/bin/env python3
# license removed for brevity
# Adaptive sampling group of NTNU


#%% =============== USR SECTION =================


from Prior import *

today = date.today()
d1 = today.strftime("%d_%m_%Y")
datafolder = os.getcwd() + "/" + d1
datapath = datafolder + "/Data/"

beta0 = np.loadtxt(datapath + 'beta0.txt', delimiter=",")
beta1 = np.loadtxt(datapath + 'beta1.txt', delimiter=",")
mu_prior_sal = np.loadtxt(datapath + 'mu_prior_sal.txt', delimiter=",")
mu_prior_temp = np.loadtxt(datapath + 'mu_prior_temp.txt', delimiter=",")
print("Congrats!!! Prior is built successfully!!!")
print("Fitted beta0: \n", beta0)
print("Fitted beta1: \n", beta1)


## Section I: Setup parameters
sigma_sal = np.sqrt(4)  # scaling coef in matern kernel for salinity
tau_sal = np.sqrt(.3)  # iid noise
Threshold_S = 23  # 20

sigma_temp = np.sqrt(0.5)
tau_temp = np.sqrt(.1)
Threshold_T = 10.5

eta = 4.5 / 400  # coef in matern kernel
ksi = 1000 / 24 / 0.5  # scaling factor in 3D
N_steps = 60  # number of steps desired to conduct


## Section II: Set up the waypoint and grid
nx = 25  # number of grid points along x-direction
ny = 25  # number of grid points along y-direction
L = 1000  # distance of the square
alpha = -60  # angle of the inclined grid
distance = L / (nx - 1)
distance_depth = depth_obs[1] - depth_obs[0]
gridx, gridy = rotateXY(nx, ny, distance, alpha)
grid = []
for k in depth_obs:
    for i in range(gridx.shape[0]):
        for j in range(gridx.shape[1]):
            grid.append([gridx[i, j], gridy[i, j], k])
grid = np.array(grid)

H = compute_H(grid, ksi)

Sigma_prior = Matern_cov(sigma_sal, eta, H)

EP_prior = EP_1D(mu_prior_sal, Sigma_prior, Threshold_S)


## Part II : Path planning

path = []
path_cand = []
coords = []
mu = []
Sigma = []
t_elapsed = []

loc = find_starting_loc(EP_prior, N1, N2, N3)
xstart, ystart, zstart = loc

xnow, ynow, znow = xstart, ystart, zstart
xpre, ypre, zpre = xnow, ynow, znow

lat_start, lon_start = xy2latlon(xstart, ystart, origin, distance, alpha)
path.append([xnow, ynow, znow])
coords.append([lat_start, lon_start])

print("The starting location is [{:.2f}, {:.2f}]".format(lat_start, lon_start))

mu_cond = mu_prior_sal
Sigma_cond = Sigma_prior
mu.append(mu_cond)
Sigma.append(Sigma_cond)

noise = tau_sal ** 2
R = np.diagflat(noise)


#%% =============== ROS SECTION =================


data_timestamp = []
data_temperature = []
data_salinity = []
data_x = []
data_y = []
data_z = []
data_lat = []
data_lon = []


def save_data(datapath, timestamp, data_lat, data_lon, data_x, data_y, data_z, data_salinity, data_temperature):
    data = np.hstack((np.array(timestamp).reshape(-1, 1),
                      np.array(data_lat).reshape(-1, 1),
                      np.array(data_lon).reshape(-1, 1),
                      np.array(data_x).reshape(-1, 1),
                      np.array(data_y).reshape(-1, 1),
                      np.array(data_z).reshape(-1, 1),
                      np.array(data_z).reshape(-1, 1),
                      np.array(data_salinity).reshape(-1, 1),
                      np.array(data_temperature).reshape(-1, 1)))
    np.savetxt(datapath + "data_ES3D1.txt", data, delimiter = ",")


import rospy
import numpy as np
from auv_handler import AuvHandler

import imc_ros_interface
from imc_ros_interface.msg import Temperature, Salinity, EstimatedState


class ES3D1:
    def __init__(self):
        self.node_name = 'ES3D1'
        rospy.init_node(self.node_name, anonymous=True)
        self.rate = rospy.Rate(1)  # 1Hz

        self.auv_handler = AuvHandler(self.node_name, "ES3D1")

        rospy.Subscriber("/Vehicle/Out/Temperature_filtered", Temperature, self.TemperatureCB)
        rospy.Subscriber("/Vehicle/Out/Salinity_filtered", Salinity, self.SalinityCB)
        rospy.Subscriber("/Vehicle/Out/EstimatedState_filtered", EstimatedState, self.EstimatedStateCB)

        self.speed = 2.0  # m/s
        self.depth = 0.0  # meters
        self.last_state = "unavailable"
        self.rate.sleep()
        self.auv_handler.setWaypoint(deg2rad(lat4), deg2rad(lon4))
        # self.auv_handler.setWaypoint(lat_start, lon_start, depth_obs[zstart])
        self.init = True
        self.currentTemperature = 0.0
        self.currentSalinity = 0.0
        self.vehicle_pos = [0, 0, 0]

    def TemperatureCB(self, msg):
        self.currentTemperature = msg.value.data

    def SalinityCB(self, msg):
        self.currentSalinity = msg.value.data

    def EstimatedStateCB(self, msg):
        offset_north = msg.lat.data - deg2rad(lat4)
        offset_east = msg.lon.data - deg2rad(lon4)
        circumference = 40075000.0
        N = offset_north * circumference / (2.0 * np.pi)
        E = offset_east * circumference * np.cos(deg2rad(lat4)) / (2.0 * np.pi)
        D = msg.z.data
        self.vehicle_pos = [N, E, D]

    def move_to_start_loc(self):
        while not rospy.is_shutdown():
            if self.init:
                if self.auv_handler.getState() == "waiting":
                    print("We are at the origin, will move to the starting location")
                    self.auv_handler.setWaypoint(lat_start, lon_start, depth_obs[zstart])
                self.last_state = self.auv_handler.getState()
                self.auv_handler.spin()
            self.rate.sleep()

    def run(self):
        # ===== Path planning section =====

        for j in range(N_steps):
            xcand, ycand, zcand = find_candidates_loc(xnow, ynow, znow, N1, N2, N3)

            t1 = time.time()
            xnext, ynext, znext = find_next_EIBV_1D(xcand, ycand, zcand,
                                                    xnow, ynow, znow,
                                                    xpre, ypre, zpre,
                                                    N1, N2, N3, Sigma_cond,
                                                    mu_cond, tau_sal, Threshold_S)
            t2 = time.time()
            t_elapsed.append(t2 - t1)
            print("It takes {:.2f} seconds to compute the next waypoint".format(t2 - t1))
            print("next is ", xnext, ynext, znext)
            lat_next, lon_next = xy2latlon(xnext, ynext, origin, distance, alpha)
            depth_next = depth_obs[znext]
            # ====
            ind_next = ravel_index([xnext, ynext, znext], N1, N2, N3)
            F = np.zeros([1, N])
            F[0, ind_next] = True

            # ===== End of path planning section =====

            counter = 0
            counter_datasave = 0
            counter_total_datasaved = 0
            timestamp = 0
            while not rospy.is_shutdown():
                if self.init:
                    data_timestamp.append(timestamp)
                    data_temperature.append(self.currentTemperature)
                    data_salinity.append(self.currentSalinity)
                    data_x.append(self.vehicle_pos[0])
                    data_y.append(self.vehicle_pos[1])
                    data_z.append(self.vehicle_pos[-1])
                    data_lat.append(lat4)
                    data_lon.append(lon4)
                    if counter_datasave >= 10:
                        save_data(datapath, data_timestamp, data_lat, data_lon, data_x, data_y, data_z, data_salinity, data_temperature)
                        print("Data saved {:d} times".format(counter_total_datasaved))
                        counter_datasave = 0
                        counter_total_datasaved = counter_total_datasaved + 1
                    timestamp = timestamp + 1
                    counter_datasave = counter_datasave + 1

                    if self.auv_handler.getState() == "waiting":
                        print("Arrived the current location")
                        save_data(datapath, data_timestamp, data_lat, data_lon, data_x, data_y, data_z, data_salinity, data_temperature)
                        counter_total_datasaved = counter_total_datasaved + 1
                        print("Data saved {:02d} times".format(counter_total_datasaved))
                        print("Move to new way point, lat: {:.2f}, lon: {:.2f}, depth: {:.2f}".format(lat_next, lon_next, depth_next))

                        sal_sampled = np.mean(data_salinity[-10:]) # take the past ten samples and average
                        mu_cond, Sigma_cond = GPupd(mu_cond, Sigma_cond, R, F, sal_sampled)

                        xpre, ypre, zpre = xnow, ynow, znow
                        xnow, ynow, znow = xnext, ynext, znext

                        path.append([xnow, ynow, znow])
                        path_cand.append([xcand, ycand, zcand])
                        coords.append([lat_next, lon_next])
                        mu.append(mu_cond)
                        Sigma.append(Sigma_cond)
                        self.auv_handler.setWaypoint(lat_next, lon_next, depth_next)


if __name__ == "__main__":
    go = ES3D1()
    try:
        go.move_to_start_loc()
        go.run()
    except rospy.ROSInterruptException:
        pass


# %% Save data section
datapath = '/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Nidelva/SimulationData/'
data_path = np.array(path)
data_path_cand = np.array(path_cand)
data_coords = np.array(coords)
data_mu = np.array(mu)
data_Sigma = np.array(Sigma)
data_perr = np.zeros_like(data_mu)
for i in range(data_Sigma.shape[0]):
    data_perr[i, :, :] = np.diag(data_Sigma[i, :, :]).reshape(-1, 1)
data_t_elapsed = np.array(t_elapsed)

shape_path = data_path.shape
shape_path_cand = data_path_cand.shape
shape_coords = data_coords.shape
shape_mu = data_mu.shape
shape_perr = data_perr.shape
shape_t_elapsed = data_t_elapsed.shape

np.savetxt(datapath + "shape_path.txt", shape_path, delimiter=", ")
np.savetxt(datapath + "shape_path_cand.txt", shape_path_cand, delimiter=", ")
np.savetxt(datapath + "shape_coords.txt", shape_coords, delimiter=", ")
np.savetxt(datapath + "shape_mu.txt", shape_mu, delimiter=", ")
np.savetxt(datapath + "shape_perr.txt", shape_perr, delimiter=", ")
np.savetxt(datapath + "shape_t_elapsed.txt", shape_t_elapsed, delimiter=", ")

np.savetxt(datapath + "data_path.txt", data_path.reshape(-1, 1), delimiter=", ")
np.savetxt(datapath + "data_path_cand.txt", data_path_cand.reshape(-1, 1), delimiter=", ")
np.savetxt(datapath + "data_coords.txt", data_coords.reshape(-1, 1), delimiter=", ")
np.savetxt(datapath + "data_mu.txt", data_mu.reshape(-1, 1), delimiter=", ")
np.savetxt(datapath + "data_perr.txt", data_perr.reshape(-1, 1), delimiter=", ")
np.savetxt(datapath + "data_t_elapsed.txt", data_t_elapsed.reshape(-1, 1), delimiter=", ")

