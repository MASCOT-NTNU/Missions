#!/usr/bin/env python3
# license removed for brevity
# Adaptive sampling group of NTNU


# =============== USR SECTION =================
import matplotlib.pyplot as plt

from usr_func import *
lat4, lon4 = 63.446905, 10.419426 # right bottom corner
origin = [lat4, lon4]
distance = 1000
depth_obs = [0.5, 1.0, 1.5, 2.0, 2.5] # planned depth to be observed
box = BBox(lat4, lon4, distance, 60)

N1 = 25 # number of grid points along north direction
N2 = 25 # number of grid points along east direction
N3 = 5 # number of layers in the depth dimension
N = N1 * N2 * N3 # total number of grid points

XLIM = [0, 1000]
YLIM = [0, 1000]
ZLIM = [0.5, 2.5]
x = np.linspace(XLIM[0], XLIM[1], N1)
y = np.linspace(YLIM[0], YLIM[1], N2)
z = np.array([0.5, 1.0, 1.5, 2.0, 2.5]).reshape(-1, 1)
xm, ym, zm = np.meshgrid(x, y, z)
xv = xm.reshape(-1, 1) # sites1v is the vectorised version
yv = ym.reshape(-1, 1)
zv = zm.reshape(-1, 1)
dx = x[1] - x[0]
dy = y[1] - y[0]
dz = z[1] - z[0]

coordinates= getCoordinates(box, N1, N2, dx, 60)

F_odd = np.zeros([6, coordinates.shape[0]])
F_even = np.zeros([6, coordinates.shape[0]])
cnt = 0
cnt_i = 0
for i in [4, 12, 20]:
    for j in [0, N1 - 1]:
        if cnt_i % 2 == 0:
            ind_odd = ravel_index([j, i, 0], N1, N2, N3)
            ind_even = ravel_index([N1 - j - 1, 20 - 8 * cnt_i, 0], N1, N2, N3)
        else:
            ind_odd = ravel_index([N1 - j - 1, i, 0], N1, N2, N3)
            ind_even = ravel_index([j, 20 - 8 * cnt_i, 0], N1, N2, N3)

        F_odd[cnt, ind_odd] = True
        F_even[cnt, ind_even] = True
        cnt = cnt + 1
    cnt_i = cnt_i + 1


Path_PreRun = []
for i in range(len(depth_obs)):
    if i % 2 == 0:
        for j in range(F_even.shape[0]):
            loc = F_even[j, :] @ coordinates
            lat = loc[0]
            lon = loc[1]
            depth = depth_obs[i]
            Path_PreRun.append([deg2rad(lat), deg2rad(lon), depth])
    else:
        for j in range(F_odd.shape[0]):
            loc = F_odd[j, :] @ coordinates
            lat = loc[0]
            lon = loc[1]
            depth = depth_obs[i]
            Path_PreRun.append([deg2rad(lat), deg2rad(lon), depth])
    Path_PreRun.append([deg2rad(lat), deg2rad(lon), 0])

N_steps = len(Path_PreRun)
print("Total steps is ", N_steps)

# '''
# Export data to local file named "data.txt" for the later use
# '''

print(Path_PreRun)

# datapath = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Nidelva/June17/"

#%%
# figpath = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Nidelva/Path/"
# for i in range(len(Path_PreRun)):
#     plt.figure(figsize=(5, 5))
#     plt.plot(coordinates[:, 1], coordinates[:, 0], 'k.')
#     plt.plot(rad2deg(Path_PreRun[i][1]), rad2deg(Path_PreRun[i][0]), 'r.')
#     plt.savefig(figpath + "P_{:03d}.pdf".format(i))
#     plt.show()


# =============== ROS SECTION =================

import rospy
import numpy as np
from auv_handler import AuvHandler

import imc_ros_interface
from imc_ros_interface.msg import Temperature, Salinity, EstimatedState

class PreRun:
    def __init__(self):
        self.node_name = 'PreRun'
        rospy.init_node(self.node_name,anonymous=True)
        self.rate = rospy.Rate(1) # 1Hz

        self.auv_handler = AuvHandler(self.node_name,"PreRun")

        rospy.Subscriber("/Vehicle/Out/Temperature_filtered", Temperature, self.TemperatureCB)
        rospy.Subscriber("/Vehicle/Out/Salinity_filtered", Salinity, self.SalinityCB)
        rospy.Subscriber("/Vehicle/Out/EstimatedState_filtered", EstimatedState, self.EstimatedStateCB)

        self.speed = 2.0 #m/s
        self.depth = 0.0 #meters
        self.last_state = "unavailable"
        self.rate.sleep()
        self.auv_handler.setWaypoint(deg2rad(lat4), deg2rad(lon4))

        self.init = True
        self.currentTemperature = 0.0
        self.currentSalinity = 0.0
        self.vehicle_pos = [0, 0, 0]

    def TemperatureCB(self,msg):
        self.currentTemperature = msg.value.data

    def SalinityCB(self,msg):
        self.currentSalinity = msg.value.data

    def EstimatedStateCB(self,msg):
        offset_north = msg.lat.data - deg2rad(lat4)
        offset_east = msg.lon.data - deg2rad(lon4)
        circumference = 40075000.0
        N = offset_north * circumference / (2.0 * np.pi)
        E = offset_east * circumference * np.cos(deg2rad(lat4)) / (2.0 * np.pi)
        D = msg.z.data
        self.vehicle_pos = [N, E, D]

    def run(self):
        counter = 0
        while not rospy.is_shutdown():
            if self.init:
                print("The temperature is ", self.Temperature)
                print("The salinity is ", self.Salinity)
                print("The N E D is ", self.vehicle_pos)
                if self.auv_handler.getState() == "waiting":
                    print("Arrived the current location \n")
                    if counter < N_steps:
                        print("Move to new way point, lat: {:.2f}, lon: {:.2f}, depth: {:.2f}".format(Path_PreRun[counter][0], Path_PreRun[counter][1], Path_PreRun[counter][-1]))
                        self.auv_handler.setWaypoint(Path_PreRun[counter][0], Path_PreRun[counter][1], Path_PreRun[counter][-1])
                        if Path_PreRun[counter][-1] == 0:
                            for i in range(60):
                                print(i)
                                print("Sleep {:01d} seconds".format(i))
                                self.auv_handler.spin() # publishes the reference, stay on the surface
                                self.rate.sleep() # 
                        counter = counter + 1

                self.last_state = self.auv_handler.getState()
                self.auv_handler.spin()
            
            self.rate.sleep()

if __name__ == "__main__":
    go = PreRun()
    try:
        go.run()
    except rospy.ROSInterruptException:
        pass

