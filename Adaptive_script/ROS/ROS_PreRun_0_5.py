#!/usr/bin/env python3
# license removed for brevity
# Adaptive sampling group of NTNU

# =============== USR SECTION =================
from usr_func import *


lat4, lon4 = 63.446905, 10.419426  # right bottom corner
origin = [lat4, lon4]
# distance = 100
distance = 1000
depth_obs = 0.5  # planned depth to be observed
box = BBox(lat4, lon4, distance, 60)

N1 = 25  # number of grid points along north direction
N2 = 25  # number of grid points along east direction
N3 = 1
N = N1 * N2  # total number of grid points

XLIM = [0, distance]
YLIM = [0, distance]
x = np.linspace(XLIM[0], XLIM[1], N1)
y = np.linspace(YLIM[0], YLIM[1], N2)
z = np.array(depth_obs).reshape(-1, 1)
xm, ym = np.meshgrid(x, y)
xv = xm.reshape(-1, 1)  # sites1v is the vectorised version
yv = ym.reshape(-1, 1)
dx = x[1] - x[0]
dy = y[1] - y[0]

coordinates = getCoordinates(box, N1, N2, dx, 60)

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
if depth_obs / 0.5 % 2 == 0:
    odd = False
    F_sampling = F_even
else:
    odd = True
    F_sampling = F_odd

for j in range(F_sampling.shape[0]):
    loc = F_sampling[j, :] @ coordinates
    lat = loc[0]
    lon = loc[1]
    depth = depth_obs
    Path_PreRun.append([deg2rad(lat), deg2rad(lon), depth])
    if (j + 1) % 2 == 0:
        Path_PreRun.append([deg2rad(lat), deg2rad(lon), 0])


N_steps = len(Path_PreRun)
print("Total steps is ", N_steps)

# '''
# Export data to local file named "data.txt" for the later use
# '''

print(Path_PreRun)
# === TEST CODE ====
# figpath = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Nidelva/Path/"
# for i in range(len(Path_PreRun)):
#     plt.figure(figsize=(5, 5))
#     plt.plot(coordinates[:, 1], coordinates[:, 0], 'k.')
#     plt.plot(rad2deg(Path_PreRun[i][1]), rad2deg(Path_PreRun[i][0]), 'r.')
#     plt.savefig(figpath + "P_{:03d}.pdf".format(i))
#     plt.show()


today = date.today()
d1 = today.strftime("%d_%m_%Y")
datafolder = os.getcwd() + "/" + d1
if not os.path.exists(datafolder):
    os.mkdir(datafolder)

datapath = datafolder + "/Data/"
if not os.path.exists(datapath):
    os.mkdir(datapath)
    print(datapath + " is constructed successfully")

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
    np.savetxt(datapath + "data.txt", data, delimiter=",")

logfile = open(datapath + "log.txt", "w+")

# =============== ROS SECTION =================

import rospy
import numpy as np
from auv_handler import AuvHandler
import imc_ros_interface
from imc_ros_interface.msg import Temperature, Salinity, EstimatedState


class PreRun:
    def __init__(self):
        self.node_name = 'PreRun'
        rospy.init_node(self.node_name, anonymous=True)
        self.rate = rospy.Rate(1)  # 1Hz

        self.auv_handler = AuvHandler(self.node_name, "PreRun")

        rospy.Subscriber("/Vehicle/Out/Temperature_filtered", Temperature, self.TemperatureCB)
        rospy.Subscriber("/Vehicle/Out/Salinity_filtered", Salinity, self.SalinityCB)
        rospy.Subscriber("/Vehicle/Out/EstimatedState_filtered", EstimatedState, self.EstimatedStateCB)

        self.speed = 1.5  # m/s
        self.depth = 0.0  # meters
        self.last_state = "unavailable"
        self.rate.sleep()
        # self.auv_handler.setWaypoint(deg2rad(lat4), deg2rad(lon4))
        self.auv_handler.setWaypoint(Path_PreRun[0][0], Path_PreRun[0][1], Path_PreRun[0][2])

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

    def run(self):
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
                    save_data(datapath, data_timestamp, data_lat, data_lon, data_x, data_y, data_z, data_salinity,
                              data_temperature)
                    s = "Data saved {:d} times\n".format(counter_total_datasaved)
                    print(s)
                    logfile.write(s)
                    counter_datasave = 0
                    counter_total_datasaved = counter_total_datasaved + 1
                timestamp = timestamp + 1
                counter_datasave = counter_datasave + 1

                if self.auv_handler.getState() == "waiting":
                    print("Arrived the current location")
                    logfile.write("Arrived the current location\n")
                    save_data(datapath, data_timestamp, data_lat, data_lon, data_x, data_y, data_z, data_salinity,
                              data_temperature)
                    counter_total_datasaved = counter_total_datasaved + 1
                    print("Data saved {:02d} times".format(counter_total_datasaved))
                    if counter < N_steps:
                        print("Move to new way point, lat: {:.2f}, lon: {:.2f}, depth: {:.2f}".format(
                            Path_PreRun[counter][0], Path_PreRun[counter][1], Path_PreRun[counter][-1]))
                        logfile.write("Move to new way point, lat: {:.2f}, lon: {:.2f}, depth: {:.2f}\n".format(
                            Path_PreRun[counter][0], Path_PreRun[counter][1], Path_PreRun[counter][-1]))
                        self.auv_handler.setWaypoint(Path_PreRun[counter][0], Path_PreRun[counter][1],
                                                     Path_PreRun[counter][-1])

                        if Path_PreRun[counter][-1] == 0:
                            for i in range(45):
                                print(i)
                                print("Sleep {:d} seconds".format(i))
                                logfile.write("Sleep {:d} seconds\n".format(i))
                                self.auv_handler.spin()  # publishes the reference, stay on the surface
                                self.rate.sleep()  #

                        counter = counter + 1
                    else:
                        save_data(datapath, data_timestamp, data_lat, data_lon, data_x, data_y, data_z, data_salinity,
                                  data_temperature)
                        counter_total_datasaved = counter_total_datasaved + 1
                        s = "Data saved {:d} times\n".format(counter_total_datasaved)
                        print(s)
                        logfile.write(s)
                        logfile.write("Mission completed !!! \n")
                        rospy.signal_shutdown("Mission completed!!!")

                self.last_state = self.auv_handler.getState()
                self.auv_handler.spin()

            self.rate.sleep()


if __name__ == "__main__":
    go = PreRun()
    try:
        go.run()
    except rospy.ROSInterruptException:
        pass

logfile.close()
