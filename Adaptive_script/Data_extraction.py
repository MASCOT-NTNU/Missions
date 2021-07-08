# datapath = '/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Nidelva/May27/Data/'
datapath = '/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Nidelva/July06/Data/'
figpath = '/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Nidelva/July06/Fig/'
import numpy as np
import pandas as pd
from datetime import datetime
import os

#% Data extraction from the raw data
rawTemp = pd.read_csv(datapath + "Temperature.csv", delimiter=', ', header=0, engine='python')
rawLoc = pd.read_csv(datapath + "EstimatedState.csv", delimiter=', ', header=0, engine='python')
rawSal = pd.read_csv(datapath + "Salinity.csv", delimiter=', ', header=0, engine='python')
rawDepth = pd.read_csv(datapath + "Depth.csv", delimiter=', ', header=0, engine='python')
rawGPS = pd.read_csv(datapath + "GpsFix.csv", delimiter=', ', header=0, engine='python')
rawCurrent = pd.read_csv(datapath + "EstimatedStreamVelocity.csv", delimiter=', ', header=0, engine='python')

# To group all the time stamp together, since only second accuracy matters
rawSal.iloc[:, 0] = np.ceil(rawSal.iloc[:, 0])
rawTemp.iloc[:, 0] = np.ceil(rawTemp.iloc[:, 0])
rawCTDTemp = rawTemp[rawTemp.iloc[:, 2] == 'SmartX']
rawLoc.iloc[:, 0] = np.ceil(rawLoc.iloc[:, 0])
rawDepth.iloc[:, 0] = np.ceil(rawDepth.iloc[:, 0])
rawDepth.iloc[:, 0] = np.ceil(rawDepth.iloc[:, 0])

depth_ctd = rawDepth[rawDepth.iloc[:, 2] == 'SmartX']["value (m)"].groupby(rawDepth["timestamp"]).mean()
depth_dvl = rawDepth[rawDepth.iloc[:, 2] == 'DVL']["value (m)"].groupby(rawDepth["timestamp"]).mean()
depth_est = rawLoc["depth (m)"].groupby(rawLoc["timestamp"]).mean()

# indices used to extract data
lat_origin = rawLoc["lat (rad)"].groupby(rawLoc["timestamp"]).mean()
lon_origin = rawLoc["lon (rad)"].groupby(rawLoc["timestamp"]).mean()
x_loc = rawLoc["x (m)"].groupby(rawLoc["timestamp"]).mean()
y_loc = rawLoc["y (m)"].groupby(rawLoc["timestamp"]).mean()
z_loc = rawLoc["z (m)"].groupby(rawLoc["timestamp"]).mean()
depth = rawLoc["depth (m)"].groupby(rawLoc["timestamp"]).mean()
time_loc = rawLoc["timestamp"].groupby(rawLoc["timestamp"]).mean()
time_sal= rawSal["timestamp"].groupby(rawSal["timestamp"]).mean()
time_temp = rawCTDTemp["timestamp"].groupby(rawCTDTemp["timestamp"]).mean()
dataSal = rawSal["value (psu)"].groupby(rawSal["timestamp"]).mean()
dataTemp = rawCTDTemp.iloc[:, -1].groupby(rawCTDTemp["timestamp"]).mean()

#% Rearrange data according to their timestamp
data = []
time_mission = []
x = []
y = []
z = []
d = []
sal = []
temp = []
lat = []
lon = []

for i in range(len(time_loc)):
    if np.any(time_sal.isin([time_loc.iloc[i]])) and np.any(time_temp.isin([time_loc.iloc[i]])):
        time_mission.append(time_loc.iloc[i])
        x.append(x_loc.iloc[i])
        y.append(y_loc.iloc[i])
        z.append(z_loc.iloc[i])
        d.append(depth.iloc[i])
        lat.append(lat_origin.iloc[i])
        lon.append(lon_origin.iloc[i])
        sal.append(dataSal[time_sal.isin([time_loc.iloc[i]])].iloc[0])
        temp.append(dataTemp[time_temp.isin([time_loc.iloc[i]])].iloc[0])
    else:
        print(datetime.fromtimestamp(time_loc.iloc[i]))
        continue

lat = np.array(lat).reshape(-1, 1)
lon = np.array(lon).reshape(-1, 1)
x = np.array(x).reshape(-1, 1)
y = np.array(y).reshape(-1, 1)
z = np.array(z).reshape(-1, 1)
d = np.array(d).reshape(-1, 1)
sal = np.array(sal).reshape(-1, 1)
temp = np.array(temp).reshape(-1, 1)
time_mission = np.array(time_mission).reshape(-1, 1)

datasheet = np.hstack((time_mission, lat, lon, x, y, z, d, sal, temp))
#%%
np.savetxt(figpath + "../data.txt", datasheet, delimiter = ",")




