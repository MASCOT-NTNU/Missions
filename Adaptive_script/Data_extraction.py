# datapath = '/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Nidelva/May27/Data/'
import matplotlib.pyplot as plt

datapath = '/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Nidelva/July06/Adaptive/Data/'
figpath = '/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Nidelva/July06/Adaptive/fig/'
import numpy as np
import pandas as pd
from datetime import datetime
import os
from usr_func import *


#% Data extraction from the raw data
rawTemp = pd.read_csv(datapath + "Temperature.csv", delimiter=', ', header=0, engine='python')
rawLoc = pd.read_csv(datapath + "EstimatedState.csv", delimiter=', ', header=0, engine='python')
rawSal = pd.read_csv(datapath + "Salinity.csv", delimiter=', ', header=0, engine='python')
rawDepth = pd.read_csv(datapath + "Depth.csv", delimiter=', ', header=0, engine='python')
# rawGPS = pd.read_csv(datapath + "GpsFix.csv", delimiter=', ', header=0, engine='python')
# rawCurrent = pd.read_csv(datapath + "EstimatedStreamVelocity.csv", delimiter=', ', header=0, engine='python')

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
        lat_temp = rad2deg(lat_origin.iloc[i]) + rad2deg(x_loc.iloc[i] * np.pi * 2.0 / circumference)
        lat.append(lat_temp)
        lon.append(rad2deg(lon_origin.iloc[i]) + rad2deg(y_loc.iloc[i] * np.pi * 2.0 / (circumference * np.cos(deg2rad(lat_temp)))))
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
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from gmplot import GoogleMapPlotter
from random import random

class CustomGoogleMapPlotter(GoogleMapPlotter):
    def __init__(self, center_lat, center_lng, zoom, apikey='AIzaSyAZ_VZXoJULTFQ9KSPg1ClzHEFjyPbJUro',
                 map_type='satellite'):
        if apikey == '':
            try:
                with open('apikey.txt', 'r') as apifile:
                    apikey = apifile.readline()
            except FileNotFoundError:
                pass
        print(apikey)
        GoogleMapPlotter(center_lat, center_lng, zoom, apikey=apikey)
        super().__init__(center_lat, center_lng, zoom, apikey = apikey)

        self.map_type = map_type
        assert(self.map_type in ['roadmap', 'satellite', 'hybrid', 'terrain'])

    def write_map(self,  f):
        f.write('\t\tvar centerlatlng = new google.maps.LatLng(%f, %f);\n' %
                (self.center[0], self.center[1]))
        f.write('\t\tvar myOptions = {\n')
        f.write('\t\t\tzoom: %d,\n' % (self.zoom))
        f.write('\t\t\tcenter: centerlatlng,\n')

        # Change this line to allow different map types
        f.write('\t\t\tmapTypeId: \'{}\'\n'.format(self.map_type))

        f.write('\t\t};\n')
        f.write(
            '\t\tvar map = new google.maps.Map(document.getElementById("map_canvas"), myOptions);\n')
        f.write('\n')

    def color_scatter(self, lats, lngs, values=None, colormap='coolwarm',
                      size=None, marker=False, s=None, **kwargs):
        def rgb2hex(rgb):
            """ Convert RGBA or RGB to #RRGGBB """
            rgb = list(rgb[0:3]) # remove alpha if present
            rgb = [int(c * 255) for c in rgb]
            hexcolor = '#%02x%02x%02x' % tuple(rgb)
            return hexcolor

        if values is None:
            colors = [None for _ in lats]
        else:
            cmap = plt.get_cmap(colormap)
            norm = Normalize(vmin=min(values), vmax=max(values))
            scalar_map = ScalarMappable(norm=norm, cmap=cmap)
            colors = [rgb2hex(scalar_map.to_rgba(value)) for value in values]
        for lat, lon, c in zip(lats, lngs, colors):
            self.scatter(lats=[lat], lngs=[lon], c=c, size=size, marker=marker,
                         s=s, **kwargs)

from usr_func import *

initial_zoom = 12
# apikey = 'AIzaSyDkWNSq_EKnrV9qP6thJe5Y8a5kVLKEjUI'
apikey = 'AIzaSyAZ_VZXoJULTFQ9KSPg1ClzHEFjyPbJUro'
gmap = CustomGoogleMapPlotter(lat[0, 0], lon[0, 0], initial_zoom, map_type='satellite', apikey = apikey)
gmap.color_scatter(lat[:, 0].tolist(), lon[:, 0].tolist(), sal.squeeze(), size = 10, colormap='hsv')

gmap.draw(figpath + "sal.html")
plt.figure()
plt.scatter(lon, lat, c = sal, cmap = 'hsv')
plt.colorbar()
plt.savefig(figpath + "sal.pdf")
plt.show()
#%%
# np.savetxt(figpath + "../data.txt", datasheet, delimiter = ",")




