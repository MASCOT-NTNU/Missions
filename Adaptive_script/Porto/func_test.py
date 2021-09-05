from Adaptive_script.Porto.Grid import GridPoly

# a = GridPoly()
from Adaptive_script.Porto.usr_func import *

import numpy as np
angle = deg2rad(np.arange(0, 360, 60))

distance = 60
y = distance * np.sin(angle)
x = distance * np.cos(angle)
lat_origin, lon_origin = 41.10251, -8.669811 # the right bottom corner coordinates

def xy2latlon(x, y, lat_origin, lon_origin):
    lat = lat_origin + rad2deg(x * np.pi * 2.0 / circumference)
    lon = lon_origin + rad2deg(y * np.pi * 2.0 / (circumference * np.cos(deg2rad(lat))))
    return lat, lon

lat, lon = xy2latlon(x, y, lat_origin, lon_origin)
latp = lat[0]
latn = lat_origin
latf = lat[5]
lonp = lon[0]
lonn = lon_origin
lonf = lon[5]
# xp = x[0]
# xn = 0
# xf = x[1]
# yp = y[0]
# yn = 0
# yf = y[1]
vec1 = np.array([latn - latp, lonn - lonp])
vec2 = np.array([latf - latn, lonf - lonn])
print(vec1, vec2)
print(vec1 @ vec2)



plt.plot(lon, lat, 'k.')
plt.plot([lonp, lonn], [latp, latn], 'g-')
plt.plot([lonn, lonf], [latn, latf], 'b-')
plt.plot(lon_origin, lat_origin, 'r.')
plt.show()

#%%
t = np.random.rand(30, 1) - .5
t2 = np.random.rand(30, 1) - .5
t3 = np.random.rand(30, 1) - .5
tt = np.sqrt(t** 2+ t2**2 + t3 ** 2)
np.where(tt < .5)[0]







