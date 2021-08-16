from usr_func import *
datapath = '/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Nidelva/June17/17_06_2021/Data/'
# datapath = '/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Nidelva/June17/00_00_S17_06_2021/Data/'

print("hello world")
data = np.loadtxt(datapath + "data.txt", delimiter=",")

timestamp = data[:, 0]
lat = data[:, 1]
lon = data[:, 2]
x = data[:, 3]
y = data[:, 4]
z = data[:, 5]
depth = data[:, 6]
sal = data[:, 7]
temp = data[:, 8]

def xyz2latlon(x, y, lat_origin, lon_origin):
    circumference = 40075000
    lat = lat_origin + rad2deg(x * np.pi * 2.0 / circumference)
    lon = lon_origin + rad2deg(y * np.pi * 2.0 / (circumference * np.cos(deg2rad(lat))))
    return lat, lon

lat, lon = xyz2latlon(x, y, lat, lon)

plt.figure(figsize = (8, 5))
plt.scatter(lon, lat, c = temp)
plt.title("Temperature versus location")
plt.xlabel("lon [deg]")
plt.ylabel("lat [deg]")
plt.colorbar()
plt.savefig(datapath + "temp.pdf")
plt.show()



