from usr_func import *

#%% Section I: Compute variogram and trend coefficients
data = np.loadtxt("data.txt", delimiter=",")
timestamp = data[:, 0]
lat = data[:, 1]
lon = data[:, 2]
xauv = data[:, 3]
yauv = data[:, 4]
zauv = data[:, 5]
depth = data[:, 6]
sal = data[:, 7]
temp = data[:, 8]

lat_auv = lat + rad2deg(xauv * np.pi * 2.0 / circumference)
lon_auv = lon + rad2deg(yauv * np.pi * 2.0 / (circumference * np.cos(deg2rad(lat_auv))))

depth_obs = [0.5, 1.0, 1.5, 2.0, 2.5]

fp='samples_2020.05.01.nc'
nc = netCDF4.Dataset(fp)
beta0, beta1, sal_residual, temp_residual, x_loc, y_loc = getCoefficients(data, nc, [0.5, 1.0, 1.5, 2.0, 2.5])
figpath = '/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Nidelva/Variogram/'
# This tries to find the suitable methods and estimators
for i in range(len(depth_obs)):
# for i in range(1):
    V_v = Variogram(coordinates = np.hstack((y_loc[i].reshape(-1, 1), x_loc[i].reshape(-1, 1))),
                    values = sal_residual[i].squeeze(), use_nugget=True, model = "Matern", normalize = False,
                    n_lags = 100) # model = "Matern" check
    # V_v.estimator = 'cressie'
    V_v.fit_method = 'trf' # moment method

    fig = V_v.plot(hist = False)
    # fig.suptitle("test")
    # fig = V_v.plot(hist = True)
    # fig.savefig(figpath + "sal_{:03d}.pdf".format(i))
    print(V_v)

#%%
depth_obs = [0.5, 2.0, 5.0]
timestamp = data[:, 0].reshape(-1, 1)
lat_auv = rad2deg(data[:, 1].reshape(-1, 1))
lon_auv = rad2deg(data[:, 2].reshape(-1, 1))
xauv = data[:, 3].reshape(-1, 1)
yauv = data[:, 4].reshape(-1, 1)
zauv = data[:, 5].reshape(-1, 1)
depth_auv = data[:, 6].reshape(-1, 1)
sal_auv = data[:, 7].reshape(-1, 1)
temp_auv = data[:, 8].reshape(-1, 1)
lat_auv = lat_auv + rad2deg(xauv * np.pi * 2.0 / circumference)
lon_auv = lon_auv + rad2deg(yauv * np.pi * 2.0 / (circumference * np.cos(deg2rad(lat_auv))))

depthl = np.array(depth_obs) - err_bound
depthu = np.array(depth_obs) + err_bound
x_obs = []
y_obs = []
sal_obs = []
temp_obs = []

for i in range(len(depth_obs)):
    ind_obs = (depthl[i] <= depth_auv) & (depth_auv <= depthu[i])
    x_obs.append(np.floor(xauv[ind_obs].reshape(-1, 1)))
    y_obs.append(np.floor(yauv[ind_obs].reshape(-1, 1)))
    sal_obs.append(sal_auv[ind_obs].reshape(-1, 1))
    temp_obs.append(temp_auv[ind_obs].reshape(-1, 1))

plt.figure()
plt.plot(y_obs[0])
plt.plot(y_obs[1])
plt.plot(y_obs[-1])
plt.show()
a = np.intersect1d(x_obs[0], x_obs[1])
b = np.intersect1d(a, x_obs[-1])

#%% Plot the depth variogram
z = []
sal_depth = []
temp_depth = []
for i in range(len(depth_obs)):
    for j in range(len(sal_residual[i])):
        sal_depth.append(sal_residual[i][j, 0])
        temp_depth.append(temp_residual[i][j, 0])
        z.append(depth_obs[i])

V_v = Variogram(coordinates = z, values = np.array(sal_depth), use_nugget=True, model = "Matern", normalize = False,
                n_lags = 10) # model = "Matern" check
# V_v.estimator = 'cressie'
V_v.fit_method = 'trf' # moment method
fig = V_v.plot(hist = False)
# fig = V_v.plot(hist = True)
fig.savefig(figpath + "sal_depth.pdf")
print(V_v)

#%% To find the depth \ksi

plt.figure()
plt.plot(xauv, 'k.')
# plt.xlabel("Depth [m]")
# plt.ylabel("Temperature residual")
# plt.title("Temperature residual vs depth")
# plt.savefig(figpath + "TempVSDepth.pdf")
plt.show()
from scipy.stats import pearsonr
# print(pearsonr(temp_depth, z)[1])
print(np.corrcoef(temp_depth, z))


#%% Cross plot between temperature and salinity residuals
plt.figure()
plt.plot(sal_residual[0], temp_residual[0], 'k.')
plt.xlabel("Salinity")
plt.title("Cross plot of the residuals")
plt.ylabel("Temperature")
plt.savefig(figpath + "Cross.pdf")
plt.show()
from scipy.stats import pearsonr
print(pearsonr(sal_residual[0].squeeze(), temp_residual[0].squeeze())[0])


#%% Create the map plotter:
# apikey = 'AIzaSyAZ_VZXoJULTFQ9KSPg1ClzHEFjyPbJUro' # (your API key here)
apikey = 'AIzaSyDkWNSq_EKnrV9qP6thJe5Y8a5kVLKEjUI'
gmap = gmplot.GoogleMapPlotter(box[-1, 0], box[-1, 1], 14, apikey=apikey)

# Highlight some attractions:
attractions_lats = coordinates[:, 0]
attractions_lngs = coordinates[:, 1]
# gmap.scatter(attractions_lats, attractions_lngs, color='#3B0B39', size=4, marker=False)
gmap.scatter(attractions_lats, attractions_lngs, color='#FF0000', size=4, marker=False)

# Mark a hidden gem:
for i in range(box.shape[0]):
    gmap.marker(box[i, 0], box[i, 1], color='cornflowerblue')

# Draw the map:
gmap.draw(os.getcwd() + '/map.html')
