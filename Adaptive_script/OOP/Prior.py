
import numpy as np
from GP import GaussianProcess


class Prior(GaussianProcess):
    AUVdata = None
    mu_prior_sal = None
    mu_prior_temp = None
    Sigma_prior_sal = None
    Sigma_prior_temp = None
    beta0 = None
    beta1 = None
    lat_grid = np.zeros([GaussianProcess.N1, GaussianProcess.N2])
    lon_grid = np.zeros([GaussianProcess.N1, GaussianProcess.N2])
    coord_grid = np.zeros([GaussianProcess.N1 * GaussianProcess.N2, 2])
    depth_tolerance = 0.25 # tolerance +/- in depth, 0.5 m == [0.25 ~ 0.75]m

    SINMOD_datapath = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Adaptive_script/samples_2020.05.01.nc"
    SINMOD_Data = None

    def __init__(self):
        GaussianProcess.__init__(self)
        self.generateCoordinates()
        self.getAUVData()
        self.getSINMODData()
        self.getCoefficients()
        self.get_mu_prior()
        self.saveCoef()
        self.print_Prior()

    def print_Prior(self):
        print("lat_grid: ", self.lat_grid.shape)
        print("lon_grid: ", self.lon_grid.shape)
        print("coord_grid: ", self.coord_grid.shape)
        print("beta0: \n", self.beta0)
        print("beta1: \n", self.beta1)
        print("mu_sal_prior: ", self.mu_prior_sal.shape)
        print("mu_temp_prior: ", self.mu_prior_temp.shape)
        print("Prior is setup successfully!\n\n")

    def set_depth_tolerance(self, value):
        Prior.depth_tolerance = value

    def generateCoordinates(self):
        for i in range(GaussianProcess.N1):
            for j in range(GaussianProcess.N2):
                xnew, ynew = GaussianProcess.R @ np.vstack((GaussianProcess.x[i], GaussianProcess.y[j]))

                Prior.lat_grid[i, j] = GaussianProcess.lat_origin + \
                            GaussianProcess.rad2deg(self, xnew * np.pi * 2.0 /
                                                    GaussianProcess.circumference)
                Prior.lon_grid[i, j] = GaussianProcess.lon_origin + \
                            GaussianProcess.rad2deg(self, ynew * np.pi * 2.0 /
                                                    (GaussianProcess.circumference *
                                                     np.cos(GaussianProcess.deg2rad(self, Prior.lat_grid[i, j]))))
        Prior.coord_grid = np.hstack((Prior.lat_grid.reshape(-1, 1), Prior.lon_grid.reshape(-1, 1)))

    def getAUVData(self):
        Prior.AUVdata = np.loadtxt('data.txt', delimiter = ",")

    def getSINMODData(self):
        import netCDF4
        Prior.SINMOD_Data = netCDF4.Dataset(Prior.SINMOD_datapath)

    def getSINMODFromCoordsDepth(self, coordinates, depth):
        salinity = np.mean(Prior.SINMOD_Data['salinity'][:, :, :, :], axis=0)
        temperature = np.mean(Prior.SINMOD_Data['temperature'][:, :, :, :], axis=0) - 273.15
        depth_sinmod = np.array(Prior.SINMOD_Data['zc'])
        lat_sinmod = np.array(Prior.SINMOD_Data['gridLats'][:, :]).reshape(-1, 1)
        lon_sinmod = np.array(Prior.SINMOD_Data['gridLons'][:, :]).reshape(-1, 1)
        sal_sinmod = np.zeros([coordinates.shape[0], 1])
        temp_sinmod = np.zeros([coordinates.shape[0], 1])

        for i in range(coordinates.shape[0]):
            lat, lon = coordinates[i]
            ind_depth = np.where(np.array(depth_sinmod) == depth)[0][0]
            idx = np.argmin((lat_sinmod - lat) ** 2 + (lon_sinmod - lon) ** 2)
            sal_sinmod[i] = salinity[ind_depth].reshape(-1, 1)[idx]
            temp_sinmod[i] = temperature[ind_depth].reshape(-1, 1)[idx]
        return sal_sinmod, temp_sinmod

    def getCoefficients(self):
        # timestamp = Prior.AUVdata[:, 0].reshape(-1, 1)
        lat_auv_origin = self.rad2deg(Prior.AUVdata[:, 1]).reshape(-1, 1)
        lon_auv_origin = self.rad2deg(Prior.AUVdata[:, 2]).reshape(-1, 1)
        xauv = Prior.AUVdata[:, 3].reshape(-1, 1)
        yauv = Prior.AUVdata[:, 4].reshape(-1, 1)
        # zauv = Prior.AUVdata[:, 5].reshape(-1, 1)
        depth_auv = Prior.AUVdata[:, 6].reshape(-1, 1)
        sal_auv = Prior.AUVdata[:, 7].reshape(-1, 1)
        temp_auv = Prior.AUVdata[:, 8].reshape(-1, 1)
        lat_auv = lat_auv_origin + self.rad2deg(xauv * np.pi * 2.0 / self.circumference)
        lon_auv = lon_auv_origin + self.rad2deg(yauv * np.pi * 2.0 / (self.circumference * np.cos(self.deg2rad(lat_auv))))

        depthl = np.array(self.depth_obs) - self.depth_tolerance
        depthu = np.array(self.depth_obs) + self.depth_tolerance

        Prior.beta0 = np.zeros([len(self.depth_obs), 2])
        Prior.beta1 = np.zeros([len(self.depth_obs), 2])
        sal_residual = []
        temp_residual = []

        for i in range(len(self.depth_obs)):
            # sort out AUV data
            ind_obs = (depthl[i] <= depth_auv) & (depth_auv <= depthu[i])
            lat_obs = lat_auv[ind_obs].reshape(-1, 1)
            lon_obs = lon_auv[ind_obs].reshape(-1, 1)
            sal_obs = sal_auv[ind_obs].reshape(-1, 1)
            temp_obs = temp_auv[ind_obs].reshape(-1, 1)
            coord_obs = np.hstack((lat_obs, lon_obs))

            # sort out SINMOD data
            sal_sinmod, temp_sinmod = self.getSINMODFromCoordsDepth(coord_obs, self.depth_obs[i])

            # compute the coef for salinity
            sal_modelX = np.hstack((np.ones_like(sal_sinmod), sal_sinmod))
            sal_modelY = sal_obs
            Beta_sal = np.linalg.solve((sal_modelX.T @ sal_modelX), (sal_modelX.T @ sal_modelY))
            # compute the coef for temperature
            temp_modelX = np.hstack((np.ones_like(temp_sinmod), temp_sinmod))
            temp_modelY = temp_obs
            Beta_temp = np.linalg.solve((temp_modelX.T @ temp_modelX), (temp_modelX.T @ temp_modelY))

            Prior.beta0[i, :] = np.hstack((Beta_sal[0], Beta_temp[0]))
            Prior.beta1[i, :] = np.hstack((Beta_sal[1], Beta_temp[1]))

            sal_residual.append(sal_obs - Beta_sal[0] - Beta_sal[1] * sal_sinmod)
            temp_residual.append(temp_obs - Beta_temp[0] - Beta_temp[1] * temp_sinmod)

    def get_mu_prior(self):
        Prior.mu_prior_sal = []
        Prior.mu_prior_temp = []
        for i in range(len(self.depth_obs)):
            sal_sinmod, temp_sinmod = self.getSINMODFromCoordsDepth(Prior.coord_grid, self.depth_obs[i])
            Prior.mu_prior_sal.append(Prior.beta0[i, 0] + Prior.beta1[i, 0] * sal_sinmod)
            Prior.mu_prior_temp.append(Prior.beta0[i, 1] + Prior.beta1[i, 1] * temp_sinmod)
        Prior.mu_prior_sal = np.array(Prior.mu_prior_sal).reshape(-1, 1)
        Prior.mu_prior_temp = np.array(Prior.mu_prior_temp).reshape(-1, 1)

    def saveCoef(self):
        np.savetxt("beta0.txt", Prior.beta0, delimiter=",")
        np.savetxt("beta1.txt", Prior.beta1, delimiter=",")
        np.savetxt("mu_prior_sal.txt", Prior.mu_prior_sal, delimiter=",")
        np.savetxt("mu_prior_temp.txt", Prior.mu_prior_temp, delimiter=",")

    def checkCoords(self):
        from gmplot import GoogleMapPlotter
        initial_zoom = 12
        apikey = 'AIzaSyAZ_VZXoJULTFQ9KSPg1ClzHEFjyPbJUro'
        gmap = GoogleMapPlotter(self.lat_origin, self.lon_origin, initial_zoom, map_type='satellite', apikey=apikey)
        gmap.scatter(self.coord_grid[:, 0], self.coord_grid[:, 1], color='#99ff00', size=20, marker=False)
        gmap.draw("/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/MapPlot/map.html")