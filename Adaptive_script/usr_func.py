import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from scipy.stats import mvn
import scipy.spatial.distance as scdist
import netCDF4
import os
import gmplot
from skgstat import Variogram, DirectionalVariogram
from sklearn.linear_model import LinearRegression
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'font.style': 'oblique'})
circumference = 40075000 # [m]
err_bound = 0.1 # [m]


def rotateXY(nx, ny, distance, alpha):
    '''
    :param nx:
    :param ny:
    :param distance:
    :param alpha:
    :return:
    '''
    R = np.array([[np.cos(deg2rad(alpha)), np.sin(deg2rad(alpha))],
                  [-np.sin(deg2rad(alpha)), np.cos(deg2rad(alpha))]])
    x = np.arange(nx) * distance
    y = np.arange(ny) * distance
    gridx, gridy = np.meshgrid(x, y)
    gridxnew = np.zeros_like(gridx)
    gridynew = np.zeros_like(gridy)
    for i in range(nx):
        for j in range(ny):
            gridxnew[i, j], gridynew[i, j] = R @ np.vstack((gridx[i, j], gridy[i, j]))
    return gridxnew, gridynew


def getTrend(data, depth):
    xauv = data[:, 3].reshape(-1, 1)
    yauv = data[:, 4].reshape(-1, 1)
    depth_auv = data[:, 6].reshape(-1, 1)
    sal_auv = data[:, 7].reshape(-1, 1)
    temp_auv = data[:, 8].reshape(-1, 1)

    depthl = np.array(depth) - err_bound
    depthu = np.array(depth) + err_bound
    beta0_sal = []
    beta1_sal = []
    beta0_temp = []
    beta1_temp = []

    for i in range(len(depth)):
        ind_obs = (depthl[i] <= depth_auv) & (depth_auv <= depthu[i])
        xobs = xauv[ind_obs].reshape(-1, 1)
        yobs = yauv[ind_obs].reshape(-1, 1)
        sal_obs = sal_auv[ind_obs].reshape(-1, 1)
        temp_obs = temp_auv[ind_obs].reshape(-1, 1)
        X = np.hstack((xobs, yobs))

        model_sal = LinearRegression()
        model_sal.fit(X, sal_obs)
        beta0_sal.append(model_sal.intercept_)
        beta1_sal.append(model_sal.coef_)

        model_temp = LinearRegression()
        model_temp.fit(X, temp_obs)
        beta0_temp.append(model_temp.intercept_)
        beta1_temp.append(model_temp.coef_)

    return np.array(beta0_sal), np.array(beta1_sal).squeeze(), np.array(beta0_temp), np.array(beta1_temp).squeeze()


def getCoefficients(data, SINMOD, depth):
    '''
    :param data:
    :param SINMOD:
    :param depth:
    :return:
    '''
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

    depthl = np.array(depth) - err_bound
    depthu = np.array(depth) + err_bound

    beta0 = np.zeros([len(depth), 2])
    beta1 = np.zeros([len(depth), 2])
    sal_residual = []
    temp_residual = []
    x_loc = []
    y_loc = []
    for i in range(len(depth)):
        ind_obs = (depthl[i] <= depth_auv) & (depth_auv <= depthu[i])
        lat_obs = lat_auv[ind_obs].reshape(-1, 1)
        lon_obs = lon_auv[ind_obs].reshape(-1, 1)
        sal_obs = sal_auv[ind_obs].reshape(-1, 1)
        temp_obs = temp_auv[ind_obs].reshape(-1, 1)
        sal_SINMOD, temp_SINMOD = GetSINMODFromCoordinates(SINMOD, np.hstack((lat_obs, lon_obs)), depth[i])
        model_sal = LinearRegression()
        model_sal.fit(sal_SINMOD, sal_obs)
        beta0[i, 0] = model_sal.intercept_
        beta1[i, 0] = model_sal.coef_
        model_temp = LinearRegression()
        model_temp.fit(temp_SINMOD, temp_obs)
        beta0[i, 1] = model_temp.intercept_
        beta1[i, 1] = model_temp.coef_
        sal_residual.append(sal_obs - sal_SINMOD)
        temp_residual.append(temp_obs - temp_SINMOD)
        x_loc.append(xauv[ind_obs])
        y_loc.append(yauv[ind_obs])

    return beta0, beta1, sal_residual, temp_residual, x_loc, y_loc



def deg2rad(deg):
    '''
    :param deg:
    :return:
    '''
    return deg / 180 * np.pi


def rad2deg(rad):
    '''
    :param rad:
    :return:
    '''
    return rad / np.pi * 180


def BBox(lat, lon, distance, alpha):
    '''
    :param lat:
    :param lon:
    :param distance:
    :param alpha:
    :return:
    '''
    lat4 = deg2rad(lat)
    lon4 = deg2rad(lon)

    lat2 = lat4 + distance * np.sin(deg2rad(alpha)) / circumference * 2 * np.pi
    lat1 = lat2 + distance * np.cos(deg2rad(alpha)) / circumference * 2 * np.pi
    lat3 = lat4 + distance * np.sin(np.pi / 2 - deg2rad(alpha)) / circumference * 2 * np.pi

    lon2 = lon4 + distance * np.cos(deg2rad(alpha)) / (circumference * np.cos(lat2)) * 2 * np.pi
    lon3 = lon4 - distance * np.cos(np.pi / 2 - deg2rad(alpha)) / (circumference * np.cos(lat3)) * 2 * np.pi
    lon1 = lon3 + distance * np.cos(deg2rad(alpha)) / (circumference * np.cos(lat1)) * 2 * np.pi

    box = np.vstack((np.array([lat1, lat2, lat3, lat4]), np.array([lon1, lon2, lon3, lon4]))).T

    return rad2deg(box)


def getCoordinates(box, nx, ny, distance, alpha):
    '''
    :param box:
    :param nx:
    :param ny:
    :param distance:
    :param alpha:
    :return:
    '''
    R = np.array([[np.cos(deg2rad(alpha)), np.sin(deg2rad(alpha))],
                  [-np.sin(deg2rad(alpha)), np.cos(deg2rad(alpha))]])

    lat_origin, lon_origin = box[-1, :]
    x = np.arange(nx) * distance
    y = np.arange(ny) * distance
    gridx, gridy = np.meshgrid(x, y)

    lat = np.zeros([nx, ny])
    lon = np.zeros([nx, ny])
    for i in range(nx):
        for j in range(ny):
            xnew, ynew = R @ np.vstack((gridx[i, j], gridy[i, j]))
            lat[i, j] = lat_origin + rad2deg(xnew * np.pi * 2.0 / circumference)
            lon[i, j] = lon_origin + rad2deg(ynew * np.pi * 2.0 / (circumference * np.cos(deg2rad(lat[i, j]))))
    coordinates = np.hstack((lat.reshape(-1, 1), lon.reshape(-1, 1)))
    return coordinates


def GetSINMODFromCoordinates(SINMOD, coordinates, depth):
    '''
    :param SINMOD:
    :param coordinates:
    :param depth:
    :return:
    '''
    salinity = np.mean(SINMOD['salinity'][:, :, :, :], axis=0)
    temperature = np.mean(SINMOD['temperature'][:, :, :, :], axis=0) - 273.15
    depth_sinmod = np.array(SINMOD['zc'])
    lat_sinmod = np.array(SINMOD['gridLats'][:, :]).reshape(-1, 1)
    lon_sinmod = np.array(SINMOD['gridLons'][:, :]).reshape(-1, 1)
    sal_sinmod = np.zeros([coordinates.shape[0], 1])
    temp_sinmod = np.zeros([coordinates.shape[0], 1])

    for i in range(coordinates.shape[0]):
        lat, lon = coordinates[i]
        ind_depth = np.where(np.array(depth_sinmod) == depth)[0][0]
        idx = np.argmin((lat_sinmod - lat) ** 2 + (lon_sinmod - lon) ** 2)
        sal_sinmod[i] = salinity[ind_depth].reshape(-1, 1)[idx]
        temp_sinmod[i] = temperature[ind_depth].reshape(-1, 1)[idx]
    return sal_sinmod, temp_sinmod


def Matern_cov(sigma, eta, t):
    '''
    :param sigma: scaling coef
    :param eta: range coef
    :param t: distance matrix
    :return: matern covariance
    '''
    return sigma ** 2 * (1 + eta * t) * np.exp(-eta * t)


def mu(H, beta):
    '''
    :param H: design matrix
    :param beta: regression coef
    :return: mean
    '''
    return np.dot(H, beta)


def EIBV(threshold, mu, Sig, F, R):
    '''
    :param threshold:
    :param mu:
    :param Sig:
    :param F: sampling matrix
    :param R: noise matrix
    :return: EIBV evaluated at every point
    '''
    Sigxi = Sig @ F.T @ np.linalg.solve( F @ Sig @ F.T + R, F @ Sig)
    V = Sig - Sigxi
    sa2 = np.diag(V).reshape(-1, 1) # the corresponding variance term for each location
    IntA = 0.0
    for i in range(len(mu)):
        sn2 = sa2[i]
        m = mu[i]
        IntA = IntA + mvn.mvnun(-np.inf, threshold, m, sn2)[0] - mvn.mvnun(-np.inf, threshold, m, sn2)[0] ** 2
    return IntA


def GPupd(mu, Sig, R, F, y_sampled):
    C = F @ Sig @ F.T + R
    mu_p = mu + Sig @ F.T @ np.linalg.solve(C, (y_sampled - F @ mu))
    Sigma_p = Sig - Sig @ F.T @ np.linalg.solve(C, F @ Sig)
    return mu_p, Sigma_p


def find_candidates_loc(north_ind, east_ind, depth_ind, N1, N2, N3):
    '''
    This will find the neighbouring loc
    But also limit it inside the grid
    :param idx:
    :param idy:
    :return:
    '''

    north_ind_l = [north_ind - 1 if north_ind > 0 else north_ind]
    north_ind_u = [north_ind + 1 if north_ind < N1 - 1 else north_ind]
    east_ind_l = [east_ind - 1 if east_ind > 0 else east_ind]
    east_ind_u = [east_ind + 1 if east_ind < N2 - 1 else east_ind]
    depth_ind_l = [depth_ind - 1 if depth_ind > 0 else depth_ind]
    depth_ind_u = [depth_ind + 1 if depth_ind < N3 - 1 else depth_ind]

    north_ind_v = np.unique(np.vstack((north_ind_l, north_ind, north_ind_u)))
    east_ind_v = np.unique(np.vstack((east_ind_l, east_ind, east_ind_u)))
    depth_ind_v = np.unique(np.vstack((depth_ind_l, depth_ind, depth_ind_u)))

    north_ind, east_ind, depth_ind = np.meshgrid(north_ind_v, east_ind_v, depth_ind_v)

    return north_ind.reshape(-1, 1), east_ind.reshape(-1, 1), depth_ind.reshape(-1, 1)


def find_next_EIBV(north_cand, east_cand, depth_cand, north_now, east_now, depth_now,
                   north_pre, east_pre, depth_pre, N1, N2, N3, Sig, mu, tau, Thres):

    id = []
    dnorth1 = north_now - north_pre
    deast1 = east_now - east_pre
    ddepth1 = depth_now - depth_pre
    vec1 = np.array([dnorth1, deast1, ddepth1])
    for i in north_cand:
        for j in east_cand:
            for z in depth_cand:
                if i == north_now and j == east_now and z == depth_now:
                    continue
                dnorth2 = i - north_now
                deast2 = j - east_now
                ddepth2 = z - depth_now
                vec2 = np.array([dnorth2, deast2, ddepth2])
                if np.dot(vec1, vec2) >= 0:
                    id.append(np.ravel_multi_index((i, j, z), (N1, N2, N3)))
                else:
                    continue
    id = np.unique(np.array(id))

    M = len(id)
    noise = tau ** 2
    R = np.diagflat(noise)  # diag not anymore support constructing matrix from vector
    N = N1 * N2 * N3
    eibv = []
    for k in range(M):
        F = np.zeros([1, N])
        F[0, id[k]] = True
        eibv.append(EIBV(Thres, mu, Sig, F, R))
    ind_desired = np.argmin(np.array(eibv))
    north_next, east_next, depth_next = np.unravel_index(id[ind_desired], (N1, N2, N3))

    return north_next, east_next, depth_next

