# ! /usr/bin/env python3
__author__ = "Yaolin Ge"
__copyright__ = "Copyright 2021, The MASCOT Project, NTNU (https://wiki.math.ntnu.no/mascot)"
__credits__ = ["Yaolin Ge"]
__license__ = "MIT"
__version__ = "1.0.1"
__maintainer__ = "Yaolin Ge"
__email__ = "yaolin.ge@ntnu.no"
__status__ = "UnderDevelopment"



import matplotlib.pyplot as plt
import numpy as np
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'font.style': 'oblique'})
circumference = 40075000 # [m], circumference
from scipy.stats import norm

def deg2rad(deg):
    return deg / 180 * np.pi

def rad2deg(rad):
    return rad / np.pi * 180

def latlon2xy(lat, lon, lat_origin, lon_origin):
    x = deg2rad((lat - lat_origin)) / 2 / np.pi * circumference
    y = deg2rad((lon - lon_origin)) / 2 / np.pi * circumference * np.cos(deg2rad(lat))
    return x, y

def xy2latlon(x, y, lat_origin, lon_origin):
    lat = lat_origin + rad2deg(x * np.pi * 2.0 / circumference)
    lon = lon_origin + rad2deg(y * np.pi * 2.0 / (circumference * np.cos(deg2rad(lat))))
    return lat, lon

def compute_H(grid1, grid2, ksi):
    X1 = grid1[:, 0].reshape(-1, 1)
    Y1 = grid1[:, 1].reshape(-1, 1)
    Z1 = grid1[:, -1].reshape(-1, 1)
    X2 = grid2[:, 0].reshape(-1, 1)
    Y2 = grid2[:, 1].reshape(-1, 1)
    Z2 = grid2[:, -1].reshape(-1, 1)

    distX = X1 @ np.ones([1, X2.shape[0]]) - np.ones([X1.shape[0], 1]) @ X2.T
    distY = Y1 @ np.ones([1, Y2.shape[0]]) - np.ones([Y1.shape[0], 1]) @ Y2.T
    distXY = distX ** 2 + distY ** 2
    distZ = Z1 @ np.ones([1, Z2.shape[0]]) - np.ones([Z1.shape[0], 1]) @ Z2.T
    dist = np.sqrt(distXY + (ksi * distZ) ** 2)
    return dist

## Functions used
def Matern_cov(sigma, eta, H):
    '''
    :param sigma: scaling coef
    :param eta: range coef
    :param H: distance matrix
    :return: matern covariance
    '''
    return sigma ** 2 * (1 + eta * H) * np.exp(-eta * H)


def EP_1D(mu, Sigma, Threshold):
    '''
    This function computes the excursion probability
    :param mu:
    :param Sigma:
    :param Threshold:
    :return:
    '''
    EP = np.zeros_like(mu)
    for i in range(EP.shape[0]):
        EP[i] = norm.cdf(Threshold, mu[i], Sigma[i, i])
    return EP



