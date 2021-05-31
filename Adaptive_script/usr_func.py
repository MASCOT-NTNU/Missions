import numpy as np
import matplotlib.pyplot as plt

from random import sample
from scipy.stats import norm, mvn
import scipy.spatial.distance as scdist
from matplotlib.gridspec import GridSpec
import seaborn as sns
import os
import time

## SETUP FONT ##
# plt.ioff()  # Running plt.ioff() - plots are kept in background, plt.ion() plots are shown as they are generated
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'font.style': 'oblique'})

from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

cmap_proba = ListedColormap(sns.color_palette("RdBu_r", 30))
CMAP_EXCU = ListedColormap(sns.color_palette("Reds", 300))
## CONSTRUCT A BIG CLASS ##




#
# def plot_posterior(mu, Sig, s):
#     fig = plt.figure(figsize=(20, 5))
#     gs = GridSpec(nrows=1, ncols=3)
#     m = [mu[np.arange(0, N, 2)], mu[np.arange(1, N, 2)]]
#
#     i = 0
#     axes = fig.add_subplot(gs[i])
#     im = axes.imshow(m[i].reshape(n1, n2))
#     plt.colorbar(im, fraction=0.045, pad=0.04)
#     plt.gca().invert_yaxis()
#     axes.set(title = "ES", xlabel = 's1', ylabel = 's2')
#
#     i = 1
#     axes = fig.add_subplot(gs[i])
#     im = axes.imshow(m[i].reshape(n1, n2))
#     plt.colorbar(im, fraction=0.045, pad=0.04)
#     plt.gca().invert_yaxis()
#     axes.set(title="ES temp", xlabel='s1', ylabel='s2')
#
#     i = 2
#     axes = fig.add_subplot(gs[i])
#     im = axes.imshow(np.sqrt(np.diag(Sig)[np.arange(0, N, 2)]).reshape(n1, n2))
#     plt.colorbar(im, fraction=0.045, pad=0.04)
#     plt.gca().invert_yaxis()
#     axes.set(title="ES sal", xlabel='s1', ylabel='s2')
#
#     if not os.path.exists(figpath + "Cond/"):
#         os.mkdir(figpath + "Cond/")
#     fig.savefig(figpath + "Cond/M_{:03d}.pdf".format(s))


def check_path_iter(path):
    i = 0
    while os.path.exists(path + "EIBV_%s/" % i):
        i += 1
    name = path + "EIBV_%s/" % i
    os.mkdir(name)
    print(name)
    print("new path has been made")
    return name

def check_path(path):
    if not os.path.exists(path):
        print("path is not existed, now make new path ")
        os.mkdir(path)
        print("new path has been made")


def distance_matrix(sites1v, sites2v):
    '''
    :param sites1v:
    :param sites2v:
    :return:
    '''
    n = len(sites1v)
    ddE = np.abs(sites1v * np.ones([1, n]) - np.ones([n, 1]) * sites1v.T)
    dd2E = ddE * ddE
    ddN = np.abs(sites2v * np.ones([1, n]) - np.ones([n, 1]) * sites2v.T)
    dd2N = ddN * ddN
    t = np.sqrt(dd2E + dd2N)
    return t


## Functions used
def Matern_cov(sigma, eta, t):
    '''
    :param sigma: scaling coef
    :param eta: range coef
    :param t: distance matrix
    :return: matern covariance
    '''
    return sigma ** 2 * (1 + eta * t) * np.exp(-eta * t)


def Exp_cov(eta, t):
    '''
    :param eta:
    :param t:
    :return:
    '''
    return np.exp(eta * t)


# def
def plotf(Y, string, xlim = None, ylim = None, vmin = None, vmax = None):
    '''
    :param Y:
    :param string:
    :return:
    '''
    # plt.figure(figsize=(5,5))
    fig = plt.gcf()
    plt.imshow(Y, vmin = vmin, vmax = vmax, extent=(xlim[0], xlim[1], ylim[0], ylim[1]))
    plt.title(string)
    plt.xlabel("s1")
    plt.ylabel("s2")
    plt.colorbar(fraction=0.045, pad=0.04)
    # plt.gca().invert_yaxis()
    # plt.show()
    # plt.savefig()
    return fig


def mu(H, beta):
    '''
    :param H: design matrix
    :param beta: regression coef
    :return: prior mean
    '''
    # beta = np.hstack((-alpha, alpha, alpha))
    return np.dot(H, beta)


def GRF2D(Sigma, F, T, y_sampled, mu_prior):
    '''
    :param Sigma:
    :param F:
    :param T:
    :param y_sampled:
    :param mu_prior:
    :return:
    '''
    Cmatrix = np.dot(F, np.dot(Sigma, F.T)) + T
    mu_posterior = mu_prior + np.dot(Sigma, np.dot(F.T, np.linalg.solve(Cmatrix, (y_sampled - np.dot(F, mu_prior)))))
    Sigma_posterior = Sigma - np.dot(Sigma, np.dot(F.T, np.linalg.solve(Cmatrix, np.dot(F, Sigma))))
    return (mu_posterior, Sigma_posterior)


def EIBV(Sig, H, R, mu, N, T_thres, S_thres):
    '''
    :param Sig:
    :param H: Sampling matrix
    :param R: Noise matrix
    :param mu: cond mean
    :param N: number of points to be integrateed
    :param T_thres: threshold for Temp
    :param S_thres: threshold for Salinity
    :return:
    '''

    # Update the field variance
    a = np.dot(Sig, H.T)
    b = np.dot(np.dot(H, Sig), H.T) + R
    c = np.dot(H, Sig)
    Sigxi = np.dot(a, np.linalg.solve(b, c))  # new covariance matrix
    V = Sig - Sigxi  # Uncertainty reduction # updated covariance

    IntA = 0.0

    # integrate out all elements in the bernoulli variance term
    for i in np.arange(0, N, 2):

        # extract the corresponding variance reduction term
        SigMxi = Sigxi[np.ix_([i, i + 1], [i, i + 1])]

        # extract the corresponding mean terms
        Mxi = [mu[i], mu[i + 1]] # temp and salinity

        sn2 = V[np.ix_([i, i + 1], [i, i + 1])]
        # vv2 = np.add(sn2, SigMxi) # was originally used to make it obscure
        vv2 = Sig[np.ix_([i, i + 1], [i, i + 1])]

        # compute the first part of the integration
        Thres = np.vstack((T_thres, S_thres))
        mur = np.subtract(Thres, Mxi)
        IntB_a = mvn.mvnun(np.array([[-np.inf], [-np.inf]]), np.zeros([2, 1]), mur, vv2)[0]

        # compute the second part of the integration, which is squared
        mm = np.vstack((Mxi, Mxi))
        # SS = np.array([[vv2, SigMxi], [SigMxi, vv2]]) # thought of it as a simplier version
        SS = np.add(np.vstack((np.hstack((sn2, np.zeros((2, 2)))), np.hstack((np.zeros((2, 2)), sn2)))),
                    np.vstack((np.hstack((SigMxi, SigMxi)), np.hstack((SigMxi, SigMxi)))))
        Thres = np.vstack((T_thres, S_thres, T_thres, S_thres))
        mur = np.subtract(Thres, mm)
        IntB_b = mvn.mvnun(np.array([[-np.inf], [-np.inf], [-np.inf], [-np.inf]]), np.zeros([4, 1]), mur, SS)[0]

        # compute the total integration
        IntA = IntA + np.nansum([IntB_a, -IntB_b])

    return IntA



def ExpectedVariance2(threshold, mu, Sig, H, R, eval_indexes, evar_debug=False):
    # __slots__ = ('Sigxi', 'Sig', 'muxi', 'a', 'b', 'c')
    # H, design matrix used for in front of beta
    # R, noise matrix
    # eval_indexes, indices for the grid points where EV is computed
    """
    Computes IntA = \sum_x \int  p_x(y) (1-p_x(y)) p (y) dy
    x is a discretization of the spatial domain
    y is the data
    p_x(y)=P(T_x<T_threshold , S_x < S_threshold | y) = ...
    \Phi_2_corrST ( [T_threshold-E(T_x|y)] /Std(T_x/y) , S_threshold-E(S_x|y)] /Std(S_x/y)] # once it is standardized, then it can be computed from the standard normal cdf
    E(T,S|y)=mu+Sig*H'*(H*Sig*H'+R)\(y-H mu ) = xi
    where xi \sim N (mu, Sig*H'*((H*Sig*H'+R)\(H*Sig)) is the only variable
    that matters in the integral and for each x, this is an integral over
    xi_x = (xi_xT,xi_xS)
    """
    # For debug
    # H = np.zeros((2*50,2*50*50))
    # H[0:50, 0:50] = np.eye(50)
    # H[50:100, 50:100] = np.eye(50)
    # R = 0.25 * np.eye(100)

    # Xi variable distribution N(muxi, Sigxi)
    a = np.dot(Sig, H.T)
    b = np.dot(np.dot(H, Sig), H.T) + R
    c = np.dot(H, Sig)
    Sigxi = np.dot(a, np.linalg.solve(b, c)) # new covariance matrix
    V = Sig - Sigxi  # Uncertainty reduction # updated covariance
    n = int(mu.flatten().shape[0]/2)
    muxi = np.copy(mu)

    IntA = 0.0
    pp = None

    if evar_debug:
        pp = []

    for i in eval_indexes:

        SigMxi = Sigxi[np.ix_([i, n+i], [i, n+i])]
        rho = V[i, n+i] / np.sqrt(V[i, i]*V[n+i, n+i])

        if np.isnan(rho):
            rho = 0.6

        Mxi = [muxi[i], muxi[n+i]]
        sn_1 = np.sqrt(V[i, i])
        sn_2 = np.sqrt(V[n+i, n+i])
        sn2 = np.array([[sn_1**2, sn_1*sn_2*rho], [sn_1*sn_2*rho, sn_2**2]])

        if evar_debug:
            pp.append(mvn.mvnun(np.array([[-np.inf], [-np.inf]]), np.array([[0], [0]]), np.subtract([threshold[0], threshold[1]], np.array(Mxi).ravel()), SigMxi)[0])

        mm = np.vstack((Mxi, Mxi))
        SS = np.add(np.vstack((np.hstack((sn2, np.zeros((2, 2)))), np.hstack((np.zeros((2, 2)), sn2)))), np.vstack((np.hstack((SigMxi, SigMxi)), np.hstack((SigMxi, SigMxi)))))
        vv2 = np.add(sn2, SigMxi)
        Thres = np.array([threshold[0], threshold[1]])
        mur = np.subtract(Thres, Mxi)
        IntB_a = mvn.mvnun(np.array([[-np.inf], [-np.inf]]), np.array([[0], [0]]), mur, vv2)[0]
        Thres = np.array([threshold[0], threshold[1], threshold[0], threshold[1]])
        mur = np.subtract(Thres, mm)
        IntB_b = mvn.mvnun(np.array([[-np.inf], [-np.inf], [-np.inf], [-np.inf]]), np.array([[0], [0], [0], [0]]), mur, SS)[0]

        IntA = IntA + np.nansum([IntB_a, -IntB_b])

    if evar_debug:
        plt.figure()
        plt.imshow(np.array(pp).reshape(30, 30))
        plt.show()

    return IntA