print("hello world")
from usr_func import *

# figpath = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Projects_practice/ES_3D_scratch/fig/EIBV_2D1_Rule_Based/"

#%%
data_path = '/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/In-situ/pmel-20170813T154236.000-lauv-xplore-1.nc'
# data_path = '/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/In-situ/pmel-20170813T152758.000-lauv-xplore-2.nc'
import netCDF4
data = netCDF4.Dataset(data_path)
lat = np.array(data['lat']).reshape(-1, 1)
lon = np.array(data['lon']).reshape(-1, 1)
salinity = np.array(data['sal']).reshape(-1, 1)
import matplotlib.pyplot as plt
plt.scatter(lon, lat, c = salinity, vmin = 33, cmap = "Paired")
plt.colorbar()
plt.show()

#%%
# check_path(figpath)

# Setup the grid
n1 = 25 # number of grid points along east direction, or x, or number of columns
n2 = 25 # number of grid points along north direction, or y, or number of rows
n = n1 * n2 # total number of  grid points

XLIM = [0, 1] # limits in the grid
YLIM = [0, 1]


sites1 = np.linspace(0, 1, n1)
sites2 = np.linspace(0, 1, n2)
sites1m, sites2m = np.meshgrid(sites1, sites2)
sites1v = sites1m.reshape(-1, 1) # sites1v is the vectorised version
sites2v = sites2m.reshape(-1, 1)

# Compute the distance matrix
grid = np.hstack((sites1v, sites2v))
t = scdist.cdist(grid, grid)
plotf(t, "Distance matrix for the grid", xlim = XLIM, ylim = YLIM)
plt.show()

#%%
# Simulate the initial random field
# alpha = 1.0 # beta as in regression model
sigma = 0.5  # scaling coef in matern kernel
eta = 2.5 # coef in matern kernel
tau = .01 # iid noise

S_thres = 29.12

# only one parameter is considered for salinity
beta = [[29.0], [.25], [0.0]] # [intercept, trend along east and north

Sigma = Matern_cov(sigma, eta, t)  # matern covariance
plotf(Sigma, "Matern covariance matrix for the grid", xlim = XLIM, ylim = YLIM)
plt.show()

#%%

L = np.linalg.cholesky(Sigma)  # lower triangle covariance matrix
z = np.dot(L, np.random.randn(n).reshape(-1, 1)) # sampled randomly with covariance structure

# generate the prior of the field
H = np.hstack((np.ones([n, 1]), sites1v, sites2v)) # different notation for the project
mu_prior = mu(H, beta).reshape(n, 1)
plt.close("all")

# rearrange the long vector, so it contains the information having temperature and salinity
# grouped together for each location, i.e., mu_t1, mu_s1, mu_t2, mu_s2, ...,
fig = plotf(np.copy(mu_prior).reshape(n2, n1), "Salinity prior mean", xlim = XLIM, ylim = YLIM)
fig.savefig(figpath + "Prior_Sal.pdf")
plt.show()
plt.close("all")

mu_real = mu_prior + z  # add covariance structured noise
fig = plotf(np.copy(mu_real).reshape(n2, n1).T, "True salinity field", xlim = XLIM, ylim = YLIM)
fig.savefig(figpath + "True_Sal.pdf")
plt.show()
plt.close("all")

# Compute the ES
# Section II: compute its corresponding excursion probabilities
def EP_1D(mu, Sig, Thres):
    '''
    :param mu:
    :param Sig:
    :param T_thres:
    :param S_thres:
    :return:
    '''
    n = mu.shape[0]
    ES_Prob = np.zeros([n, 1])
    for i in range(n):
        ES_Prob[i] = norm.cdf(Thres, mu[i], Sig[i, i])
    return ES_Prob


def EP_1D_mvn(mu, Sig, Thres):
    '''
    :param mu:
    :param Sig:
    :param T_thres:
    :param S_thres:
    :return:
    '''
    n = mu.shape[0]
    ES_Prob = np.zeros([n, 1])
    for i in range(n):
        ES_Prob[i] = mvn.mvnun(-np.inf, Thres, mu[i], Sig[i, i])[0]
    return ES_Prob


def compute_ES(mu, Thres):
    '''
    :param mu:
    :param Tthres:
    :return:
    '''
    excursion = np.copy(mu)

    excursion[mu > Thres] = 0
    excursion[mu < Thres] = 1

    return excursion


ES_Prob_prior = EP_1D_mvn(mu_prior, Sigma, S_thres)
ES_Prob_m_prior = np.array(ES_Prob_prior).reshape(n2, n1) # ES_Prob_m is the reshaped matrix form of ES_Prob
fig = plt.figure(figsize=(5, 5))
fig = plotf(ES_Prob_m_prior, "Excursion probability of the prior field", xlim = XLIM, ylim = YLIM)
fig.savefig(figpath + "EP_prior.pdf")
plt.close("all")

ES_Prob_true = EP_1D_mvn(mu_real, Sigma, S_thres)
ES_Prob_m_true = np.array(ES_Prob_true).reshape(n2, n1)
fig = plt.figure(figsize=(5, 5))
fig = plotf(ES_Prob_m_true, "Excursion probability of the real field", xlim = XLIM, ylim = YLIM)
fig.savefig(figpath + "EP_real.pdf")
plt.close("all")
#% Plot excursion set in multiple patches

# Recompute the thresholds to make it stable
S_thres = np.mean(mu_real)

ES_true = compute_ES(mu_real, S_thres)
ES_true_m = np.array(ES_true).reshape(n2, n1)
fig = plt.figure(figsize=(5, 5))
fig = plotf(ES_true_m, "Excursion set of the real field", xlim = XLIM, ylim = YLIM)
fig.savefig(figpath + "ES_real.pdf")
plt.close("all")

ES_prior = compute_ES(mu_prior, S_thres)
ES_prior_m = np.array(ES_prior).reshape(n2, n1)
fig = plt.figure(figsize=(5, 5))
fig = plotf(ES_prior_m, "Excursion set of the prior field", xlim = XLIM, ylim = YLIM)
fig.savefig(figpath + "ES_prior.pdf")
plt.close("all")


#%% Method I: functions used for path planning purposes


def find_starting_loc(ep, n1, n2):
    '''
    This will find the starting location in
    the grid according to the excursion probability
    which is closest to 0.5
    :param ep:
    :return:
    '''
    ep_criterion = 0.5
    ind = (np.abs(ep - ep_criterion)).argmin()
    row_ind, col_ind = np.unravel_index(ind, (n2, n1))
    return row_ind, col_ind

print(find_starting_loc(ES_Prob_prior, n1, n2))


def find_neighbouring_loc(row_ind, col_ind):
    '''
    This will find the neighbouring loc
    But also limit it inside the grid
    :param idx:
    :param idy:
    :return:
    '''

    row_ind_l = [row_ind - 1 if row_ind > 0 else row_ind]
    row_ind_u = [row_ind + 1 if row_ind < n2 - 1 else row_ind]
    col_ind_l = [col_ind - 1 if col_ind > 0 else col_ind]
    col_ind_u = [col_ind + 1 if col_ind < n1 - 1 else col_ind]

    row_ind_v = np.unique(np.vstack((row_ind_l, row_ind, row_ind_u)))
    col_ind_v = np.unique(np.vstack((col_ind_l, col_ind, col_ind_u)))

    row_ind, col_ind = np.meshgrid(row_ind_v, col_ind_v)

    return row_ind.reshape(-1, 1), col_ind.reshape(-1, 1)


def ExpectedVarianceUsr(threshold, mu, Sig, F, R):
    '''
    :param threshold:
    :param mu:
    :param Sig:
    :param F: sampling matrix
    :param R: noise matrix
    :return:
    '''
    Sigxi = np.dot(Sig, np.dot(F.T, np.linalg.solve(np.dot(F, np.dot(Sig, F.T)) + R, np.dot(F, Sig))))
    V = Sig - Sigxi
    sa2 = np.diag(V).reshape(-1, 1) # the corresponding variance term for each location
    IntA = 0.0
    for i in range(len(mu)):
        sn2 = sa2[i]
        sn = np.sqrt(sn2) # the corresponding standard deviation term
        m = mu[i]
        # mur = (threshold - m) / sn
        IntA = IntA + mvn.mvnun(-np.inf, threshold, m, sn2)[0] - mvn.mvnun(-np.inf, threshold, m, sn2)[0] ** 2

    return IntA


def find_next_EIBV(row_neighbour, col_neighbour, row_now, col_now, Sig, mu, tau, S_thres):

    id = []

    for i in row_neighbour:
        for j in col_neighbour:
            if i == row_now and j == col_now:
                continue
            id.append(np.ravel_multi_index((i, j), (n2, n1)))
    id = np.unique(np.array(id))

    M = len(id)
    R = tau ** 2

    eibv = []
    for k in range(M):
        F = np.zeros([1, n])
        F[0, id[k]] = True
        eibv.append(ExpectedVarianceUsr(S_thres, mu, Sig, F, R))
    ind_desired = np.argmin(np.array(eibv))
    row_next, col_next = np.unravel_index(id[ind_desired], (n2, n1))
    return row_next, col_next, id, eibv


def GPupd(mu, Sig, R, F, y_sampled):
    C = np.dot(F, np.dot(Sig, F.T)) + R
    mu_p = mu + np.dot(Sig, np.dot(F.T, np.linalg.solve(C, (y_sampled - np.dot(F, mu)))))
    Sigma_p = Sig - np.dot(Sig, np.dot(F.T, np.linalg.solve(C, np.dot(F, Sig))))
    return mu_p, Sigma_p


#%% Static design
# move a vertical line
M = n2
mu_cond = mu_prior
Sigma_cond = Sigma
path_row = []
path_col = []

if not os.path.exists(figpath + "Static/"):
    os.mkdir(figpath + "Static/")
for j in range(M):
    F = np.zeros([1, n])
    F[0, np.ravel_multi_index((j, 12), (n1, n2))] = True # select col 13 to move along
    path_row.append(j) # only for plotting
    path_col.append(12)

    R = np.diagflat(tau ** 2)
    y_sampled = np.dot(F, mu_real) + tau * np.random.randn(1)

    mu_cond, Sigma_cond = GPupd(mu_cond, Sigma_cond, R, F, y_sampled)

    ES_Prob = EP_1D(mu_cond, Sigma_cond, S_thres)
    ES_Prob_m = np.array(ES_Prob).reshape(n1, n2) # ES_Prob_m is the reshaped matrix form of ES_Prob

    fig = plt.figure(figsize=(35, 5))
    gs = GridSpec(nrows=1, ncols=5)

    i = 0
    axes = fig.add_subplot(gs[i])
    im = axes.imshow(ES_Prob_m, vmin=0, vmax=1);
    plt.title("Excursion probability on cond field")
    plt.plot(path_col, path_row, 'r.-', linewidth=2)
    plt.xlabel("s1");
    plt.ylabel("s2");
    plt.colorbar(im, fraction=0.045, pad=0.04);

    i = 1
    axes = fig.add_subplot(gs[i])
    im = axes.imshow(mu_cond.reshape(n2, n1));
    plt.title("Cond mean")
    plt.plot(path_col, path_row, 'r.-', linewidth=2)
    plt.xlabel("s1");
    plt.ylabel("s2");
    plt.colorbar(im, fraction=0.045, pad=0.04);

    i = 2
    axes = fig.add_subplot(gs[i])
    im = axes.imshow(np.sqrt(np.diag(Sigma_cond)).reshape(n2, n1));
    plt.title("Prediction error")
    plt.plot(path_col, path_row, 'r.-', linewidth=2)
    plt.xlabel("s1");
    plt.ylabel("s2");
    plt.colorbar(im, fraction=0.045, pad=0.04);

    i = 3
    axes = fig.add_subplot(gs[i])
    im = axes.imshow(ES_Prob_m_true, vmin=0, vmax=1);
    plt.title("True Excursion probablity")
    plt.plot(path_col, path_row, 'r.-', linewidth=2)
    plt.xlabel("s1");
    plt.ylabel("s2");
    plt.colorbar(im, fraction=0.045, pad=0.04);

    i = 4
    axes = fig.add_subplot(gs[i])
    es_mu = compute_ES(mu_real, S_thres)
    im = axes.imshow(es_mu.reshape(n2, n1), vmin=0, vmax=1);
    plt.title("True Excursion Set")
    plt.plot(path_col, path_row, 'r.-', linewidth=2)
    plt.xlabel("s1");
    plt.ylabel("s2");
    plt.colorbar(im, fraction=0.045, pad=0.04);

    plt.savefig(figpath + "Static/S_{:03d}.png".format(j))



#%% Method I: EIBV grid path planning design
# plan the path according to EIBV criterion
figpath = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Projects_practice/ES_3D_scratch/fig/EIBV_2D/"
N_steps = 200
row_start, col_start = find_starting_loc(np.array(ES_Prob_prior), n1, n2)


row_now = row_start
col_now = col_start

row_now = 15
col_now = 0

mu_posterior = mu_prior
Sigma_posterior = Sigma
noise = tau ** 2
R = np.diagflat(noise)

path_row = []
path_col = []
path_row.append(row_now)
path_col.append(col_now) # only for plotting

for j in range(N_steps):
    row_neighbour, col_neighbour = find_neighbouring_loc(row_now, col_now)

    row_next, col_next, id, eibv = find_next_EIBV(row_neighbour, col_neighbour, row_now, col_now,
                                        Sigma_posterior, mu_posterior, tau, S_thres)

    row_now, col_now = row_next, col_next
    path_row.append(row_now)
    path_col.append(col_now)

    ind_next = np.ravel_multi_index((row_next, col_next), (n1, n2))
    F = np.zeros([1, n])
    F[0, ind_next] = True

    y_sampled = np.dot(F, mu_real) + tau * np.random.randn(1).reshape(-1, 1)

    mu_posterior, Sigma_posterior = GPupd(mu_posterior, Sigma_posterior, R, F, y_sampled)

    ES_Prob = EP_1D(mu_posterior, Sigma_posterior, S_thres)

    ES_Prob_m = np.array(ES_Prob).reshape(n1, n2) # ES_Prob_m is the reshaped matrix form of ES_Prob


    fig = plt.figure(figsize=(35, 5))
    gs = GridSpec(nrows=1, ncols=5)

    i = 0
    axes = fig.add_subplot(gs[i])
    im = axes.imshow(ES_Prob_m, vmin = 0, vmax = 1); plt.title("Excursion probability on cond field")
    plt.plot(path_col, path_row, 'r.-', linewidth=2)
    plt.xlabel("s1");plt.ylabel("s2");plt.colorbar(im, fraction=0.045, pad=0.04);

    i = 1
    axes = fig.add_subplot(gs[i])
    im = axes.imshow(mu_posterior.reshape(n1, n2)); plt.title("Cond mean")
    plt.plot(path_col, path_row, 'r.-', linewidth=2)
    plt.xlabel("s1");plt.ylabel("s2");plt.colorbar(im, fraction=0.045, pad=0.04);

    i = 2
    axes = fig.add_subplot(gs[i])
    im = axes.imshow(np.sqrt(np.diag(Sigma_posterior)).reshape(n1, n2)); plt.title("Prediction error")
    plt.plot(path_col, path_row, 'r.-', linewidth=2)
    plt.xlabel("s1");plt.ylabel("s2");plt.colorbar(im, fraction=0.045, pad=0.04);

    i = 3
    axes = fig.add_subplot(gs[i])
    im = axes.imshow(ES_Prob_m_true, vmin = 0, vmax = 1); plt.title("True Excursion probablity")
    plt.plot(path_col, path_row, 'r.-', linewidth=2)
    plt.xlabel("s1");plt.ylabel("s2");plt.colorbar(im, fraction=0.045, pad=0.04);

    i = 4
    axes = fig.add_subplot(gs[i])
    es_mu = compute_ES(mu_real, S_thres)
    im = axes.imshow(es_mu.reshape(n2, n1), vmin=0, vmax=1);
    plt.title("True Excursion Set")
    plt.plot(path_col, path_row, 'r.-', linewidth=2)
    plt.xlabel("s1");
    plt.ylabel("s2");
    plt.colorbar(im, fraction=0.045, pad=0.04);


    plt.savefig(figpath + "EIBV/E_{:03d}.png".format(j))

    # if not os.path.exists(figpath + "Cond/"):
    #     os.mkdir(figpath + "Cond/")
    # fig.savefig(figpath + "Cond/M_{:03d}.pdf".format(s))




#%% Method II: Supplementary functions for dynamic EIBV


def find_starting_loc_dynamic(mu_prior, Thres):
    '''
    :param mu_prior:
    :param Thres:
    :return: returns the starting locations in the grid
    '''
    ind_v = (np.abs(mu_prior - Thres)).argmin()
    x_start = sites1v[ind_v]
    y_start = sites2v[ind_v]
    return x_start, y_start


def find_candidates_loc(x_now, y_now, heading, radius, no_candidiates, angle_gap):
    '''
    This will find all possible candidates in the circular range
    :param x_now:
    :param y_now:
    :param heading:
    :param radius:
    :param no_candidiates: This is the number of candidates per side, in total needs to be doubled
    :param angle_gap:
    :return:
    '''
    candidates_angles = []
    for i in range(no_candidiates + 1):
        candidates_angles.append(heading + i * angle_gap)
        candidates_angles.append(heading - i * angle_gap)
    candidates_angles = np.unique(np.array(candidates_angles))
    # print(candidates_angles)

    x_cand = []
    y_cand = []
    heading_cand = []
    for j in range(len(candidates_angles)):
        ang = candidates_angles[j]
        x_cand.append(x_now + np.cos(ang / 180.0 * np.pi) * radius)
        y_cand.append(y_now + np.sin(ang / 180.0 * np.pi) * radius)
        heading_cand.append(ang)

    x_cand = np.array(x_cand)
    y_cand = np.array(y_cand)
    heading_cand = np.array(heading_cand)
    xc = x_cand[(x_cand <= XLIM[1]) * (x_cand >= XLIM[0]) * (y_cand <= YLIM[1]) * (y_cand >= YLIM[0])]
    yc = y_cand[(x_cand <= XLIM[1]) * (x_cand >= XLIM[0]) * (y_cand <= YLIM[1]) * (y_cand >= YLIM[0])]
    headingc = heading_cand[(x_cand <= XLIM[1]) * (x_cand >= XLIM[0]) * (y_cand <= YLIM[1]) * (y_cand >= YLIM[0])]

    return xc, yc, headingc


def EIBV(threshold, mu, Sigma):
    '''
    :param threshold:
    :param mu:
    :param Sig:
    :param R: noise matrix
    :return:
    '''
    sa2 = np.diag(Sigma) # the corresponding variance term for each location
    IntA = 0.0
    for i in range(len(mu)):
        sn2 = sa2[i]
        m = mu[i]
        IntA = IntA + mvn.mvnun(-np.inf, threshold, m, sn2)[0] - mvn.mvnun(-np.inf, threshold, m, sn2)[0] ** 2
    return IntA



def GPupd_xy(x_cand, y_cand, Sigma, mu, tau):
    '''
    :param x_cand:
    :param y_cand:
    :param Sigma:
    :param mu:
    :param tau:
    :return:
    '''

    obs_loc = np.hstack((x_cand, y_cand)).reshape(1, -1)
    Xobs = np.hstack((1, x_cand, y_cand)).reshape(1, -1)

    t_obs = scdist.cdist(obs_loc, obs_loc)  # distance matrix for the observation
    C = Matern_cov(sigma, eta, t_obs) + tau ** 2
    LC = np.linalg.cholesky(C)
    z = np.random.randn(1)
    y_sampled = np.dot(LC, z) + np.dot(Xobs, beta) + np.random.randn(1) * tau

    t_grid_obs = scdist.cdist(grid, obs_loc)  # distance matrix between grid the observation
    C0_ = Matern_cov(sigma, eta, t_grid_obs)

    C0 = Sigma

    # mu_posterior = mu + np.dot(C0_, np.linalg.solve(C, (y_sampled - np.dot(Xobs, beta))))
    col_close = np.abs(sites1 - x_cand).argmin()
    row_close = np.abs(sites2 - y_cand).argmin()
    mu_m = mu.reshape(n2, n1)

    mu_posterior = mu + np.dot(C0_, np.linalg.solve(C, (y_sampled - mu_m[row_close, col_close])))
    Sigma_posterior = C0 - np.dot(C0_, np.linalg.solve(C, C0_.T))

    return mu_posterior, Sigma_posterior



# #%% test of GP for random locations
#
# x1 = 0.5
# x2 = 0.25
# y1 = 0.5
# y2 = 0.25
#
# x3 = 0
# y3 = 0
#
# mu_cond, Sigma_cond = GPupd_xy(x1, y1, Sigma, mu_prior, tau)
# plotf(np.diag(Sigma_cond).reshape(n2, n1), "Prec", xlim = XLIM, ylim = YLIM)
# plt.show()
#
# mu_cond, Sigma_cond = GPupd_xy(x2, y2, Sigma_cond, mu_cond, tau)
# plotf(np.diag(Sigma_cond).reshape(n2, n1), "Prec", xlim = XLIM, ylim = YLIM)
# plt.show()
#
# mu_cond, Sigma_cond = GPupd_xy(x3, y3, Sigma_cond, mu_cond, tau)
# plotf(np.diag(Sigma_cond).reshape(n2, n1), "Prec", xlim = XLIM, ylim = YLIM)
# plt.show()
#
# a = EP_1D_mvn(mu_cond, Sigma_cond, S_thres)
# plotf(a.reshape(n2, n1), "ep", xlim = XLIM, ylim = YLIM)
# plt.show()

## %% continued function for sampling design

def find_next_loc_EIBV(x_cand, y_cand, heading_cand, x_now, y_now, heading_now, Sigma, mu, tau, Thres):
    '''
    :param x_cand:
    :param y_cand:
    :param heading_cand:
    :param Sigma:
    :param mu:
    :param tau:
    :param Thres:
    :return:
    '''
    eibv = []
    print(x_cand, y_cand)
    print(len(x_cand))
    for j in range(len(x_cand)):
        mu_posterior, Sigma_posterior = GPupd_xy(x_cand[j], y_cand[j], Sigma, mu, tau)
        eibv.append(EIBV(Thres, mu_posterior, Sigma_posterior))
    ind_desired = np.argmin(np.array(eibv))
    x_next, y_next, heading_next = x_cand[ind_desired], y_cand[ind_desired], heading_cand[ind_desired]
    if x_next == x_now and y_next == y_now and heading_next == heading_now:
        x_next, y_next, heading_next = x_cand[np.random.randint(len(x_cand))], \
                                       y_cand[np.random.randint(len(x_cand))], \
                                       heading_cand[np.random.randint(len(x_cand))]
    return x_next, y_next, heading_next



#%% Method II: Using Dynamic path planning for generating candidates points
figpath = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Projects_practice/ES_3D_scratch/fig/EIBV_2D1_Dynamic/"
N_steps = 10
x_start, y_start = find_starting_loc_dynamic(mu_prior, S_thres)

print(x_start, y_start)

# x_now = x_start
# y_now = y_start

x_now = 0
y_now = 0

mu_posterior = mu_prior
Sigma_posterior = Sigma

path_x = []
path_y = []
path_x.append(x_now)
path_y.append(y_now) # only for plotting

heading_start = 0
heading = []
heading.append(heading_start)
heading_now = heading_start
radius = 0.05
no_candidates = 2
angle_gap = 30
#
# x_cand = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# y_cand = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# for j in range(len(x_cand)):
for j in range(N_steps):
    x_cand, y_cand, heading_cand = find_candidates_loc(x_now, y_now, heading_now, radius, no_candidates, angle_gap)
    x_next, y_next, heading_next = find_next_loc_EIBV(x_cand, y_cand, heading_cand, x_now, y_now, heading_now, Sigma_posterior, mu_posterior, tau, S_thres)

    x_now, y_now, heading_now = x_next, y_next, heading_next
    path_x.append(x_now)
    path_y.append(y_now)
    heading.append(heading_now)

    #==== testing part ======
    # print(x_next, y_next, heading_next)
    # x_next = x_cand[j]
    # y_next = y_cand[j]
    # x_now, y_now = x_next, y_next
    # path_x.append(x_now)
    # path_y.append(y_now)
    # print(x_next, y_next)

    # mu_posterior, Sigma_posterior = GPupd_xy(x_next, y_next, Sigma_posterior, mu_posterior, tau)

    # plotf(np.diag(Sigma_posterior).reshape(n2, n1), "Pred", xlim = XLIM, ylim = YLIM)
    # plt.show()

    # plotf(mu_posterior.reshape(n2, n1), "Cond", xlim=XLIM, ylim=YLIM)
    # plt.show()
    # plotf(ES_Prob_m, "ESPEROB", xlim = XLIM, ylim = YLIM)
    # plt.show()
    #====== end of testing ===

    mu_posterior, Sigma_posterior = GPupd_xy(x_next, y_next, Sigma_posterior, mu_posterior, tau)

    ES_Prob = EP_1D_mvn(mu_posterior, Sigma_posterior, S_thres)
    ES_Prob_m = np.rot90(np.array(ES_Prob).reshape(n2, n1)) # ES_Prob_m is the reshaped matrix form of ES_Prob

    fig = plt.figure(figsize=(35, 5))
    gs = GridSpec(nrows=1, ncols=5)

    xlim = [0, 1]
    ylim = [0, 1]

    i = 0
    axes = fig.add_subplot(gs[i])
    im = axes.imshow(ES_Prob_m, vmin = 0, vmax = 1, extent=(xlim[0], xlim[1], ylim[0], ylim[1]));
    plt.title("Excursion probability on cond field"); plt.plot(path_x, path_y, 'r.-', linewidth=2)
    plt.xlabel("s1");plt.ylabel("s2");plt.colorbar(im, fraction=0.045, pad=0.04);

    i = 1
    axes = fig.add_subplot(gs[i])
    im = axes.imshow(np.rot90(mu_posterior.reshape(n2, n1)), extent=(xlim[0], xlim[1], ylim[0], ylim[1]));
    plt.title("Cond mean"); plt.plot(path_x, path_y, 'r.-', linewidth=2)
    plt.xlabel("s1");plt.ylabel("s2");plt.colorbar(im, fraction=0.045, pad=0.04);

    i = 2
    axes = fig.add_subplot(gs[i])
    # im = axes.imshow(np.rot90(np.sqrt(np.diag(Sigma_posterior)).reshape(n2, n1)), extent=(xlim[0], xlim[1], ylim[0], ylim[1]));
    im = axes.imshow(np.rot90(np.diag(Sigma_posterior).reshape(n2, n1)),
                     extent=(xlim[0], xlim[1], ylim[0], ylim[1]));
    plt.title("Prediction error"); plt.plot(path_x, path_y, 'r.-', linewidth=2)
    plt.xlabel("s1");plt.ylabel("s2");plt.colorbar(im, fraction=0.045, pad=0.04);

    i = 3
    axes = fig.add_subplot(gs[i])
    im = axes.imshow(np.rot90(ES_Prob_m_true), vmin = 0, vmax = 1, extent=(xlim[0], xlim[1], ylim[0], ylim[1]));
    plt.title("True Excursion probablity"); plt.plot(path_x, path_y, 'r.-', linewidth=2)
    plt.xlabel("s1");plt.ylabel("s2");plt.colorbar(im, fraction=0.045, pad=0.04);

    i = 4
    axes = fig.add_subplot(gs[i])
    es_mu = compute_ES(mu_real, S_thres)
    im = axes.imshow(np.rot90(es_mu.reshape(n2, n1)), vmin=0, vmax=1, extent=(xlim[0], xlim[1], ylim[0], ylim[1]));
    plt.title("True Excursion Set")
    plt.plot(path_x, path_y, 'r.-', linewidth=2)
    plt.xlabel("s1");
    plt.ylabel("s2");
    plt.colorbar(im, fraction=0.045, pad=0.04);

    plt.savefig(figpath + "EIBV/E_{:03d}.png".format(j))

    # if not os.path.exists(figpath + "Cond/"):
    #     os.mkdir(figpath + "Cond/")
    # fig.savefig(figpath + "Cond/M_{:03d}.pdf".format(s))

plt.close("all")

#%% Method III: Functions used for rule-based system
'''
In this section, the rule-based system is implemented to do the filtering 
procedure on the candidate grid nodes where unrealistic nodes are eliminated 
using cross-product rule, which only allows the AUV to run with a certain operational 
angle limits 
'''


def find_starting_loc_rule(ep):
    '''
    This will find the starting location in
    the grid according to the excursion probability
    which is closest to 0.5
    :param ep:
    :return:
    '''
    ep_criterion = 0.5
    ind = (np.abs(ep - ep_criterion)).argmin()
    row_ind, col_ind = np.unravel_index((ind), (n2, n1))
    return row_ind, col_ind

# print(find_starting_loc_rule(ES_Prob_prior))


def find_candidates_loc_rule(row_ind, col_ind):
    '''
    This will find the neighbouring loc
    But also limit it inside the grid
    :param idx:
    :param idy:
    :return:
    '''

    row_ind_l = [row_ind - 1 if row_ind > 0 else row_ind]
    row_ind_u = [row_ind + 1 if row_ind < n2 - 1 else row_ind]
    col_ind_l = [col_ind - 1 if col_ind > 0 else col_ind]
    col_ind_u = [col_ind + 1 if col_ind < n1 - 1 else col_ind]

    row_ind_v = np.unique(np.vstack((row_ind_l, row_ind, row_ind_u)))
    col_ind_v = np.unique(np.vstack((col_ind_l, col_ind, col_ind_u)))

    row_ind, col_ind = np.meshgrid(row_ind_v, col_ind_v)

    return row_ind.reshape(-1, 1), col_ind.reshape(-1, 1)


def find_next_EIBV_rule(row_cand, col_cand, row_now, col_now, row_previous, col_previous, Sig, mu, tau, Thres):

    id = []
    drow1 = row_now - row_previous
    dcol1 = col_now - col_previous
    vec1 = np.array([dcol1, drow1])
    for i in row_cand:
        for j in col_cand:
            if i == row_now and j == col_now:
                continue
            drow2 = i - row_now
            dcol2 = j - col_now
            vec2 = np.array([dcol2, drow2])
            if np.dot(vec1, vec2) >= 0: # add the rule for not turning sharply
                id.append(np.ravel_multi_index((i, j), (n2, n1)))
            else:
                continue
    id = np.unique(np.array(id))

    M = len(id)
    R = tau ** 2

    eibv = []
    for k in range(M):
        F = np.zeros([1, n])
        F[0, id[k]] = True
        eibv.append(ExpectedVarianceUsr(Thres, mu, Sig, F, R))
    ind_desired = np.argmin(np.array(eibv))
    row_next, col_next = np.unravel_index(id[ind_desired], (n2, n1))
    return row_next, col_next, id, eibv


#%% Method III: Running script
figpath = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Projects_practice/ES_3D_scratch/fig/EIBV_2D1_Rule_Based/"
N_steps = 100
row_start, col_start = find_starting_loc_rule(np.array(ES_Prob_prior))
row_now = row_start
col_now = col_start

# row_now = 15
# col_now = 0
row_previous = row_now
col_previous = col_now

mu_posterior = mu_prior
Sigma_posterior = Sigma
noise = tau ** 2
R = np.diagflat(noise)

path_row = []
path_col = []
path_row.append(row_now)
path_col.append(col_now) # only for plotting

for j in range(N_steps):
    row_cand, col_cand = find_candidates_loc_rule(row_now, col_now)

    row_next, col_next, id, eibv = find_next_EIBV_rule(row_cand, col_cand, row_now, col_now, row_previous,
                                             col_previous, Sigma_posterior, mu_posterior, tau, S_thres)
    # row_next, col_next, id, eibv = find_next_EIBV(row_neighbour, col_neighbour, row_now, col_now,
    #                                     Sigma_posterior, mu_posterior, tau, S_thres)

    row_previous, col_previous = row_now, col_now
    row_now, col_now = row_next, col_next
    path_row.append(row_now)
    path_col.append(col_now)

    ind_next = np.ravel_multi_index((row_next, col_next), (n1, n2))
    F = np.zeros([1, n])
    F[0, ind_next] = True

    y_sampled = np.dot(F, mu_real) + tau * np.random.randn(1).reshape(-1, 1)

    mu_posterior, Sigma_posterior = GPupd(mu_posterior, Sigma_posterior, R, F, y_sampled)

    ES_Prob = EP_1D(mu_posterior, Sigma_posterior, S_thres)

    ES_Prob_m = np.array(ES_Prob).reshape(n1, n2) # ES_Prob_m is the reshaped matrix form of ES_Prob


    fig = plt.figure(figsize=(35, 5))
    gs = GridSpec(nrows=1, ncols=5)

    i = 0
    axes = fig.add_subplot(gs[i])
    im = axes.imshow(ES_Prob_m, vmin = 0, vmax = 1); plt.title("Excursion probability on cond field")
    plt.plot(path_col, path_row, 'r.-', linewidth=2)
    plt.xlabel("s1");plt.ylabel("s2");plt.colorbar(im, fraction=0.045, pad=0.04);

    i = 1
    axes = fig.add_subplot(gs[i])
    im = axes.imshow(mu_posterior.reshape(n1, n2)); plt.title("Cond mean")
    plt.plot(path_col, path_row, 'r.-', linewidth=2)
    plt.xlabel("s1");plt.ylabel("s2");plt.colorbar(im, fraction=0.045, pad=0.04);

    i = 2
    axes = fig.add_subplot(gs[i])
    im = axes.imshow(np.sqrt(np.diag(Sigma_posterior)).reshape(n1, n2)); plt.title("Prediction error")
    plt.plot(path_col, path_row, 'r.-', linewidth=2)
    plt.xlabel("s1");plt.ylabel("s2");plt.colorbar(im, fraction=0.045, pad=0.04);

    i = 3
    axes = fig.add_subplot(gs[i])
    im = axes.imshow(ES_Prob_m_true, vmin = 0, vmax = 1); plt.title("True Excursion probablity")
    plt.plot(path_col, path_row, 'r.-', linewidth=2)
    plt.xlabel("s1");plt.ylabel("s2");plt.colorbar(im, fraction=0.045, pad=0.04);

    i = 4
    axes = fig.add_subplot(gs[i])
    es_mu = compute_ES(mu_real, S_thres)
    im = axes.imshow(es_mu.reshape(n2, n1), vmin=0, vmax=1);
    plt.title("True Excursion Set")
    plt.plot(path_col, path_row, 'r.-', linewidth=2)
    plt.xlabel("s1");
    plt.ylabel("s2");
    plt.colorbar(im, fraction=0.045, pad=0.04);


    plt.savefig(figpath + "EIBV/E_{:03d}.png".format(j))
    plt.close("all")
    # if not os.path.exists(figpath + "Cond/"):
    #     os.mkdir(figpath + "Cond/")
    # fig.savefig(figpath + "Cond/M_{:03d}.pdf".format(s))


plt.close("all")

#%%
np.savetxt(figpath+"path_row.txt", path_row, fmt = '%02d')
np.savetxt(figpath+"path_col.txt", path_col, fmt = '%02d')

#%% load text file and compute
path_x = np.loadtxt(figpath+"path_row.txt")



#%% make animation
import imageio
top_path_eibv = figpath + "EIBV/"
top_path_static = figpath + "Static/"

png_dir = top_path_static
png_dir = top_path_eibv


# os.chdir(top_path)

def make_folder(folder_name):
    if not os.path.exists(folder_name):
        print(folder_name + " is created")
        os.mkdir(folder_name)
    else:
        print(folder_name + " is already existed")


# start to make animation for salinity variation at each layer
# for layer in range(5):
#     png_dir = "fig/dir_sal/" + str(layer) # figure path
    # print(png_dir)
    # make_folder(png_dir)

i = 0
while os.path.exists(figpath + "/Animation/EIBV_%s.gif" % i):
    i += 1
name = figpath + "/Animation/EIBV_%s.gif" % i

image_file_names = []
images = []

# print(os.listdir(png_dir))
for file_name in os.listdir(png_dir):
    if file_name.endswith('.png'):
        # print(file_name)
        image_file_names.append(file_name)

sorted_files = sorted(image_file_names)

frame_length = 0.1 # seconds between frames
end_pause = 4 # seconds to stay on last frame
# loop through files, join them to image array, and write to GIF called 'wind_turbine_dist.gif'
# for ii in range(0,len(sorted_files)):
for ii in range(0,len(sorted_files)):
    file_path = os.path.join(png_dir, sorted_files[ii])
    if ii==len(sorted_files)-1:
        for jj in range(0,int(end_pause/frame_length)):
            images.append(imageio.imread(file_path))
    else:
        images.append(imageio.imread(file_path))
    print(ii)
# the duration is the time spent on each image (1/duration is frame rate)
imageio.mimsave(name, images,'GIF',duration=frame_length)

print("finished")


