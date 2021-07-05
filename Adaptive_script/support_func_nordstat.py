from usr_func import *

# Setup the grid
n1 = 10 # number of grid points along east direction
n2 = 10 # number of grid points along north direction
n3 = 10 # number of layers in the depth dimension
n = n1 * n2 * n3 # total number of grid points
N = n

x = np.linspace(0, 1, n1)
y = np.linspace(0, 1, n2)
z = np.linspace(0, 1, n3)
xx, yy, zz = np.meshgrid(x, y, z)

xv = xx.reshape(-1, 1) # sites1v is the vectorised version
yv = yy.reshape(-1, 1)
zv = zz.reshape(-1, 1)
grid = np.hstack((xv, yv, zv))
t = scdist.cdist(grid, grid)
# Simulate the initial random field
sigma = 0.5  # scaling coef in matern kernel
eta = 5.5 # coef in matern kernel
tau = .05 # iid noise
S_thres = 1.5

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


def find_starting_loc(ep, n1, n2, n3):
    '''
    This will find the starting location in
    the grid according to the excursion probability
    which is closest to 0.5
    :param ep:
    :return:
    '''
    ep_criterion = 0.5
    ind = (np.abs(ep - ep_criterion)).argmin()
    row_ind, col_ind, dep_ind = np.unravel_index(ind, (n2, n1, n3))
    return row_ind, col_ind, dep_ind


def find_neighbouring_loc(row_ind, col_ind, dep_ind):
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
    dep_ind_l = [dep_ind - 1 if dep_ind > 0 else dep_ind]
    dep_ind_u = [dep_ind + 1 if dep_ind < n3 - 1 else dep_ind]

    row_ind_v = np.unique(np.vstack((row_ind_l, row_ind, row_ind_u)))
    col_ind_v = np.unique(np.vstack((col_ind_l, col_ind, col_ind_u)))
    dep_ind_v = np.unique(np.vstack((dep_ind_l, dep_ind, dep_ind_u)))

    row_ind, col_ind, dep_ind = np.meshgrid(row_ind_v, col_ind_v, dep_ind_v)

    return row_ind.reshape(-1, 1), col_ind.reshape(-1, 1), dep_ind.reshape(-1, 1)


def find_candidates_loc(row_ind, col_ind, dep_ind):
    '''
    :param row_ind:
    :param col_ind:
    :param dep_ind:
    :return:
    '''
    row_ind_l = [row_ind - 1 if row_ind > 0 else row_ind]
    row_ind_u = [row_ind + 1 if row_ind < n2 - 1 else row_ind]
    col_ind_l = [col_ind - 1 if col_ind > 0 else col_ind]
    col_ind_u = [col_ind + 1 if col_ind < n1 - 1 else col_ind]
    dep_ind_l = [dep_ind - 1 if dep_ind > 0 else dep_ind]
    dep_ind_u = [dep_ind + 1 if dep_ind < n3 - 1 else dep_ind]

    row_ind_v = np.unique(np.vstack((row_ind_l, row_ind, row_ind_u)))
    col_ind_v = np.unique(np.vstack((col_ind_l, col_ind, col_ind_u)))
    dep_ind_v = np.unique(np.vstack((dep_ind_l, dep_ind, dep_ind_u)))

    row_ind, col_ind, dep_ind = np.meshgrid(row_ind_v, col_ind_v, dep_ind_v)
    return row_ind.reshape(-1, 1), col_ind.reshape(-1, 1), dep_ind.reshape(-1, 1)


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


def find_next_EIBV(row_neighbour, col_neighbour, dep_neighbour, row_now, col_now, dep_now, Sig, mu, tau, Thres):

    id = []
    for i in row_neighbour:
        for j in col_neighbour:
            for z in dep_neighbour:
                if i == row_now and j == col_now and z == dep_now:
                    continue
                id.append(np.ravel_multi_index((i, j, z), (n2, n1, n3)))
    id = np.unique(np.array(id))

    M = len(id)
    noise = tau ** 2
    R = np.diagflat(noise)  # diag not anymore support constructing matrix from vector

    eibv = []
    for k in range(M):
        F = np.zeros([1, N])
        F[0, id[k]] = True
        eibv.append(ExpectedVarianceUsr(Thres, mu, Sig, F, R))
    ind_desired = np.argmin(np.array(eibv))
    row_next, col_next, dep_next = np.unravel_index(id[ind_desired], (n2, n1, n3))

    return row_next, col_next, dep_next


def GPupd(mu, Sig, R, F, y_sampled):
    C = np.dot(F, np.dot(Sig, F.T)) + R
    mu_p = mu + np.dot(Sig, np.dot(F.T, np.linalg.solve(C, (y_sampled - np.dot(F, mu)))))
    Sigma_p = Sig - np.dot(Sig, np.dot(F.T, np.linalg.solve(C, np.dot(F, Sig))))
    return mu_p, Sigma_p


def plotf_depth_adaptive(val, string, i, path_col, path_row, vmin = None, vmax = None):
    val = val.reshape(n2, n1, n3)
    for j in range(n3):
        axes = fig.add_subplot(gs[i, j])
        im = axes.imshow(val[:, :, j], vmin=vmin, vmax=vmax);
        plt.title(string + " on layer " + str(j))
        plt.plot(path_col[j], path_row[j], 'r.-', linewidth=2)
        plt.xlabel("s1");
        plt.ylabel("s2");
        plt.colorbar(im, fraction=0.045, pad=0.04);



#% Method II: EIBV rule-based system supporting functions


def find_candidates_loc_rule(row_ind, col_ind, dep_ind):
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
    dep_ind_l = [dep_ind - 1 if dep_ind > 0 else dep_ind]
    dep_ind_u = [dep_ind + 1 if dep_ind < n3 - 1 else dep_ind]

    row_ind_v = np.unique(np.vstack((row_ind_l, row_ind, row_ind_u)))
    col_ind_v = np.unique(np.vstack((col_ind_l, col_ind, col_ind_u)))
    dep_ind_v = np.unique(np.vstack((dep_ind_l, dep_ind, dep_ind_u)))

    row_ind, col_ind, dep_ind = np.meshgrid(row_ind_v, col_ind_v, dep_ind_v)

    return row_ind.reshape(-1, 1), col_ind.reshape(-1, 1), dep_ind.reshape(-1, 1)


def find_next_EIBV_rule(row_cand, col_cand, dep_cand, row_now, col_now, dep_now,
                        row_pre, col_pre, dep_pre, Sig, mu, tau, Thres):

    id = []
    drow1 = row_now - row_pre
    dcol1 = col_now - col_pre
    ddep1 = dep_now - dep_pre
    vec1 = np.array([dcol1, drow1, ddep1])
    for i in row_cand:
        for j in col_cand:
            for z in dep_cand:
                if i == row_now and j == col_now and z == dep_now:
                    continue
                drow2 = i - row_now
                dcol2 = j - col_now
                ddep2 = z - dep_now
                vec2 = np.array([dcol2, drow2, ddep2])
                if np.dot(vec1, vec2) >= 0:
                    id.append(np.ravel_multi_index((i, j, z), (n2, n1, n3)))
                else:
                    continue
    id = np.unique(np.array(id))

    M = len(id)
    noise = tau ** 2
    R = np.diagflat(noise)  # diag not anymore support constructing matrix from vector

    eibv = []
    for k in range(M):
        F = np.zeros([1, N])
        F[0, id[k]] = True
        eibv.append(ExpectedVarianceUsr(Thres, mu, Sig, F, R))
    ind_desired = np.argmin(np.array(eibv))
    row_next, col_next, dep_next = np.unravel_index(id[ind_desired], (n2, n1, n3))

    return row_next, col_next, dep_next, id, eibv

