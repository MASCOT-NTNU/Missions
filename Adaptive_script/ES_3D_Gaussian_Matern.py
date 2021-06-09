import numpy as np

from Prior import *


#%% Section I: Build the waypoint and grid
sigma = np.sqrt(30)  # scaling coef in matern kernel
eta = np.sqrt(3) / 900 # coef in matern kernel
tau = .01 # iid noise

nx = 25
ny = 25
L = 1000
distance = L / (nx - 1)
gridx, gridy = rotateXY(nx, ny, distance, 60)
grid = []
for k in depth_obs:
    for i in range(gridx.shape[0]):
        for j in range(gridx.shape[1]):
                grid.append([gridx[i, j], gridy[i, j], k])
grid = np.array(grid)


#%% Section II: Update the random field
t = scdist.cdist(grid, grid) # distance matrix for the whole grid

# def getMean(beta0_sal, beta1_sal, beta0_temp, beta1_temp, grid):
beta0_sal = np.kron(beta0_sal, np.ones([N1 * N2, 1]))
beta1_sal = np.kron(beta1_sal, np.ones([N1 * N2, 1]))
beta0_temp = np.kron(beta0_temp, np.ones([N1 * N2, 1]))
beta1_temp = np.kron(beta1_temp, np.ones([N1 * N2, 1]))
mu_sal = np.zeros([grid.shape[0], 1])
mu_temp = np.zeros([grid.shape[0], 1])
#%%
for i in range(grid.shape[0]):
    mu_sal[i] = beta0_sal[i] + grid[i, 0:2] @ beta1_sal[i, :]
    mu_temp[i] = beta1_temp[i] + grid[i, 0:2] @ beta1_temp[i, :]

    # return mu_sal, mu_temp

# a, b = getMean(beta0_sal, beta1_sal, beta0_temp, beta1_temp, grid)


#%% Part II : Path planning
N_steps = 50

north_now, east_now, depth_now = north_start, east_start, depth_start
north_pre, east_pre, depth_pre = north_now, east_now, depth_now

print("The starting point is ")
print(north_now, east_now, depth_now)

mu_cond = mu_prior
Sigma_cond = Sigma
noise = tau ** 2
R = np.diagflat(noise)

path_north = []
path_east = []
path_depth = []

for j in range(N_steps):

    north_cand, east_cand, depth_cand = find_candidates_loc(north_now, east_now, depth_now, N1, N2, N3)

    north_next, east_next, depth_next = find_next_EIBV(north_cand, east_cand, depth_cand,
                                                       north_now, east_now, depth_now,
                                                       north_pre, east_pre, depth_pre,
                                                       N1, N2, N3, Sigma_cond, mu_cond, tau, S_thres)

    ind_next = np.ravel_multi_index((north_next, east_next, depth_next), (N1, N2, N3))
    F = np.zeros([1, N])
    F[0, ind_next] = True

    # ====
    '''
    Here comes the action part, move to the next waypoint and sample
    
    
    
    '''
    # ====
    # y_sampled =

    mu_cond, Sigma_cond = GPupd(mu_cond, Sigma_cond, R, F, y_sampled)
    
    path_north.append(north_now)
    path_east.append(east_now)
    path_depth.append(depth_now)
    north_pre, east_pre, depth_pre = north_now, east_now, depth_now
    north_now, east_now, depth_now = north_next, east_next, depth_next
    print("Step NO.", str(j), ". The current ind is ", north_now, east_now, depth_now)


