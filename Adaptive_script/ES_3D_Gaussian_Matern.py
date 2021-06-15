from Prior import *

#%% Section I: Build the waypoint and grid
sigma = np.sqrt(30)  # scaling coef in matern kernel
eta = np.sqrt(3) / 500 # coef in matern kernel
tau = .3 # iid noise
Threshold_S = 23 # 20
Threshold_T = 10.5
nx = 25
ny = 25
L = 1000
alpha = -60
distance = L / (nx - 1)
distance_depth = depth_obs[1] - depth_obs[0]
gridx, gridy = rotateXY(nx, ny, distance, alpha)
grid = []
for k in depth_obs:
    for i in range(gridx.shape[0]):
        for j in range(gridx.shape[1]):
                grid.append([gridx[i, j], gridy[i, j], k])
grid = np.array(grid)
t = scdist.cdist(grid, grid) # distance matrix for the whole grid
Sigma_prior = Matern_cov(sigma, eta, t)

def EP_1D(mu, Sigma, Threshold):
    EP_Prior = np.zeros_like(mu)
    for i in range(EP_Prior.shape[0]):
        EP_Prior[i] = norm.cdf(Threshold, mu[i], Sigma[i, i])
    return EP_Prior

EP_prior = EP_1D(mu_prior_sal, Sigma_prior, Threshold_S)

mup = EP_prior.reshape(N3, N1, N2)
fig = plt.figure(figsize=(35, 5))
gs = GridSpec(nrows = 1, ncols = 5)
for i in range(len(depth_obs)):
    ax = fig.add_subplot(gs[i])
    im = ax.imshow(np.rot90(mup[i, :, :]), vmin = 0.45, vmax = .7, extent = (0, 1000, 0, 1000))
    ax.set(title = "Prior excursion probabilities at depth {:.1f} meter".format(depth_obs[i]))
    plt.colorbar(im)
fig.savefig(figpath + "EP_Prior.pdf")
plt.show()


loc = find_starting_loc(EP_prior, N1, N2, N3)
xstart, ystart, zstart = loc


#%% #%% Part II : Path planning
N_steps = 10

path = []
path_cand = []
coords = []
mu = []
Sigma = []
t_elapsed = []

xnow, ynow, znow = xstart, ystart, zstart
xpre, ypre, zpre = xnow, ynow, znow

lat_start, lon_start = xy2latlon(xstart, ystart, origin, distance, alpha)
path.append([xnow, ynow, znow])
coords.append([lat_start, lon_start])

print("The starting point is ")
print(xnow, ynow, znow)

mu_cond = mu_prior_sal
Sigma_cond = Sigma_prior
mu.append(mu_cond)
Sigma.append(Sigma_cond)

noise = tau ** 2
R = np.diagflat(noise)

for j in range(N_steps):
# for j in range(1):
    xcand, ycand, zcand = find_candidates_loc(xnow, ynow, znow, N1, N2, N3)

    t1 = time.time()
    xnext, ynext, znext = find_next_EIBV(xcand, ycand, zcand,
                                         xnow, ynow, znow,
                                         xpre, ypre, zpre,
                                         N1, N2, N3, Sigma_cond, mu_cond, tau, Threshold_S)
    t2 = time.time()
    t_elapsed.append(t2 - t1)
    print("It takes {:.2f} seconds to compute the next waypoint".format(t2 - t1))
    print("next is ", xnext, ynext, znext)
    lat_next, lon_next = xy2latlon(xnext, ynext, origin, distance, alpha)

    # ====
    '''
    Here comes the action part, move to the next waypoint and sample
    
    - move to [lat_next, lon_next]

    - sal_sampled = obtain salinity from CTD sensor (gather around at that specific point)

    '''
    # ====
    ind_next = ravel_index([xnext, ynext, znext], N1, N2, N3)
    F = np.zeros([1, N])
    F[0, ind_next] = True

    sal_sampled = F @ TEST_SAL
    mu_cond, Sigma_cond = GPupd(mu_cond, Sigma_cond, R, F, sal_sampled)

    xpre, ypre, zpre = xnow, ynow, znow
    xnow, ynow, znow = xnext, ynext, znext

    path.append([xnow, ynow, znow])
    path_cand.append([xcand, ycand, zcand])
    coords.append([lat_next, lon_next])
    mu.append(mu_cond)
    Sigma.append(Sigma_cond)

    print("Step NO.", str(j), ". The current ind is ", xnow, ynow, znow)


#%%
datapath = '/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Nidelva/SimulationData/'
data_path = np.array(path)
data_path_cand = np.array(path_cand)
data_coords = np.array(coords)
data_mu = np.array(mu)
data_Sigma = np.array(Sigma)
data_perr = np.zeros_like(data_mu)
for i in range(data_Sigma.shape[0]):
    data_perr[i, :, :] = np.diag(data_Sigma[i, :, :]).reshape(-1, 1)
data_t_elapsed = np.array(t_elapsed)

shape_path = data_path.shape
shape_path_cand = data_path_cand.shape
shape_coords = data_coords.shape
shape_mu = data_mu.shape
shape_perr = data_perr.shape
shape_t_elapsed = data_t_elapsed.shape

np.savetxt(datapath + "shape_path.txt", shape_path, delimiter=", ")
np.savetxt(datapath + "shape_path_cand.txt", shape_path_cand, delimiter=", ")
np.savetxt(datapath + "shape_coords.txt", shape_coords, delimiter=", ")
np.savetxt(datapath + "shape_mu.txt", shape_mu, delimiter=", ")
np.savetxt(datapath + "shape_perr.txt", shape_perr, delimiter=", ")
np.savetxt(datapath + "shape_t_elapsed.txt", shape_t_elapsed, delimiter=", ")

np.savetxt(datapath + "data_path.txt", data_path.reshape(-1, 1), delimiter=", ")
np.savetxt(datapath + "data_path_cand.txt", data_path_cand.reshape(-1, 1), delimiter=", ")
np.savetxt(datapath + "data_coords.txt", data_coords.reshape(-1, 1), delimiter=", ")
np.savetxt(datapath + "data_mu.txt", data_mu.reshape(-1, 1), delimiter=", ")
np.savetxt(datapath + "data_perr.txt", data_perr.reshape(-1, 1), delimiter=", ")
np.savetxt(datapath + "data_t_elapsed.txt", data_t_elapsed.reshape(-1, 1), delimiter=", ")

# np.savetxt(datapath + "path.txt", path, delimiter=',')
# np.savetxt(datapath + "mu.txt", mu, delimiter=",")

