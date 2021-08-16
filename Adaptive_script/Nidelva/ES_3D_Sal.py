from Prior import *

today = date.today()
d1 = today.strftime("%d_%m_%Y")
datafolder = os.getcwd() + "/" + d1
datapath = datafolder + "/Data/"

beta0 = np.loadtxt(datapath + 'beta0.txt', delimiter = ",")
beta1 = np.loadtxt(datapath + 'beta1.txt', delimiter = ",")
mu_prior_sal = np.loadtxt(datapath + 'mu_prior_sal.txt', delimiter = ",")
mu_prior_temp = np.loadtxt(datapath + 'mu_prior_temp.txt', delimiter = ",")
print("Congrats!!! Prior is built successfully!!!")
print("Fitted beta0: \n", beta0)
print("Fitted beta1: \n", beta1)

#%% Section I: Setup parameters
sigma_sal = np.sqrt(4)  # scaling coef in matern kernel for salinity
tau_sal = np.sqrt(.3) # iid noise
Threshold_S = 23 # 20

sigma_temp = np.sqrt(0.5)
tau_temp = np.sqrt(.1)
Threshold_T = 10.5

eta = 4.5 / 400 # coef in matern kernel
ksi = 1000 / 24 / 0.5 # scaling factor in 3D
N_steps = 60 # number of steps desired to conduct

#%% Section II: Set up the waypoint and grid
nx = 25 # number of grid points along x-direction
ny = 25 # number of grid points along y-direction
L = 1000 # distance of the square
alpha = -60 # angle of the inclined grid
distance = L / (nx - 1)
distance_depth = depth_obs[1] - depth_obs[0]
gridx, gridy = rotateXY(nx, ny, distance, alpha)
grid = []
for k in depth_obs:
    for i in range(gridx.shape[0]):
        for j in range(gridx.shape[1]):
                grid.append([gridx[i, j], gridy[i, j], k])
grid = np.array(grid)

H = compute_H(grid, ksi)

Sigma_prior = Matern_cov(sigma_sal, eta, H)

EP_prior = EP_1D(mu_prior_sal, Sigma_prior, Threshold_S)
#%% #%% Part II : Path planning

path = []
path_cand = []
coords = []
mu = []
Sigma = []
t_elapsed = []

loc = find_starting_loc(EP_prior, N1, N2, N3)
xstart, ystart, zstart = loc

xnow, ynow, znow = xstart, ystart, zstart
xpre, ypre, zpre = xnow, ynow, znow

lat_start, lon_start = xy2latlon(xstart, ystart, origin, distance, alpha)
path.append([xnow, ynow, znow])
coords.append([lat_start, lon_start])

print("The starting location is [{:.2f}, {:.2f}]".format(lat_start, lon_start))

mu_cond = mu_prior_sal
Sigma_cond = Sigma_prior
mu.append(mu_cond)
Sigma.append(Sigma_cond)

noise = tau_sal ** 2
R = np.diagflat(noise)

for j in range(N_steps):
    xcand, ycand, zcand = find_candidates_loc(xnow, ynow, znow, N1, N2, N3)

    t1 = time.time()
    xnext, ynext, znext = find_next_EIBV_1D(xcand, ycand, zcand,
                                            xnow, ynow, znow,
                                            xpre, ypre, zpre,
                                            N1, N2, N3, Sigma_cond,
                                            mu_cond, tau_sal, Threshold_S)
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

    # sal_sampled = F @ TEST_SAL
    sal_sampled = 0
    mu_cond, Sigma_cond = GPupd(mu_cond, Sigma_cond, R, F, sal_sampled)

    xpre, ypre, zpre = xnow, ynow, znow
    xnow, ynow, znow = xnext, ynext, znext

    path.append([xnow, ynow, znow])
    path_cand.append([xcand, ycand, zcand])
    coords.append([lat_next, lon_next])
    mu.append(mu_cond)
    Sigma.append(Sigma_cond)

#%% Save data section
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

