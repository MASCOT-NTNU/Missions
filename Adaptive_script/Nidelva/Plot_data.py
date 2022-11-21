import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
datapath = '/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Nidelva/SimulationData/'

shape_path = np.loadtxt(datapath + "shape_path.txt").astype(int)
shape_path_cand = np.loadtxt(datapath + "shape_path_cand.txt").astype(int)
shape_coords = np.loadtxt(datapath + "shape_coords.txt").astype(int)
shape_mu = np.loadtxt(datapath + "shape_mu.txt").astype(int)
shape_perr = np.loadtxt(datapath + "shape_perr.txt").astype(int)
shape_t_elapsed = np.loadtxt(datapath + "shape_t_elapsed.txt").astype(int)

data_path = np.loadtxt(datapath + "data_path.txt").reshape(shape_path).astype(int)
data_path_cand = np.loadtxt(datapath + "data_path_cand.txt").reshape(shape_path_cand).astype(int)
data_coords = np.loadtxt(datapath + "data_coords.txt").reshape(shape_coords)
data_mu = np.loadtxt(datapath + "data_mu.txt").reshape(shape_mu)
data_perr = np.loadtxt(datapath + "data_perr.txt").reshape(shape_perr)
data_t_elapsed = np.loadtxt(datapath + "data_t_elapsed.txt").reshape(shape_t_elapsed)

figpath = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Nidelva/Adaptive/"

N1 = 25
N2 = 25
N3 = 5
depth_obs = [0.5, 1.0, 1.5, 2.0, 2.5]


#%%
path_x = []
path_y = []
for k in range(N3):
    path_x.append([])
    path_y.append([])

for j in range(data_mu.shape[0]):
    fig = plt.figure(figsize=(35, 5))
    gs = GridSpec(nrows=1, ncols=5)
    for i in range(N3):
        if i == data_path[j, -1]:
            path_x[i].append(data_path[j, 0])
            path_y[i].append(data_path[j, 1])
        else:
            path_x[i].append(np.nan)
            path_y[i].append(np.nan)

        ax = fig.add_subplot(gs[i])
        im = ax.imshow(data_mu[j, :, :].reshape(N3, N1, N2)[i, :, :])
        plt.plot(path_x[i], path_y[i], 'r-')
        ax.set(title = "Posterior mean at depth {:.1f} meter at time {:02d}".format(depth_obs[i], j))
        plt.colorbar(im)
    fig.savefig(figpath + "Mean/Mean_{:03d}.pdf".format(j))
    print(j)
    # plt.show()

path_x = []
path_y = []
for k in range(N3):
    path_x.append([])
    path_y.append([])


for j in range(data_perr.shape[0]):
    fig = plt.figure(figsize=(35, 5))
    gs = GridSpec(nrows=1, ncols=5)
    for i in range(5):
        if i == data_path[j, -1]:
            path_x[i].append(data_path[j, 0])
            path_y[i].append(data_path[j, 1])
        else:
            path_x[i].append(np.nan)
            path_y[i].append(np.nan)

        ax = fig.add_subplot(gs[i])
        im = ax.imshow(np.sqrt(data_perr[j, :, :].reshape(N3, N1, N2)[i, :, :]))
        plt.plot(path_x[i], path_y[i], 'r-')
        ax.set(title = "Prediction error at depth {:.1f} meter at time {:02d}".format(depth_obs[i], j))
        plt.colorbar(im)
    fig.savefig(figpath + "Perr/Perr_{:03d}.pdf".format(j))
    # plt.show()
#     print(j)






