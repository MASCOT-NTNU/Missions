from usr_func import *

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