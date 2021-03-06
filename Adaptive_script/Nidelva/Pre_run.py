from usr_func import *
lat4, lon4 = 63.446905, 10.419426 # right bottom corner
origin = [lat4, lon4]
distance = 1000
depth_obs = [0.5, 1.0, 1.5, 2.0, 2.5] # planned depth to be observed
box = BBox(lat4, lon4, distance, 60)

N1 = 25 # number of grid points along north direction
N2 = 25 # number of grid points along east direction
N3 = 5 # number of layers in the depth dimension
N = N1 * N2 * N3 # total number of grid points

XLIM = [0, 1000]
YLIM = [0, 1000]
ZLIM = [0.5, 2.5]
x = np.linspace(XLIM[0], XLIM[1], N1)
y = np.linspace(YLIM[0], YLIM[1], N2)
z = np.array([0.5, 1.0, 1.5, 2.0, 2.5]).reshape(-1, 1)
xm, ym, zm = np.meshgrid(x, y, z)
xv = xm.reshape(-1, 1) # sites1v is the vectorised version
yv = ym.reshape(-1, 1)
zv = zm.reshape(-1, 1)
dx = x[1] - x[0]
dy = y[1] - y[0]
dz = z[1] - z[0]

mu_prior_sal = []
mu_prior_temp = []
coordinates= getCoordinates(box, N1, N2, dx, 60)

F_odd = np.zeros([6, coordinates.shape[0]])
F_even = np.zeros([6, coordinates.shape[0]])
cnt = 0
cnt_i = 0
for i in [4, 12, 20]:
    for j in [0, N1 - 1]:
        if cnt_i % 2 == 0:
            ind_odd = ravel_index([j, i, 0], N1, N2, N3)
            ind_even = ravel_index([N1 - j - 1, 20 - 8 * cnt_i, 0], N1, N2, N3)
        else:
            ind_odd = ravel_index([N1 - j - 1, i, 0], N1, N2, N3)
            ind_even = ravel_index([j, 20 - 8 * cnt_i, 0], N1, N2, N3)

        F_odd[cnt, ind_odd] = True
        F_even[cnt, ind_even] = True
        cnt = cnt + 1
    cnt_i = cnt_i + 1


#%% Pre run section
for i in range(len(depth_obs)):
    if i % 2 == 0:
        for j in range(F_even.shape[0]):
            loc = F_even[j, :] @ coordinates
            '''
            lat = loc[0]
            lon = loc[1]
            depth = depth_obs[i]
            move (lat, lon, depth)
            '''
    else:
        for j in range(F_odd.shape[0]):
            loc = F_odd[j, :] @ coordinates
            '''
            lat = loc[0]
            lon = loc[1]
            depth = depth_obs[i]
            move (lat, lon, depth)
            '''

'''
Export data to local file named "data.txt" for the later use
'''


figpath = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/Nidelva/Path/"

# for i in range(F_odd.shape[0]):
#     '''
#     Move to loc = F[i, :] @ coordinates
#     '''
    # plt.figure(figsize=(5, 5))
    # plt.plot(coordinates[:, 1], coordinates[:, 0], 'k.')
    # loc = F_odd[i, :] @ coordinates
    # plt.plot(loc[1], loc[0], 'r.')
    # plt.savefig(figpath + "P_{:03d}.pdf".format(i))
    # plt.close("all")


# for i in range(F_even.shape[0]):
#     '''
#     Move to loc = F[i, :] @ coordinates
#     '''
    # plt.figure(figsize=(5, 5))
    # plt.plot(coordinates[:, 1], coordinates[:, 0], 'k.')
    # loc = F_even[i, :] @ coordinates
    # plt.plot(loc[1], loc[0], 'r.')
    # plt.savefig(figpath + "PE_{:03d}.pdf".format(i))
    # plt.close("all")

'''
Surfacing in between
'''




