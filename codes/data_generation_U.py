import meshio
import numpy as np
import h5py
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def all_U(serial):
    sim_matrix = []
    ims = []
    fig = plt.figure()
    for i in range(41):
        path = "simulation_data/" + serial + "/VTK/" + serial + "_" + str(i * 250) + ".vtk"
        _, _, _, cell_data, _ = meshio.read(path)  # mesh.io to read the .vtk file.

        result3d = cell_data["hexahedron"]["U"]  # Contain information of velocity in 3 direction
        result2d = np.delete(result3d, 2, 1)  # Delete z axis. All values are 0.
        magnitude = np.linalg.norm(result2d, axis=1)  # Calculate the magnitude

        X = []
        Y = []
        XY = []
        with open("simulation_data/" + serial + '/cellInformation') as file:
            for line in file:
                x, y = (int(i) for i in line.split())
                X.append(x), Y.append(y), XY.append(x * y)

        rect_1 = np.flip(magnitude[0:XY[0]].reshape(Y[0], X[0]), 0)
        rect_2 = np.flip(magnitude[XY[0]:XY[0] + XY[1]].reshape(Y[1], X[1]), 0)
        rect_3 = np.flip(magnitude[XY[0] + XY[1]:XY[0] + XY[1] + XY[2]].reshape(Y[2], X[2]), 0)
        rect_4 = np.zeros((Y[0], X[2])) - 1

        con_1 = np.concatenate((rect_2, rect_3), axis=1)
        con_2 = np.concatenate((rect_1, rect_4), axis=1)

        final = np.concatenate((con_1, con_2), axis=0)
        final.astype(np.float32)
        sim_matrix.append(final)

        extent = 0, 3, 0, 1
        im = plt.imshow(final, cmap=plt.cm.viridis, alpha=.9, interpolation='bilinear', extent=extent, animated=True)
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=50)
    ani.save("simulation_data/" + serial + "/velocity_simulation.mp4", metadata={'Title': 'U_' + serial, 'Artist': "Afzal Hussain"})
    plt.close()

    return sim_matrix


directory_list = next(os.walk('simulation_data'))[1]
total_sim = len(directory_list)

hdf5_path = 'dl_data/all_U.hdf5'
train_shape = (total_sim, 41, 50, 150)
hdf5_file = h5py.File(hdf5_path, mode='w')

hdf5_file.create_dataset("sim_data", train_shape, np.float32)
hdf5_file.create_dataset("sim_no", (total_sim, 1), np.int16)

count = 0

for sim_no in tqdm(directory_list):
    sim_matrix = all_U(sim_no)
    hdf5_file["sim_data"][count, ...] = sim_matrix
    hdf5_file["sim_no"][count, 0] = int(sim_no)
    count = count + 1

hdf5_file.close()
