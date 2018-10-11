import meshio
import numpy as np
import h5py
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def gen_matrix(array=None):
    rect_1 = np.flip(array[0:XY[0]].reshape(Y[0], X[0]), 0)
    rect_2 = np.flip(array[XY[0]:XY[0] + XY[1]].reshape(Y[1], X[1]), 0)
    rect_3 = np.flip(array[XY[0] + XY[1]:XY[0] + XY[1] + XY[2]].reshape(Y[2], X[2]), 0)
    rect_4 = np.zeros((Y[0], X[2])) - 1.0

    con_1 = np.concatenate((rect_2, rect_3), axis=1)
    con_2 = np.concatenate((rect_1, rect_4), axis=1)

    final = np.concatenate((con_1, con_2), axis=0)
    final.astype(np.float32)

    return final


def extract_simulation_data(serial=None):
    data = {'Ux': [], 'Uy': [], 'p': [], 'T': []}
    for entity in ['p', 'T', 'U']:
        ims = []
        fig = plt.figure()
        for time in range(41):
            path = 'simulation_data/' + serial + '/VTK/' + serial + '_' + str(time * 250) + '.vtk'
            _, _, _, cell_data, _ = meshio.read(path)  # mesh.io to read the .vtk file.
            vtk_info = cell_data['hexahedron'][entity]
            if entity == 'U':
                vtk_info = np.delete(vtk_info, 2, 1)
                data['Ux'].append(gen_matrix(vtk_info[:, 0]))
                data['Uy'].append(gen_matrix(vtk_info[:, 1]))

            else:
                matrix = gen_matrix(vtk_info)
                extent = 0, 3, 0, 1
                im = plt.imshow(matrix, cmap=plt.cm.viridis, alpha=.9, interpolation='bilinear', extent=extent,
                                animated=True)
                ims.append([im])
                data[entity].append(matrix)

        if entity != 'U':  # Velocity animation was generated previously
            ani = animation.ArtistAnimation(fig, ims, interval=50)
            ani.save('simulation_data/' + serial + '/' + entity + '_simulation.mp4',
                     metadata={'Title': entity + '_' + serial, 'Artist': 'Afzal Hussain'})
            plt.close()

    return data


directory_list = next(os.walk('simulation_data'))[1]
total_sim = len(directory_list)
train_shape = (total_sim, 41, 50, 150, 4)

hdf5_path = 'dl_data/all_data.hdf5'
hdf5_file = h5py.File(hdf5_path, mode='w')
hdf5_file.create_dataset('sim_data', train_shape, np.float32)
hdf5_file.create_dataset('sim_no', (total_sim, 1), np.int16)

X, Y, XY = ([] for i in range(3))
count = 0

for sim_no in tqdm(directory_list):
    with open('simulation_data/' + sim_no + '/cellInformation') as file:
        for line in file:
            x, y = (int(i) for i in line.split())
            X.append(x), Y.append(y), XY.append(x * y)

    hdf5_file['sim_no'][count, 0] = int(sim_no)

    data = extract_simulation_data(sim_no)
    hdf5_file['sim_data'][count, ..., 0] = data['Ux']
    hdf5_file['sim_data'][count, ..., 1] = data['Uy']
    hdf5_file['sim_data'][count, ..., 2] = data['p']
    hdf5_file['sim_data'][count, ..., 3] = data['T']

    X.clear(), Y.clear(), XY.clear()
    count = count + 1

hdf5_file.close()
