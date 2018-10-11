import h5py
import numpy as np
x = []
y = []
hdf5_file = h5py.File('all_data.hdf5', "r")
data = hdf5_file["sim_data"][:, ...]
hdf5_file.close()
print(data.shape)
for i in range(data.shape[0]):
    for j in range(data.shape[1]-1):
        x.append(data[i, j, ...])
        y.append(data[i, j+1, ...])

train_shape = (len(x), 50, 150, 4)

hdf5_path = 'unified_data.hdf5'
hdf5_file = h5py.File(hdf5_path, mode='w')
hdf5_file.create_dataset('x', train_shape, np.float32)
hdf5_file.create_dataset('y', train_shape, np.float32)

hdf5_file['x'] = np.asarray(x)
hdf5_file['y'] = np.asarray(y)

hdf5_file.close()

