from keras.models import load_model
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def read_dataset(path=None, split=0.8, print_shape=False):
    hdf5_file = h5py.File(path, "r")
    x = hdf5_file["sim_data"][:10, 0:40, ...]
    y = hdf5_file["sim_data"][:10, 1:41, ...]
    sim_no = hdf5_file["sim_no"][:, 0]
    hdf5_file.close()

    total_sim = x.shape[0]
    x_train = x[:int(total_sim * split), ...]
    y_train = y[:int(total_sim * split), ...]
    x_test = x[int(total_sim * split):total_sim, ...]
    y_test = y[int(total_sim * split):total_sim, ...]
    sim_no_train = sim_no[:int(total_sim * split), ...]
    sim_no_test = sim_no[int(total_sim * split):total_sim, ...]

    if print_shape:
        print("total_sim: {}\nx_train.shape: {}\ny_train.shape: {}\nx_test.shape: {}\ny_test.shape: {}\n".format(
            total_sim,
            x_train.shape,
            y_train.shape,
            x_test.shape,
            y_test.shape))

    return x_train, y_train, x_test, y_test, sim_no_train, sim_no_test


hdf5_path = 'all_data.hdf5'
x_train, y_train, x_test, y_test, _, _ = read_dataset(path=hdf5_path, split=0.9, print_shape=True)
model = load_model('my_model.h5')


which = 0
track = x_test[which, ...]

for j in range(40):
    new_pos = model.predict(track[np.newaxis, 0:40, ::, ::, ::])
    new = new_pos[::, -1, ::, ::, ::]
    track = np.concatenate((track, new), axis=0)



class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# And then compare the predictions
# to the ground truth
track2 = x_test[which][::, ::, ::, ::]
for i in range(40):
    fig = plt.figure(figsize=(10, 5))

    ax = fig.add_subplot(121)

    if i >= 7:
        ax.text(1, 3, 'Predictions !', fontsize=20, color='w')
    else:
        ax.text(1, 3, 'Initial trajectory', fontsize=20)

    toplot = track[i, ::, ::, 0]

    plt.imshow(toplot)
    ax = fig.add_subplot(122)
    plt.text(1, 3, 'Ground truth', fontsize=20)

    toplot = track2[i, ::, ::, 0]
    if i >= 2:
        toplot = y_test[which][i - 1, ::, ::, 0]

    extent = 0, 3, 0, 1
    im = plt.imshow(toplot, cmap='RdBu_r', alpha=.9, interpolation='bilinear', norm=MidpointNormalize(midpoint=0.), extent=extent)
    plt.colorbar();
    plt.savefig('%i_animate.png' % (i + 1))