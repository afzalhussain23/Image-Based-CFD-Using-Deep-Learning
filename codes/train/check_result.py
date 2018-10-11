from keras.models import load_model
import h5py
import numpy as np
import matplotlib.pyplot as plt
model = load_model('my_model_12.h5')


def read_dataset(print_shape=True):
    x = []
    y = []
    hdf5_file = h5py.File('all_data.hdf5', "r")
    data = hdf5_file["sim_data"][:, ...]
    hdf5_file.close()
    print(data.shape)
    for i in range(10):
        for j in range(5):
            x.append(data[i, j, ...])
            y.append(data[i, j + 1, ...])

    x = np.asarray(x)
    y = np.asarray(y)

    x_test = x
    y_test = y

    if print_shape:
        print("x_test.shape: {}\ny_test.shape: {}\n".format(
            x_test.shape,
            y_test.shape))

    return x_test, y_test


x_test, y_test = read_dataset()

predicted_flow = model.predict(x_test, batch_size=4)

predicted_flow = predicted_flow.reshape(predicted_flow.shape[:3])
y_test = y_test.reshape(y_test.shape[:3])
for i in range(15):
    extent = 0, 3, 0, 1
    plt.suptitle('Comparision of OpenFOAM vs Deep Learning', fontsize=13)

    plt.subplot(211)
    plt.ylabel('OpenFOAM', fontsize=15)
    plt.imshow(y_test[i], cmap=plt.cm.viridis, alpha=.9, interpolation='bilinear', extent=extent)

    plt.subplot(212)
    plt.ylabel('Deep Learning', fontsize=15)
    plt.imshow(predicted_flow[i], cmap=plt.cm.viridis, alpha=.9, interpolation='bilinear', extent=extent)

    plt.subplots_adjust(left=0.2, wspace=0.8, top=0.85)
    plt.savefig('plots/' + str(i) + '.png')
    plt.close()