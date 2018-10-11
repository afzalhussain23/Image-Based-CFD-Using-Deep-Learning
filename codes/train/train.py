from keras.callbacks import EarlyStopping, TensorBoard
from model.flow_net_inception import keras_model
import time
import h5py
import keras
import matplotlib.pyplot as plt


def read_dataset(path, split=0.8, print_shape=False):
    hdf5_file = h5py.File(path, "r")
    x = hdf5_file["sim_data"][:, 0, ...]
    y = hdf5_file["sim_data"][:, 1, ...]
    x = x.reshape(x.shape + (1,))
    y = y.reshape(y.shape + (1,))
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


start = time.time()

hdf5_path = 'all_U.hdf5'
(x_train, y_train, x_test, y_test, sim_no_train, sim_no_test) = read_dataset(hdf5_path, split=0.9, print_shape=False)

model = keras_model(x_train)
model.compile(loss=keras.losses.mean_squared_error,
              optimizer=keras.optimizers.Adam(lr=1e-4),
              metrics=['MSE'])

# Training parameters
batch_size = 4  # Initial value was 64
epochs = 2  # Initial value was 10

callbacks = [EarlyStopping(monitor='val_mean_squared_error', min_delta=0, patience=10), TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)]

# Train model
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=2,
          validation_data=(x_test, y_test),
          callbacks=callbacks)

# Evaluate on test set
score = model.evaluate(x_test, y_test, verbose=0)
print('Average Mean Squared Error:', score[0])

# Display predictions on test set
predicted_flow = model.predict(x_test, batch_size=batch_size)
predicted_flow = predicted_flow.reshape(predicted_flow.shape[:3])
y_test = y_test.reshape(y_test.shape[:3])
for i in range(15):
    extent = 0, 3, 0, 1
    #image = np.concatenate([predicted_flow[i], y_test[i]], axis=0)
    plt.suptitle('Comparision of OpenFOAM vs Deep Learning\nAverage Mean Squared Error: {0:0.5f}'.format(score[0]), fontsize=13)

    plt.subplot(211)
    plt.ylabel('OpenFOAM', fontsize=15)
    plt.imshow(y_test[i], cmap=plt.cm.viridis, alpha=.9, interpolation='bilinear', extent=extent)

    plt.subplot(212)
    plt.ylabel('Deep Learning', fontsize=15)
    plt.imshow(predicted_flow[i], cmap=plt.cm.viridis, alpha=.9, interpolation='bilinear', extent=extent)

    plt.subplots_adjust(left=0.2, wspace=0.8, top=0.85)
    plt.savefig('plots/' + str(i) + '.png')
    plt.close()

end = time.time()
print(end - start)

#sys.stdout.close()
