from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.callbacks import EarlyStopping
import h5py


def read_dataset(path=None, split=0.8, print_shape=False):
    hdf5_file = h5py.File(path, "r")
    x = hdf5_file["sim_data"][:10, 0:20, ...]
    y = hdf5_file["sim_data"][:10, 1:21, ...]
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


def keras_model(layers=3, filters=32, kernel_size=(3, 3), activation='relu', recurrent_activation='tanh', dropout=0.0,
                recurrent_dropout=0.0):
    input_shape = (None, 50, 150, 4)

    seq = Sequential()
    if layers >= 1:
        seq.add(ConvLSTM2D(filters=filters, kernel_size=kernel_size, activation=activation, input_shape=input_shape,
                           recurrent_activation=recurrent_activation, padding='same', dropout=dropout,
                           recurrent_dropout=recurrent_dropout, return_sequences=True))
    if layers >= 2:
        seq.add(ConvLSTM2D(filters=filters, kernel_size=kernel_size, activation=activation,
                           recurrent_activation=recurrent_activation, padding='same', dropout=dropout,
                           recurrent_dropout=recurrent_dropout, return_sequences=True))
    if layers >= 3:
        seq.add(ConvLSTM2D(filters=filters, kernel_size=kernel_size, activation=activation,
                           recurrent_activation=recurrent_activation, padding='same', dropout=dropout,
                           recurrent_dropout=recurrent_dropout, return_sequences=True))
    if layers >= 4:
        seq.add(ConvLSTM2D(filters=filters, kernel_size=kernel_size, activation=activation,
                           recurrent_activation=recurrent_activation, padding='same', dropout=dropout,
                           recurrent_dropout=recurrent_dropout, return_sequences=True))
    if layers >= 5:
        seq.add(ConvLSTM2D(filters=filters, kernel_size=kernel_size, activation=activation,
                           recurrent_activation=recurrent_activation, padding='same', dropout=dropout,
                           recurrent_dropout=recurrent_dropout, return_sequences=True))
    if layers >= 6:
        seq.add(ConvLSTM2D(filters=filters, kernel_size=kernel_size, activation=activation,
                           recurrent_activation=recurrent_activation, padding='same', dropout=dropout,
                           recurrent_dropout=recurrent_dropout, return_sequences=True))
    if layers >= 7:
        seq.add(ConvLSTM2D(filters=filters, kernel_size=kernel_size, activation=activation,
                           recurrent_activation=recurrent_activation, padding='same', dropout=dropout,
                           recurrent_dropout=recurrent_dropout, return_sequences=True))
    if layers >= 8:
        seq.add(ConvLSTM2D(filters=filters, kernel_size=kernel_size, activation=activation,
                           recurrent_activation=recurrent_activation, padding='same', dropout=dropout,
                           recurrent_dropout=recurrent_dropout, return_sequences=True))
    if layers >= 9:
        seq.add(ConvLSTM2D(filters=filters, kernel_size=kernel_size, activation=activation,
                           recurrent_activation=recurrent_activation, padding='same', dropout=dropout,
                           recurrent_dropout=recurrent_dropout, return_sequences=True))

    seq.add(Conv3D(filters=4, kernel_size=(3, 3, 3), activation='linear', padding='same', data_format='channels_last'))

    return seq


hdf5_path = 'all_data.hdf5'

x_train, y_train, x_test, y_test, _, _ = read_dataset(path=hdf5_path, split=0.9, print_shape=False)

model = keras_model(layers=3, filters=32, kernel_size=(3, 3), activation='relu', recurrent_activation='tanh',
                    dropout=0.0, recurrent_dropout=0.0)

model.compile(loss='mean_squared_error', optimizer='Adam', metrics=['MSE'])

callbacks = [EarlyStopping(monitor='val_mean_squared_error', min_delta=0, patience=10)]

model.fit(x=x_train, y=y_train, batch_size=1, epochs=1, validation_data=(x_test, y_test), callbacks=callbacks)

score = model.evaluate(x=x_test, y=y_test, verbose=0)

print('Average Mean Squared Error:', score[0])
