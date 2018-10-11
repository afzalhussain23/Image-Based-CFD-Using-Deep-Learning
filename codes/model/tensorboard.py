from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.callbacks import EarlyStopping, TensorBoard
import h5py
from keras.models import Model
from keras.layers import add, Input, Cropping2D, Conv2D, Conv3D, MaxPooling2D, Conv2DTranspose, \
    TimeDistributed
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.utils.vis_utils import plot_model


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


def lstm_model():
    inputs = Input(shape=(40, 50, 150, 4))

    c1 = TimeDistributed(Conv2D(32, (2, 2), activation='relu', padding='same'))(inputs)  # (None, 50, 150, 32)
    x = TimeDistributed(MaxPooling2D((2, 2), padding='same'))(c1)  # (None, 25, 75, 32)

    c2 = TimeDistributed(Conv2D(64, (2, 2), activation='relu', padding='same'))(x)  # (None, 25, 75, 64)
    x = TimeDistributed(MaxPooling2D((2, 2), padding='same'))(c2)  # (None, 13, 38, 64)

    c3 = TimeDistributed(Conv2D(128, (2, 2), activation='relu', padding='same'))(x)  # (None, 13, 38, 128)
    x = TimeDistributed(MaxPooling2D((2, 2), padding='same'))(c3)  # (None, 7, 19, 128)

    x = ConvLSTM2D(filters=256, kernel_size=(3, 3), activation='relu',
                   recurrent_activation='hard_sigmoid', padding='same', return_sequences=True)(x)

    x = TimeDistributed(Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same'))(x)
    x = TimeDistributed(Cropping2D(cropping=((1, 0), (0, 0))))(x)
    x = add([x, c3])

    x = ConvLSTM2D(filters=128, kernel_size=(3, 3), activation='relu',
                   recurrent_activation='hard_sigmoid', padding='same', return_sequences=True)(x)

    x = TimeDistributed(Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same'))(x)
    x = TimeDistributed(Cropping2D(cropping=((1, 0), (1, 0))))(x)
    x = add([x, c2])

    x = ConvLSTM2D(filters=64, kernel_size=(3, 3), activation='relu',
                   recurrent_activation='hard_sigmoid', padding='same', return_sequences=True)(x)

    x = TimeDistributed(Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same'))(x)
    x = add([x, c1])

    x = Conv3D(filters=4, kernel_size=(3, 3, 3), activation='linear', padding='same', data_format='channels_last')(x)

    model = Model(inputs=inputs, outputs=x)
    plot_model(model, to_file='cnn_model.png', show_shapes=True, show_layer_names=False)

    return model


hdf5_path = 'all_data.hdf5'

x_train, y_train, x_test, y_test, _, _ = read_dataset(path=hdf5_path, split=0.9, print_shape=False)

model = lstm_model()
model.compile(loss='mean_squared_error', optimizer='Adam', metrics=['MSE'])

callbacks = [EarlyStopping(monitor='val_mean_squared_error', min_delta=0, patience=10), TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)]

model.fit(x=x_train, y=y_train, batch_size=1, epochs=1, validation_data=(x_test, y_test), callbacks=callbacks)

score = model.evaluate(x=x_test, y=y_test, verbose=0)

print('Average Mean Squared Error:', score[0])