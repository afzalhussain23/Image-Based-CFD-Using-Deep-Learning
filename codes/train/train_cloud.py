from keras.models import Model
from keras.layers.convolutional import ZeroPadding2D
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.utils.vis_utils import plot_model
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import h5py


def read_dataset(path, split=0.8, print_shape=False):
    hdf5_file = h5py.File(path, "r")
    x = hdf5_file['x']
    y = hdf5_file['y']
    hdf5_file.close()

    total_sim = x.shape[0]
    x_train = x[:int(total_sim * split), ...]
    y_train = y[:int(total_sim * split), ...]
    x_test = x[int(total_sim * split):total_sim, ...]
    y_test = y[int(total_sim * split):total_sim, ...]

    if print_shape:
        print("total_sim: {}\nx_train.shape: {}\ny_train.shape: {}\nx_test.shape: {}\ny_test.shape: {}\n".format(
            total_sim,
            x_train.shape,
            y_train.shape,
            x_test.shape,
            y_test.shape))

    return x_train, y_train, x_test, y_test


def inception_module(filters=None, inputs=None):
    tower_0 = Conv2D(int(filters / 4), (1, 1), padding='same', activation='relu')(inputs)

    tower_1 = Conv2D(int(filters / 4), (1, 1), padding='same', activation='relu')(inputs)
    tower_1 = Conv2D(int((filters * 3) / 8), (3, 3), padding='same', activation='relu')(tower_1)

    tower_2 = Conv2D(int(filters / 8), (1, 1), padding='same', activation='relu')(inputs)
    tower_2 = Conv2D(int(filters / 8), (5, 5), padding='same', activation='relu')(tower_2)

    tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(inputs)
    tower_3 = Conv2D(int(filters / 4), (1, 1), padding='same', activation='relu')(tower_3)

    concat = concatenate([tower_0, tower_1, tower_2, tower_3], axis=3)

    return concat


def keras_model(save_model=True):
    inputs = Input((50, 150, 4))  # Input(shape), return a tensor
    # Initial shape: (None, 50, 150, 4)

    x = Conv2D(16, (3, 3), padding='same', activation='relu')(inputs)  # (None, 50, 150, 16)
    x = Conv2D(16, (3, 3), padding='same', activation='relu')(x)  # (None, 50, 150, 16)

    c1 = inception_module(filters=32, inputs=x)  # (None, 50, 150, 32)
    x = inception_module(filters=32, inputs=c1)
    x = inception_module(filters=32, inputs=x)
    x = MaxPooling2D(pool_size=(2, 2))(x)  # (None, 25, 75, 32)

    c2 = inception_module(filters=64, inputs=x)  # (None, 25, 75, 64)
    x = inception_module(filters=64, inputs=c2)
    x = inception_module(filters=64, inputs=x)
    x = MaxPooling2D(pool_size=(2, 2))(x)  # (None, 12, 37, 64)

    c3 = inception_module(filters=128, inputs=x)  # (None, 12, 37, 128)
    x = inception_module(filters=128, inputs=c3)
    x = inception_module(filters=128, inputs=x)
    x = MaxPooling2D(pool_size=(2, 2))(x)  # (None, 6, 18, 128)

    c4 = inception_module(filters=256, inputs=x)  # (None, 6, 18, 256)
    x = inception_module(filters=256, inputs=c4)
    x = inception_module(filters=256, inputs=x)
    x = MaxPooling2D(pool_size=(2, 2))(x)  # (None, 3, 9, 256)

    x = inception_module(filters=512, inputs=x)  # (None, 3, 9, 512)
    x = inception_module(filters=512, inputs=x)
    x = inception_module(filters=512, inputs=x)  # (None, 3, 9, 512)

    x = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(x)
    x = concatenate([x, c4], axis=3)  # (None, 3, 9, 512) -> (None, 6, 18, 256)
    x = inception_module(filters=256, inputs=x)
    x = inception_module(filters=256, inputs=x)
    x = inception_module(filters=256, inputs=x)

    x = concatenate(
        [ZeroPadding2D(((0, 0), (1, 0)))(Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(x)),
         c3], axis=3)  # (None, 6, 18, 256) -> (None, 12, 36, 128) -> (None, 12, 37, 128)
    x = inception_module(filters=128, inputs=x)
    x = inception_module(filters=128, inputs=x)
    x = inception_module(filters=128, inputs=x)

    x = concatenate(
        [ZeroPadding2D(((1, 0), (1, 0)))(Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(x)),
         c2], axis=3)  # (None, 12, 37, 128) -> (None, 24, 74, 64) -> (None, 25, 75, 64)
    x = inception_module(filters=64, inputs=x)
    x = inception_module(filters=64, inputs=x)
    x = inception_module(filters=64, inputs=x)

    x = concatenate(
        [Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(x),
         c1], axis=3)  # (None, 25, 75, 64) -> (None, 50, 150, 32)

    x = inception_module(filters=32, inputs=x)
    x = inception_module(filters=32, inputs=x)
    x = inception_module(filters=32, inputs=x)

    outputs = Conv2D(4, (1, 1), activation='linear')(x)

    model = Model(inputs=[inputs], outputs=[outputs])

    plot_model(model, to_file='model_architecture_with_inception2.png', show_shapes=True, show_layer_names=False)

    return model


model = keras_model()

hdf5_path = 'unified_data.hdf5'
x_train, y_train, x_test, y_test = read_dataset(hdf5_path, split=0.8, print_shape=False)


lr = 1e-05
model = keras_model()
optimizer = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.45*lr, amsgrad=True)

model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['MSE'])
callbacks = [EarlyStopping(monitor='val_mean_squared_error', min_delta=0.001, patience=5)]
train_info = model.fit(x=x_train, y=y_train, batch_size=4, epochs=50, verbose=1, validation_data=(x_test, y_test), callbacks=callbacks)
score = model.evaluate(x=x_test, y=y_test, verbose=2)
print('Average Mean Squared Error:', score[0])
model.save('my_model.h5')