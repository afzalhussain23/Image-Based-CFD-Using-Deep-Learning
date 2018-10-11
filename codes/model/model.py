from keras.models import Model
from keras.layers import add, concatenate, Input, Cropping2D, Conv2D, Conv3D, MaxPooling2D, Conv2DTranspose, \
    TimeDistributed
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.utils.vis_utils import plot_model


def inception_module(filters=32, inputs=None):
    tower_0 = Conv2D(int(filters / 4), (1, 1), padding='same', activation='relu')(inputs)

    tower_1 = Conv2D(int(filters / 4), (1, 1), padding='same', activation='relu')(inputs)
    tower_1 = Conv2D(int((filters * 3) / 8), (3, 3), padding='same', activation='relu')(tower_1)

    tower_2 = Conv2D(int(filters / 8), (1, 1), padding='same', activation='relu')(inputs)
    tower_2 = Conv2D(int(filters / 8), (5, 5), padding='same', activation='relu')(tower_2)

    tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(inputs)
    tower_3 = Conv2D(int(filters / 4), (1, 1), padding='same', activation='relu')(tower_3)

    concat = concatenate([tower_0, tower_1, tower_2, tower_3], axis=3)

    return concat


def cnn_model():
    inputs = Input(shape=(50, 150, 4))

    c1 = inception_module(filters=32, inputs=inputs)  # (None, 50, 150, 32)
    x = MaxPooling2D((2, 2), padding='same')(c1)  # (None, 25, 75, 32)

    c2 = inception_module(filters=64, inputs=x)  # (None, 25, 75, 64)
    x = MaxPooling2D((2, 2), padding='same')(c2)  # (None, 13, 38, 64)

    c3 = inception_module(filters=128, inputs=x)  # (None, 13, 38, 128)
    x = MaxPooling2D((2, 2), padding='same')(c3)  # (None, 7, 19, 128)

    c4 = inception_module(filters=256, inputs=x)  # (None, 7, 19, 256)
    x = MaxPooling2D((2, 2), padding='same')(c4)  # (None, 4, 10, 256)

    x = inception_module(filters=512, inputs=x)  # (None, 4, 10, 512)

    x = add([Cropping2D(cropping=((1, 0), (1, 0)))(Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(x)),
             c4])  # (None, 4, 10, 512) -> (None, 8, 20, 256) -> (None, 7, 19, 256)
    x = add([Cropping2D(cropping=((1, 0), (0, 0)))(Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(x)),
             c3])  # (None, 7, 19, 256) -> (None, 14, 38, 128) -> (None, 13, 38, 128)
    x = add([Cropping2D(cropping=((1, 0), (1, 0)))(Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(x)),
             c2])  # (None, 13, 38, 128) -> (None, 26, 76, 64) -> (None, 25, 75, 64)
    x = add([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(x),
             c1])  # ((None, 25, 75, 64) -> (None, 50, 150, 32)

    model = Model(inputs=inputs, outputs=x)
    plot_model(model, to_file='cnn_model.png', show_shapes=True, show_layer_names=False)

    return model


def lstm_model(save_model=True):
    video_input = Input(shape=(40, 50, 150, 4))

    x = TimeDistributed(cnn_model())(video_input)

    x = ConvLSTM2D(filters=16, kernel_size=(3, 3), activation='relu',
                   recurrent_activation='hard_sigmoid', padding='same', return_sequences=True)(x)
    x = ConvLSTM2D(filters=16, kernel_size=(3, 3), activation='relu',
                   recurrent_activation='hard_sigmoid', padding='same', return_sequences=True)(x)
    x = ConvLSTM2D(filters=16, kernel_size=(3, 3), activation='relu',
                   recurrent_activation='hard_sigmoid', padding='same', return_sequences=True)(x)  # (None, 50, 150, 16)

    x = Conv3D(filters=4, kernel_size=(3, 3, 3), activation='linear', padding='same', data_format='channels_last')(x)

    model = Model(inputs=video_input, outputs=x)
    print(model.summary())

    if save_model:
        plot_model(model, to_file='lstm_model.png', show_shapes=True, show_layer_names=False)


lstm_model(save_model=True)
