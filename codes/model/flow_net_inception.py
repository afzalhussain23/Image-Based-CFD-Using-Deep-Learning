from keras.models import Model
from keras.layers.convolutional import ZeroPadding2D
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.utils.vis_utils import plot_model


def inception_module(filters=None, inputs=None):
    tower_0 = Conv2D(int(filters / 4), (1, 1), padding='same', activation='linear')(inputs)

    tower_1 = Conv2D(int(filters / 4), (1, 1), padding='same', activation='linear')(inputs)
    tower_1 = Conv2D(int((filters * 3) / 8), (3, 3), padding='same', activation='linear')(tower_1)

    tower_2 = Conv2D(int(filters / 8), (1, 1), padding='same', activation='linear')(inputs)
    tower_2 = Conv2D(int(filters / 8), (5, 5), padding='same', activation='linear')(tower_2)

    tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(inputs)
    tower_3 = Conv2D(int(filters / 4), (1, 1), padding='same', activation='linear')(tower_3)

    concat = concatenate([tower_0, tower_1, tower_2, tower_3], axis=3)

    return concat


def keras_model(x_train, save_model=True):
    inputs = Input(x_train.shape[1:])  # Input(shape), return a tensor
    # Initial shape: (None, 50, 150, 1). Here, the channel number / depth might get changed, but don't worry!

    # 2 Conv. layer
    conv1 = Conv2D(16, (3, 3), padding='same', activation='linear')(inputs)  # (None, 50, 150, 16)
    conv2 = Conv2D(16, (5, 5), padding='same', activation='linear')(conv1)  # (None, 50, 150, 16)

    # 4 inception_module module followed by a max pooling
    inception1 = inception_module(filters=32, inputs=conv2)  # (None, 50, 150, 32)
    # inception1 = inception_module(filters=32, inputs=inception1)
    inception1 = inception_module(filters=32, inputs=inception1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(inception1)  # (None, 25, 75, 32)

    inception2 = inception_module(filters=64, inputs=pool1)  # (None, 25, 75, 64)
    # inception2 = inception_module(filters=64, inputs=inception2)
    inception2 = inception_module(filters=64, inputs=inception2)

    pool2 = MaxPooling2D(pool_size=(2, 2))(inception2)  # (None, 12, 37, 64)

    inception3 = inception_module(filters=128, inputs=pool2)  # (None, 12, 37, 128)
    inception3 = inception_module(filters=128, inputs=inception3)

    pool3 = MaxPooling2D(pool_size=(2, 2))(inception3)  # (None, 6, 18, 128)

    inception4 = inception_module(filters=256, inputs=pool3)  # (None, 6, 18, 256)
    # inception4 = inception_module(filters=256, inputs=inception4)
    inception4 = inception_module(filters=256, inputs=inception4)

    pool4 = MaxPooling2D(pool_size=(2, 2))(inception4)  # (None, 3, 9, 256)

    # 2 inception_module module
    inception5 = inception_module(filters=512, inputs=pool4)  # (None, 3, 9, 512)
    inception6 = inception_module(filters=512, inputs=inception5)  # (None, 3, 9, 512)
    #inception6 = inception_module(filters=512, inputs=inception6)

    # 4 Residual connection
    add1 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(inception6),
                        inception4], axis=3)  # (None, 3, 9, 512) -> (None, 6, 18, 256)
    inception7 = inception_module(filters=256, inputs=add1)
    inception7 = inception_module(filters=256, inputs=inception7)
    # inception7 = inception_module(filters=256, inputs=inception7)
    add2 = concatenate(
        [ZeroPadding2D(((0, 0), (1, 0)))(Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(inception7)),
         inception3], axis=3)  # (None, 6, 18, 256) -> (None, 12, 36, 128) -> (None, 12, 37, 128)
    inception8 = inception_module(filters=128, inputs=add2)
    inception8 = inception_module(filters=128, inputs=inception8)
    # inception8 = inception_module(filters=128, inputs=inception8)
    add3 = concatenate(
        [ZeroPadding2D(((1, 0), (1, 0)))(Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(inception8)),
         inception2], axis=3)  # (None, 12, 37, 128) -> (None, 24, 74, 64) -> (None, 25, 75, 64)
    inception9 = inception_module(filters=64, inputs=add3)
    inception9 = inception_module(filters=64, inputs=inception9)
    # inception9 = inception_module(filters=64, inputs=inception9)

    add4 = concatenate(
        [Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(inception9),
         inception1], axis=3)  # (None, 25, 75, 64) -> (None, 50, 150, 32)

    # 1 Conv. for output with linear activation
    outputs = Conv2D(1, (1, 1), activation='linear')(add4)

    model = Model(inputs=[inputs], outputs=[outputs])

    if save_model:
        plot_model(model, to_file='model_architecture_with_inception.png', show_shapes=True, show_layer_names=True)

    return model

# The more deep the model the more accuracy
# TODO Will check the output with (x, y) velocity, pressure and other available results from simulation
# TODO Learn more about Conv2DTranspose output shape, usually here it double the input shape
# TODO Now try inception_module model
# TODO In the official Keras documentation they showed residual networks by adding two layers, instead of concat.
# Find why?
# TODO Check the activation for out model
# TODO Add some more conv layer at beginning and ending

# TODO Try with concat

# TODO Try with sequence model

# TODO Retry Everything in cloud in a systematic manner

# TODO Try with a simple CNN to predict the whole idea!! Seriously it's important

# TODO Make a good inception_module model with a good LSTM!


