from keras.models import Model
from keras.layers.convolutional import ZeroPadding2D
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.utils.vis_utils import plot_model


def keras_model(x_train, save_model=True):
    inputs = Input(x_train.shape[1:])  # Input(shape), return a tensor
    # Initial shape: (None, 50, 150, 1). Here, the channel number / depth might get changed, but don't worry!

    # 2 3x3 convolutions followed by a max pooling
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)  # (None, 50, 150, 32)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)  # (None, 50, 150, 32)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)  # (None, 50, 150, 32)
    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)  # (None, 50, 150, 32)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv2)  # (None, 25, 75, 32)

    # 2 3x3 convolutions followed by a max pooling
    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)  # (None, 25, 75, 64)
    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv3)  # (None, 25, 75, 64)
    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv3)  # (None, 25, 75, 64)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv3)  # (None, 25, 75, 64)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv4)  # (None, 12, 37, 64)

    # 2 3x3 convolutions followed by a max pooling
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)  # (None, 12, 37, 128)
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv5)  # (None, 12, 37, 128)
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv5)  # (None, 12, 37, 128)
    conv6 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv5)  # (None, 12, 37, 128)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv6)  # (None, 6, 18, 128)

    # 2 3x3 convolutions followed by a max pooling
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)  # (None, 6, 18, 256)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)  # (None, 6, 18, 256)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)  # (None, 6, 18, 256)
    conv8 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)  # (None, 6, 18, 256)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv8)  # (None, 3, 9, 256)

    # 2 3x3 convolutions
    conv9 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)  # (None, 3, 9, 512)
    conv9 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv9)  # (None, 3, 9, 512)
    conv9 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv9)  # (None, 3, 9, 512)
    conv10 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv9)  # (None, 3, 9, 512)

    # 1 3x3 transpose convolution and concat conv8 on the depth dim
    # TODO Learn more about Conv2DTranspose output shape, usually here it double the input shape
    concat1 = concatenate(
        [Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv10), conv8],
        axis=3)  # (None, 3, 9, 512) -> (None, 6, 18, 256) -> (None, 6, 18, 512)

    # 2 3x3 convolutions
    conv11 = Conv2D(256, (3, 3), activation='relu', padding='same')(concat1)  # (None, 6, 18, 256)
    conv11 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv11)  # (None, 6, 18, 256)
    conv11 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv11)  # (None, 6, 18, 256)
    conv12 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv11)  # (None, 6, 18, 256)

    # 1 3x3 transpose convolution and concat conv6 on the depth dim
    concat2 = concatenate(
        [ZeroPadding2D(((0, 0), (1, 0)))(Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv12)), conv6],
        axis=3)  # (None, 6, 18, 256) -> (None, 12, 36, 128) -> (None, 12, 37, 128) -> (None, 12, 37, 256)

    # 2 3x3 convolutions
    conv13 = Conv2D(128, (3, 3), activation='relu', padding='same')(concat2)  # (None, 12, 37, 128)
    conv13 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv13)  # (None, 12, 37, 128)
    conv13 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv13)  # (None, 12, 37, 128)
    conv14 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv13)  # (None, 12, 37, 128)

    # 1 3x3 transpose convolution and concat conv4 on the depth dim
    concat3 = concatenate(
        [ZeroPadding2D(((1, 0), (1, 0)))(Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv14)), conv4],
        axis=3)  # (None, 12, 37, 128) -> (None, 24, 74, 64) -> (None, 25, 75, 64) -> (None, 25, 75, 128)

    # 2 3x3 convolutions
    conv15 = Conv2D(64, (3, 3), activation='relu', padding='same')(concat3)  # (None, 25, 75, 64)
    conv15 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv15)  # (None, 25, 75, 64)
    conv15 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv15)  # (None, 25, 75, 64)
    conv16 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv15)  # (None, 25, 75, 64)

    # 1 3x3 transpose convolution and concat conv2 on the depth dim
    concat4 = concatenate(
        [Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv16), conv2],
        axis=3)  # (None, 25, 75, 64) -> (None, 50, 150, 32) -> (None, 50, 150, 64)

    # 2 3x3 convolutions
    conv17 = Conv2D(32, (3, 3), activation='relu', padding='same')(concat4)  # (None, 50, 150, 32)
    conv17 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv17)  # (None, 50, 150, 32)
    conv17 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv17)  # (None, 50, 150, 32)
    conv18 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv17)  # (None, 50, 150, 32)

    # Final 1x1 convolutions to get to the correct depth dim 1, for now we only take magnitude of velocity
    # TODO Will check the output with (x, y) velocity, pressure and other available results from simulation
    conv19 = Conv2D(1, (1, 1), activation='linear')(conv18)
    # TODO Check the activation for out model

    model = Model(inputs=[inputs], outputs=[conv19])

    if save_model:
        plot_model(model, to_file='model_architecture.png', show_shapes=True, show_layer_names=True)

    return model


# The more deep the model the more accuracy
# TODO Now try Inception model
# TODO In the official Keras documentation they showed residual networks by adding two layers, instead of concat.
# Find why?

def keras_model(inputs, save_model=True):
    inputs = Input(x_train.shape[1:])  # Input(shape), return a tensor
    # Initial shape: (None, 50, 150, 1). Here, the channel number / depth might get changed, but don't worry!

    # 2 Conv. layer
    conv1 = Conv2D(16, (3, 3), padding='same', activation='relu')(inputs)  # (None, 50, 150, 16)
    conv2 = Conv2D(16, (5, 5), padding='same', activation='relu')(conv1)  # (None, 50, 150, 16)

    # 4 Inception module followed by a max pooling
    inception1 = Inception(filters=32, inputs=conv2)  # (None, 50, 150, 32)
    inception1 = Inception(filters=32, inputs=inception1)
    inception1 = Inception(filters=32, inputs=inception1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(inception1)  # (None, 25, 75, 32)

    inception2 = Inception(filters=64, inputs=pool1)  # (None, 25, 75, 64)
    inception2 = Inception(filters=64, inputs=inception2)
    inception2 = Inception(filters=64, inputs=inception2)

    pool2 = MaxPooling2D(pool_size=(2, 2))(inception2)  # (None, 12, 37, 64)

    inception3 = Inception(filters=128, inputs=pool2)  # (None, 12, 37, 128)
    inception3 = Inception(filters=128, inputs=inception3)

    pool3 = MaxPooling2D(pool_size=(2, 2))(inception3)  # (None, 6, 18, 128)

    inception4 = Inception(filters=256, inputs=pool3)  # (None, 6, 18, 256)
    inception4 = Inception(filters=256, inputs=inception4)
    inception4 = Inception(filters=256, inputs=inception4)

    pool4 = MaxPooling2D(pool_size=(2, 2))(inception4)  # (None, 3, 9, 256)

    # 2 Inception module
    inception5 = Inception(filters=512, inputs=pool4)  # (None, 3, 9, 512)
    inception6 = Inception(filters=512, inputs=inception5)  # (None, 3, 9, 512)
    #inception6 = Inception(filters=512, inputs=inception6)

    # 4 Residual connection
    add1 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(inception6),
                        inception4], axis=3)  # (None, 3, 9, 512) -> (None, 6, 18, 256)
    inception7 = Inception(filters=256, inputs=add1)
    inception7 = Inception(filters=256, inputs=inception7)
    # inception7 = Inception(filters=256, inputs=inception7)
    add2 = concatenate(
        [ZeroPadding2D(((0, 0), (1, 0)))(Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(inception7)),
         inception3], axis=3)  # (None, 6, 18, 256) -> (None, 12, 36, 128) -> (None, 12, 37, 128)
    inception8 = Inception(filters=128, inputs=add2)
    inception8 = Inception(filters=128, inputs=inception8)
    # inception8 = Inception(filters=128, inputs=inception8)
    add3 = concatenate(
        [ZeroPadding2D(((1, 0), (1, 0)))(Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(inception8)),
         inception2], axis=3)  # (None, 12, 37, 128) -> (None, 24, 74, 64) -> (None, 25, 75, 64)
    inception9 = Inception(filters=64, inputs=add3)
    inception9 = Inception(filters=64, inputs=inception9)
    # inception9 = Inception(filters=64, inputs=inception9)

    add4 = concatenate(
        [Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(inception9),
         inception1], axis=3)  # (None, 25, 75, 64) -> (None, 50, 150, 32)

    # 1 Conv. for output with linear activation
    outputs = Conv2D(1, (1, 1), activation='linear')(add4)

    model = Model(inputs=[inputs], outputs=[outputs])

    if save_model:
        plot_model(model, to_file='model_architecture_with_inception.png', show_shapes=True, show_layer_names=True)

    return model