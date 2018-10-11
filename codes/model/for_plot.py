from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.utils.vis_utils import plot_model

seq = Sequential()
seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3), input_shape=(None, 5, 15, 1), padding='same', return_sequences=True))
seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3), padding='same', return_sequences=True))


seq.add(Conv3D(filters=1, kernel_size=(3, 3, 3), activation='sigmoid', padding='same', data_format='channels_last'))
plot_model(seq, to_file='model_ConvLSTM2D.png', show_shapes=True, show_layer_names=True)
