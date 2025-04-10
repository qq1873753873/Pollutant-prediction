import tensorflow as tf
import keras
from keras import layers
model = keras.Sequential([
    layers.LSTM(32, input_shape=[5, 20], return_sequences=True, kernel_regularizer=keras.regularizers.l2(0.01)),
    layers.LSTM(32, return_sequences=False),
    layers.Dense(3, activation='softmax'),
])

tf.keras.utils.plot_model(model, to_file='model3.png', show_shapes=True, show_layer_names=True,rankdir='TB', dpi=900, expand_nested=True)