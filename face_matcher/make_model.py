import tensorflow as tf

from gan.config import NOISE_DIM

def make_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, input_shape=(28, 28, 1, )))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Conv2D(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(7*7*256, use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    
    model.add(tf.keras.layers.Dense(NOISE_DIM, use_bias=False))
    return model