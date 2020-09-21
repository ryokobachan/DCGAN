from tensorflow.keras import layers
import tensorflow as tf

def make_generator_model(w,h,c):
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256*c, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256*c)))
    assert model.output_shape == (None, 7, 7, 256*c) # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128*c, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128*c)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64*c, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64*c)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(c, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, c)

    model.summary()

    return model

def generator_loss(fake_output,cross_entropy):
    return cross_entropy(tf.ones_like(fake_output), fake_output)
