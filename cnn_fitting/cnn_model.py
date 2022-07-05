import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd


def cnn_model_simple(input_shape, mdn=False):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (5, 5), activation='relu', padding="same", input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(BatchNormalization(axis=-1))
    model.add(Dropout(0.2))

    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding="same"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(BatchNormalization(axis=-1))
    model.add(Dropout(0.2))

    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding="same"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(BatchNormalization(axis=-1))
    model.add(Dropout(0.2))

    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization(axis=-1))
    model.add(layers.Dense(10, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.2))

    if not mdn:
        model.add(layers.Dense(1, activation='linear'))
    else:
        model.add(layers.Flatten())
        model.add(Dense(tfp.layers.IndependentNormal.params_size(1), activation=None))
        model.add(tfp.layers.IndependentNormal(1, tfd.Normal.sample))

    # compile CNN
    opt = Adam(learning_rate=1e-3, decay=1e-6, clipnorm=1.)
    model.compile(loss=loss_fn(mdn), optimizer=opt, metrics=['mse'])

    model.summary()

    return model


def load_saved_model(path, mdn=True):
    opt = Adam(learning_rate=1e-3, decay=1e-6, clipnorm=1.)
    model = models.load_model(path, compile=False)
    model.compile(loss=loss_fn(mdn), optimizer=opt, metrics=['mse'])
    return model


def loss_fn(mdn=False):
    if mdn:
        return lambda x, rv_x: -rv_x.log_prob(x)
    return "mean_squared_error"

