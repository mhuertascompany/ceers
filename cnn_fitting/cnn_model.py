import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd


class CNNModelTemplate(object):
    """
    Used to generate our multi-output model. This CNN contains three branches, one for effective radius,
    one for sersic index and another one for ellipticity.
    Each branch contains a sequence of Convolutional Layers that is defined
    on the make_default_hidden_layers method.
    """
    def __init__(self, input_shape, mdn=True):
        self.mdn = mdn
        self.model = self.assemble_full_model(input_shape)

    def load_saved_model(self, path):
        self.model.built = True
        self.compile()
        self.model.load_weights(path)
        return self.model

    def compile(self):
        opt = Adam(learning_rate=1e-3, decay=1e-6, clipnorm=1.)
        self.model.compile(optimizer=opt, loss=loss_fn(self.mdn), metrics='mse')
                           #loss={
                           #     'radius_output': loss_fn(self.mdn),
                           #     'sidx_output': loss_fn(self.mdn),
                           #     'ellipt_output': loss_fn(self.mdn)},
                           #metrics={
                           #     'radius_output': 'mse',
                           #     'sidx_output': 'mse',
                           #     'ellipt_output': 'mse'})

        self.model.summary()
        return self.model

    def make_default_hidden_layers(self, inputs):
        """
        Used to generate a default set of hidden layers. The structure used in this network is defined as:

        Conv2D -> BatchNormalization -> Pooling -> Dropout
        """
        x = Conv2D(16, (3, 3), activation='relu', padding="same")(inputs)
        x = BatchNormalization(axis=-1)(x)
        x = MaxPooling2D(pool_size=(3, 3))(x)
        x = Dropout(0.25)(x)
        x = Conv2D(32, (3, 3), activation='relu', padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)
        x = Conv2D(32, (3, 3), activation='relu', padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)
        return x

    def build_branch(self, inputs, output_name):
        """
        Used to build the race branch of our face recognition network.
        This branch is composed of three Conv -> BN -> Pool -> Dropout blocks,
        followed by the Dense output layer.
        """

        x = self.make_default_hidden_layers(inputs)
        x = Flatten()(x)
        x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(10, activation='relu', kernel_regularizer=l2(0.001))(x)

        x = Flatten()(x)
        x = Dense(tfp.layers.IndependentNormal.params_size(1), activation=None)(x)
        x = tfp.layers.IndependentNormal(1, tfd.Normal.sample, name=output_name)(x)
        return x

    def assemble_full_model(self, input_shape):
        """
        Used to assemble our multi-output model CNN.
        """

        inputs = Input(shape=input_shape)
        branch = self.build_branch(inputs, 'output')
        #sersic_index_branch = self.build_branch(inputs, 'sidx_output')
        #ellipticity_branch = self.build_branch(inputs, 'ellipt_output')

        model = tf.keras.Model(inputs=inputs,
                               outputs=[branch])
        return model


def cnn_model_simple(input_shape, mdn=False):
    model = models.Sequential()
    model.add(layers.Conv2D(16, (3, 3), activation='relu', padding="same", input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(BatchNormalization(axis=-1))
    model.add(Dropout(0.1))

    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding="same"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(BatchNormalization(axis=-1))
    model.add(Dropout(0.1))

    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding="same"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(BatchNormalization(axis=-1))
    model.add(Dropout(0.1))

    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization(axis=-1))
    model.add(layers.Dense(10, activation='relu', kernel_regularizer=l2(0.001)))
    #model.add(Dropout(0.1))

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

