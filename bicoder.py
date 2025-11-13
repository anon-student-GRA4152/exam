import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras.models import Sequential
import numpy as np

'''
Should the outputs (mu, log_var, z, idk) be instance variables??? Let's see how it works with VAE

How I wnat this to work -> color_encoder will have its own object instance Encoder(color), color decoder too Dceoder(color), 
and then bw will have its own instances Encoder(bw) and Decoder(bw). 

methods both encoder, decoder -> Bicoder methods
    - _make_neural_network_color -> abstract
    - _make_neural_network_bw -> abstract
    - get_network_output -> abstract
    - make_neural_network -> will call the correct _make_neural_network_bw/color based on img_typ of the object
    - get_network_configuration -> getter method that will just display params of the neural network we building with the current instance of an object

methods encoder
    - _make_neural_network_color -> implemented
    - _make_neural_network_bw -> implemented
    - get_network_output -> gives mu, log_var values
    - do the z stuff at the end with the output -> eps = tf.random.normal(mu.shape), z = mu + eps*std (only here this method) -> returns z
    - get_network_configuration = add the subclass instance params
    

methods decoder
    - _make_neural_network_color -> implemented
    - _make_neural_network_bw -> implemented
    - get_network_output -> gives mu (no log_var cause we just keep it constant assuming std = 0.75)
    - get_network_configuration = add the subclass instance params


instance variables both encoder, decoder -> Bicoder instance variables
    - img_type = bw vs color
    - activation = 'relu'
    - latent_dim = 20 (BW), 50 (color)
    -----------------------
    only color:
    - kernel_size = 3
    - strides = 2
    - filters = 32
    ---------------------
    only bw:
    - units = 400

instance variables encoder
    - input_shape = (28*28,) BW, (28,28,3) color 

instance variables decoder
    ------------------ 
    bw:
    - output_dim = 28*28
    ------------------ 
    color:
    - target_shape=(4,4,128)
    - channel_out=3

'''

class BiCoder(layers.Layer):

    # setting Rogelio's values for the neural network as default
    def __init__(self, img_type, latent_dim = None, units = None, activation = 'relu',  # input params for both bw and color
                *, kernel_size = 3, strides = 2, filters = 32):                         # optional input params specifically for color

        # layers.Layer constructor so that BiCoder can inherit from there
        super().__init__()

        # active debugging -> input check that type is only 'bw' or 'color'
        possible_img_types = ['bw', 'color']
        assert img_type in possible_img_types, 'Image type you have inputted is not valid. It must be one of the following options: {}'.format(possible_img_types)
        
        self._img_type = img_type
        self._activation = activation

        # set bw vs color specific defaults + the last 3 params expected only for color so initialize instance varibales only in that case, otherwise disregard
        if self._img_type == 'bw':
            self._latent_dim = 20 if latent_dim is None else latent_dim
            self._units = 400 if units is None else units

        elif self._img_type == 'color':
            self._latent_dim = 50 if latent_dim is None else latent_dim
            self._units = units
            self._kernel_size = kernel_size
            self._strides =  strides
            self._filters = filters


    # the neural network should be made depending on type
    def make_neural_network(self):
        if self._img_type == 'bw':
            bicoder = self._make_neural_network_BW()

        elif self._img_type == 'color':
            bicoder = self._make_neural_network_COLOR()
    
        return bicoder

    # abstract method
    def _make_neural_network_BW(self):
        raise NotImplementedError

    # abstract method
    def _make_neural_network_COLOR(self):
        raise NotImplementedError
    
    # abstract method
    def get_network_output(self):
        raise NotImplementedError

    def get_network_configuration(self):
        pass


class Encoder(BiCoder):
    def __init__(self, img_type, latent_dim = None, units = None, activation = 'relu',  # BiCoder input params
                input_shape = None,                                                     # extra input param for Encoder
                *, kernel_size = 3, strides = 2, filters = 32):                         # optional BiCoder input params only for color
        
        # def the superclass inistance variables
        super().__init__(img_type, latent_dim, units, activation,
                        kernel_size = kernel_size, strides = strides, filters = filters)

        # def extra encoder instance variable and set default based on img_type
        if self._img_type == 'bw':
            self._input_shape = (28*28,) if input_shape is None else input_shape
        elif self._img_type == 'color':
            self._input_shape = (28,28,3) if input_shape is None else input_shape
            

    def _make_neural_network_BW(self):
        encoder_BW = Sequential(
                        [ 
                        layers.InputLayer(input_shape=self._input_shape),
                        layers.Dense(self._units, activation=self._activation),
                        layers.Dense(2*self._latent_dim),
                        ]
                        )
        
        return encoder_BW

    def _make_neural_network_COLOR(self):
        encoder_COLOR = Sequential(
                        [
                        layers.InputLayer(input_shape=self._input_shape),
                        layers.Conv2D(
                        filters=self._filters, kernel_size=self._kernel_size, strides=self._strides, activation=self._activation, padding='same'),
                        layers.Conv2D(
                        filters=2*self._filters, kernel_size=self._kernel_size, strides=self._strides, activation=self._activation, padding='same'),
                        layers.Conv2D(
                        filters=4*self._filters, kernel_size=self._kernel_size, strides=self._strides, activation=self._activation, padding='same'),
                        layers.Flatten(),
                        layers.Dense(2*self._latent_dim)
                        ]
                        )
        
        return encoder_COLOR

    def get_network_output(self, x):
        encoder = super().make_neural_network()
        out = encoder(x)
        mu = out[:,:self._latent_dim]
        log_var = out[:,self._latent_dim:]

        return mu, log_var

    def calculate_z(self, mu, log_var):
        std = tf.math.exp(0.5*log_var)
        eps = tf.random.normal(mu.shape)
        z = mu + eps*std
        return z

    # add the encoder specific params
    def get_network_configuration(self):
        pass

        
class Decoder(BiCoder):
    def __init__(self, img_type, latent_dim = None, units = None, activation = 'relu',   # BiCoder input params
                output_dim = 28*28,                                                      # bw extra Decoder param with default
                target_shape = (4,4,128), channel_out = 3,                               # color extra Decoder params with defaults
                *, kernel_size = 3, strides = 2, filters = 32):                          # optional BiCoder input params only for color
        
        # def the superclass instance variables
        super().__init__(img_type, latent_dim,  units, activation,
                        kernel_size = kernel_size, strides = strides, filters = filters)

        # def instance variable output_dim only img_type == bw param, otherwise disregard
        if self._img_type == 'bw':
            self._output_dim = output_dim

        # def these 2 extra instance variables only for img_type = color so make them instance varibales only in that case
        if self._img_type == 'color':
            self._target_shape = target_shape
            self._channel_out = channel_out

    
    def _make_neural_network_BW(self):
        decoder_BW = Sequential(
                        [
                        layers.InputLayer(input_shape=self._latent_dim),
                        layers.Dense(self._units, activation=self._activation),
                        layers.Dense(self._output_dim),
                        ]
                        )
        
        return decoder_BW

    def _make_neural_network_COLOR(self):

        # change the units instance varoable that was initialized in superclass as None
        self._units = np.prod(self._target_shape)

        decoder_COLOR = Sequential(
                        [
                        layers.InputLayer(input_shape=(self._latent_dim,)),
                        layers.Dense(units=self._units, activation=self._activation),
                        layers.Reshape(target_shape=self._target_shape),
                        layers.Conv2DTranspose(
                            filters=self._filters*2, kernel_size=self._kernel_size, strides=self._strides, padding='same',output_padding=0,
                            activation=self._activation),
                        layers.Conv2DTranspose(
                            filters=self._filters, kernel_size=self._kernel_size, strides=self._strides, padding='same',output_padding=1,
                            activation=self._activation),
                        layers.Conv2DTranspose(
                            filters=self._channel_out, kernel_size=self._kernel_size, strides=self._strides, padding='same', output_padding=1),
                        layers.Activation('linear', dtype='float32'),
                        ]
                        )
        
        return decoder_COLOR

    def get_network_output(self, x):
        decoder = super().make_neural_network()
        mu = decoder(x)
        return mu

    # sample random x from the posterior here while just get_network_output returns the expectation -> mean (I think that's what Rogelio said?) -> test in VAE downstream tasks
    def get_x(self, mu):
        std = 0.75  # keep hard coded here since it is part of the assignment and should't rly be changed under any circumstances so no reaosn to allow inputing it or anything
        eps = tf.random.normal(mu.shape)
        x = mu + eps*std
        return x

    # add the decoder specific params
    def get_network_configuration(self):
        pass



# bw_encoder = Encoder('bw')
# mu, log_var = bw_encoder.get_network_output(tf.random.normal((4, 28*28)) )
# print(bw_encoder.calculate_z(mu, log_var))