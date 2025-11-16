# import packages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras.models import Sequential
import numpy as np

# import my own module
from bicoder import BiCoder

# docstring
import argparse
import textwrap
parser = argparse.ArgumentParser(prog='Subclass Encoder',
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=textwrap.dedent('''\
                                            Subclass Encoder
                                ----------------------------------------
                                Encoder that builds the neural network model it consists of
                                (one suitable for encoding black and white images or color images
                                depending on whether the user choses img_type 'bw' or 'color'
                                when constructing the object), runs the data through the neural
                                network to produce the encoder output (mu and log variance) and 
                                generates latent space z.
                                                            
                                Methods:
                                Inherited from superclass (make_neural_network). The 2 private
                                methods that consist of the actual neural network architecture
                                and the call method are implemented here. The make_neural_network
                                method from superclass is called in the superclass constructor
                                therefore there is no need to call it, the neural network will be
                                build automatically when object initiated.
                                
                                1)  call: Takes the input data and runs it through the encoder
                                    neural network in order to produce the outputs (mu and log_var).
                                    @param x = input data for the encoder
                                    @return mu, log_var - the outputs of the encoder neural network
                                                            
                                2)  calculate_z: Takes the encoder outputs and generates the latent
                                    space z according to z = mu + eps*std where eps ~ N(0, 1).
                                    @param mu = the output of the encoder
                                    @param log_var = the other output of the encoder
                                    @return z - latent space
                                
                                '''),

                                epilog=textwrap.dedent('''\
                                        Subclass Encoder Usage
                                ------------------------------------------
                                encoder = Encoder('bw')                 # initialize an encoder object for black and white images
                                mu, log_var = encoder(data)             # get the encoder output mu and log_var from the inputed data
                                z = encoder.calculate_z(mu, log_var)    # generate z from the encoder outputs
                                
                                ''')
                )


class Encoder(BiCoder):
    def __init__(self, img_type, latent_dim = None, units = None, activation = 'relu',  # BiCoder input params
                input_shape = None,                                                     # extra input param for Encoder
                *, kernel_size = 3, strides = 2, filters = 32):                         # optional BiCoder input params only for color
    
        # def extra encoder instance variable and set default based on img_type
        if img_type == 'bw':
            self._input_shape = (28*28,) if input_shape is None else input_shape
        elif img_type == 'color':
            self._input_shape = (28,28,3) if input_shape is None else input_shape

        # needs to be after I def the extra variables since superclass constructor calls make_neural_network which needs all the params
        super().__init__(img_type, latent_dim, units, activation,
                        kernel_size = kernel_size, strides = strides, filters = filters)

        # add the superclass help text to the subclass help
        self.parser_subclass = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                    prog = parser.prog,
                    description = parser.description + self.parser_superclass.description,
                    epilog = parser.epilog + self.parser_superclass.epilog
                    )

    # Prints the nicely formatted docstring
    @property
    def help(self):
        self.parser_subclass.print_help()

    # private method that builds the encoder neural network for BW images using Rogelio's architecture
    def _make_neural_network_BW(self):
        encoder_BW = Sequential(
                        [ 
                        layers.InputLayer(input_shape=self._input_shape),
                        layers.Dense(self._units, activation=self._activation),
                        layers.Dense(2*self._latent_dim),
                        ]
                        )
        
        return encoder_BW
    
    # private method that builds the encoder neural network for color images using Rogelio's architecture
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

    # call function so that we can get the encoder output by calling only encoder(data)
    def call(self, x):
        encoder = self._neural_network
        out = encoder(x)
        mu = out[:,:self._latent_dim]
        log_var = out[:,self._latent_dim:]
        return mu, log_var

    # use encoder output to get z
    def calculate_z(self, mu, log_var):
        std = tf.math.exp(0.5*log_var)
        eps = tf.random.normal(mu.shape)
        z = mu + eps*std
        return z
    
en = Encoder('bw')
en.help