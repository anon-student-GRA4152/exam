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
parser = argparse.ArgumentParser(prog='Subclass Decoder',
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=textwrap.dedent('''\
                                            Subclass Decoder
                                ----------------------------------------
                                Decoder that builds the neural network model it consists of
                                (one suitable for decoding black and white images or color images
                                depending on whether the user choses img_type 'bw' or 'color'
                                when constructing the object), runs the inputted z through the 
                                neural network to produce the decoder output (mu and log std) 
                                and generates x.
                                                            
                                Methods:
                                Inherited from superclass (make_neural_network). The 2 private
                                methods that consist of the actual neural network architecture
                                and the call method are implemented here. The make_neural_network
                                method from superclass is called in the superclass constructor
                                therefore there is no need to call it, the neural network will be
                                build automatically when object initiated.
                                
                                1)  call: Takes the input data and runs it through the decoder
                                    neural network in order to produce the outputs (mu). The other
                                    output parameter log_std we do not train in this case, instead
                                    we fix it at std = 0.75.
                                    @param x = input data for the decoder (typially z from the encoder)
                                    @return mu, log_sigma - the outputs of the decoder neural network
                                                            
                                2)  get_x: Takes the decoder outputs and generates x according to 
                                    x = mu + eps*std where eps ~ N(0, 1).
                                    @param mu = the output of the decoder
                                    @param log_sigma = the other output of the decoder
                                    @return x
                                
                                '''),

                                epilog=textwrap.dedent('''\
                                        Subclass Decoder Usage
                                ------------------------------------------
                                decoder = Decoder('bw')             # initialize a decoder object for black and white images
                                mu, log_sigma = decoder(z)          # get the decoder output mu and log_var from the inputed data
                                x = decoder.get_x(mu, log_sigma)    # generate x from the decoder outputs
                                
                                ''')
                )

class Decoder(BiCoder):
    def __init__(self, img_type, latent_dim = None, units = None, activation = 'relu',   # BiCoder input params
                output_dim = 28*28,                                                      # bw extra Decoder param with default
                target_shape = (4,4,128), channel_out = 3,                               # color extra Decoder params with defaults
                *, kernel_size = 3, strides = 2, filters = 32):                          # optional BiCoder input params only for color
        
        # def instance variable output_dim only for bw images, otherwise disregard the user input of this param
        if img_type == 'bw':
            self._output_dim = output_dim

        # def these 2 extra instance variables only for img_type = color, otherwise disregard
        elif img_type == 'color':
            self._target_shape = target_shape
            self._channel_out = channel_out

        # needs to be after I def the extra variables since superclass constructor calls make_neural_network which needs all the params
        super().__init__(img_type, latent_dim,  units, activation,
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

    # private method that builds the decoder neural network for BW images using Rogelio's architecture
    def _make_neural_network_BW(self):
        decoder_BW = Sequential(
                        [
                        layers.InputLayer(input_shape=self._latent_dim),
                        layers.Dense(self._units, activation=self._activation),
                        layers.Dense(self._output_dim),
                        ]
                        )
        
        return decoder_BW

    # private method that builds the decoder neural network for color images using Rogelio's architecture
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

    # call function so that we can get the decoder output by calling only decoder(data)
    def call(self, x):
        decoder = self._neural_network
        mu = decoder(x)
        std = 0.75  # keep hard coded since it is part of the assignment and should't be changed (no reaosn to allow inputing it or changing the value)
        log_sigma = tf.math.log(std)
        
        # return mu and log_sigma as it would be if we were also training std not only mu
        # so that the code that calls this method can be done in a way that would also work if we are training std (in case we ever want to change it here)
        return mu, log_sigma

    # sample x from the decoder output + random noise
    def get_x(self, mu, log_sigma):
        std = tf.math.exp(0.5*log_sigma)    # have to change back to sigma not log
        eps = tf.random.normal(mu.shape)
        x = mu + eps*std
        return x