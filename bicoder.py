# import packages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras.models import Sequential
import numpy as np

# docstring
import argparse
import textwrap
parser = argparse.ArgumentParser(prog='Superclass BiCoder',
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=textwrap.dedent('''\
                                        Superclass BiCoder
                                ----------------------------------------
                                Superclass that sets up the structure my Encoder
                                and Decoder subclasses can fit into. All the common
                                neural network parameters (e.g. latent dimension,
                                activation) that both Encoder and Dceoder will later
                                use are set up here. However it has no real functionality
                                without the subclasses extending it.
                                                            
                                Methods:
                                1)  make_neural_network: Constructor method that calls 
                                    one of the two private methods where the neural
                                    network is actually build depending on whether
                                    it should be for black and white or color images. 
                                    The user can specify this when constructing the 
                                    object with input parameter img_type. The private
                                    methods actually building the neural networks are
                                    abstract here and need to be implemented in the
                                    subclasses.
                                    @return the neural_network that was build
                                                            
                                2)  call: abstract method, needs to be implemented
                                    in the subclasses
                                
                                '''),

                                epilog=textwrap.dedent('''\
                                        Superclass BiCoder Usage
                                ------------------------------------------
                                This is essentially an empty method when it comes to 
                                real functionality. It serves the purpose of creating
                                the structure for the subclasses to fit into. But for
                                usage it really needs to have subclasses implemented.
                                ''')
                )

class BiCoder(layers.Layer):

    # setting Rogelio's values for the neural network as default
    def __init__(self, img_type, latent_dim = None, units = None, activation = 'relu',  # input params for both bw and color
                *, kernel_size = 3, strides = 2, filters = 32):                         # optional input params specifically for color

        # layers.Layer constructor so that BiCoder can inherit from there
        super().__init__()

        # active debugging - input check that type is only 'bw' or 'color'
        possible_img_types = ['bw', 'color']
        assert img_type in possible_img_types, 'Image type you have inputted is not valid. It must be one of the following options: {}'.format(possible_img_types)
        
        self._img_type = img_type
        self._activation = activation

        # set bw vs color specific defaults (latent dim + units)
        # + the last 3 input params expected only for color so initialize instance varibales only in color case
        if self._img_type == 'bw':
            self._latent_dim = 20 if latent_dim is None else latent_dim
            self._units = 400 if units is None else units

        elif self._img_type == 'color':
            self._latent_dim = 50 if latent_dim is None else latent_dim
            self._units = units
            self._kernel_size = kernel_size
            self._strides =  strides
            self._filters = filters

        # one instance of encoder/decoder should have a stable neural network made just once so keep track of
        self._neural_network = self.make_neural_network()

        # instance variable for the docstring
        self.parser_superclass = parser

    # Prints the nicely formatted docstring
    @property
    def help(self):
        self.parser_superclass.print_help()

    # the neural network should be made depending on image type
    def make_neural_network(self):
        if self._img_type == 'bw':
            neural_network = self._make_neural_network_BW()

        elif self._img_type == 'color':
            neural_network = self._make_neural_network_COLOR()
    
        return neural_network

    # abstract method
    def _make_neural_network_BW(self):
        raise NotImplementedError

    # abstract method
    def _make_neural_network_COLOR(self):
        raise NotImplementedError
    
    # abstract method
    def call(self):
        raise NotImplementedError




        
