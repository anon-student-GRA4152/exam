import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras.models import Sequential

'''
How I wnat this to work -> color_encoder will have its own object instance Encoder(color), color decoder too Dceoder(color), 
and then bw will have its own instances Encoder(bw) and Decoder(bw). 

methods both encoder, decoder -> Bicoder methods
    - _make_neural_network_color -> abstract
    - _make_neural_network_bw -> abstract
    - make_neural_network -> will call the correct _make_neural_network_bw/color based on img_typ of the object
    - show_input_params -> getter method that will just display params of the neural network we building with the current instance of an object

methods encoder
    - _make_neural_network_color -> implemented
    - _make_neural_network_bw -> implemented
    - do the z stuff at the end with the output -> eps = tf.random.normal(mu.shape), z = mu + eps*std (only here this method)
    - show_input_params = add the subclass instance params

methods decoder
    - _make_neural_network_color -> implemented
    - _make_neural_network_bw -> implemented
    - show_input_params = add the subclass instance params


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
    
    
then you have the data (x) ur passing in for traing but that's not instnace variable, that'll be arg for the method running th neural network actually
'''

class BiCoder(layers.Layer):

    # setting Rogelio's values for the neural network as default
    def __init__(self, img_type, latent_dim, activation = 'relu', *, 
                units = 400,    # bw specific
                kernel_size = 3, strides = 2, filters = 32  # color specific
                ): 
        
        # add input check that type is only 'bw' or 'color'

        # all instance variables def here
        self._img_type = img_type
        self._latent_dim = latent_dim       # could add default for bw vs color with ifs?
        self._activation = activation

        # units is only passed for bw img_type
        if self._img_type == 'bw':
            self._units = units

        # the last 3 params are expected to be passed only for img_type = color so make them instance varibales only in that case, otherwise disregard
        if self._img_type == 'color':
            self._kernel_size = kernel_size
            self._strides =  strides
            self._filters = filters

    

class Encoder(BiCoder):
    def __init__(self, img_type, latent_dim, input_shape, activation = 'relu', *, 
                units = 400,
                kernel_size = 3, strides = 2, filters = 32):
        
        # def the superclass inistance variables
        super().__init__(img_type, latent_dim, activation, 
                        units = units, 
                        kernel_size = kernel_size, strides = strides, filters = filters)

        # def all extra instance variables
        self._input_shape = input_shape      # could add default for bw vs color with ifs?



class Decoder(BiCoder):
    def __init__(self, img_type, latent_dim, activation = 'relu',   # Bicoder params
                output_dim = 28*28, # bw extra decoder param with default
                target_shape = (4,4,128), channel_out = 3, *,    # color extra decoder param with default
                units = 400,
                kernel_size = 3, strides = 2, filters = 32):
        
        # def the superclass inistance variables
        super().__init__(img_type, latent_dim, activation, 
                        units = units, 
                        kernel_size = kernel_size, strides = strides, filters = filters)

        # def instance variable output_dim only img_type == bw param, otherwise disregard
        if self._img_type == 'bw':
            self._output_dim = output_dim

        # def these 2 extra instance variables only for img_type = color so make them instance varibales only in that case
        if self._img_type == 'color':
            self._target_shape = target_shape
            self._channel_out = channel_out


