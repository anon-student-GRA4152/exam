import tensorflow as tf
import numpy as np

from bicoder import Encoder, Decoder

from training_logging import LogTraining

'''

Methods:
    - call -> input only data
    - kl_divergence -> from Rogelio
    - log_diag_mvn -> form Rogelio
    - train -> mostly from Rogelio
    - generate latent space
    - sample from prior
    - sample from posterior

'''



class VAE(tf.keras.Model):

    # need all the encoder/decoder params to be inputted here too
    def __init__(self, img_type, latent_dim = None, units = None, activation = 'relu',  # BiCoder params
                input_shape = None,                                                     # extra Encoder params
                output_dim = 28*28, target_shape = (4,4,128), channel_out = 3,          # extra Decoder params
                kernel_size = 3, strides = 2, filters = 32):                            # optional BiCoder params only for color (but no * cause everything before this "required" for VAE)

        # tf.keras.Model constructor so that VAE can inherit from there
        super().__init__()

        # create encoder and decoder objects -> stable over life of VAE instance
        self.encoder = Encoder(img_type, latent_dim , units, activation, input_shape,
                                kernel_size = kernel_size, strides = strides, filters = filters)
        self.decoder = Decoder(img_type, latent_dim, units, activation, output_dim, target_shape, channel_out, 
                                kernel_size = kernel_size, strides = strides, filters = filters)


    # mu and log_var from encoder
    def _kl_divergence(self, mu, log_var):
        return 0.5 * tf.reduce_sum(tf.square(mu) + tf.exp(log_var) - log_var - 1, axis=-1) 

    # mu, log_sigma from decoder
    def _log_diag_mvn(self, x, mu, log_sigma):
        sum_axes = tf.range(1, tf.rank(mu))
        k = tf.cast(tf.reduce_prod(tf.shape(mu)[1:]), x.dtype)
        logp =  - 0.5 * k * tf.math.log(2*np.pi) \
                - log_sigma \
                - 0.5*tf.reduce_sum(tf.square(x - mu)/tf.math.exp(2.*log_sigma),axis=sum_axes)
        return logp

    def call(self, x):
        encoder_mu, log_var = self.encoder(x)
        z = self.encoder.calculate_z(encoder_mu, log_var)
        decoder_mu, log_sigma = self.decoder(z)
        kl_div = self._kl_divergence(encoder_mu, log_var)
        log_diag_mnv = self._log_diag_mvn(x, decoder_mu, log_sigma)
        elbo = log_diag_mnv - kl_div
        loss = - elbo
        return loss
    
    # here the returned loss is for monitoring of the training (how many epochs u need)
    @tf.function    # we use our custom decorator to log training loss
    @LogTraining
    def train(self, x, optimizer):
        with tf.GradientTape() as tape:
            loss = self.call(x)
        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return loss

    def generate_latent_space_z(self, x):
        '''
        Visualize the z from encoder

        '''
        encoder_mu, log_var = self.encoder(x)
        z = self.encoder.calculate_z(encoder_mu, log_var)  
        return z

    def generate_new_images_from_prior(self, number_of_images = 100, random_sampling = False):
        '''
        prior is isotropic Gaussian dist N(0, I) so we sample z from this
        and then use this z in decoder(z)

        Use 100 as default for number of images cause ofc we nee dto visualize this and have a reasonable number
        Rogelio used 10x10 for the grid function making it 100 images as default
        
        '''
        # the latent_dim must match my usual image vector size so that encoder works (it's set up for that size) 
        z = tf.random.normal((number_of_images, self.encoder._latent_dim))

        decoder_mu, log_sigma = self.decoder(z)

        if random_sampling:
            return self.decoder.get_x(decoder_mu, log_sigma)
        
        # want the expectation for sharper image == default
        else:
            return decoder_mu



    def generate_new_images_from_posterior(self, x, random_sampling = False):
        '''
        here we use the actual z from encoder
        random sampling -> x = decoder.get_x(mu_of_x)
        expectation sampling -> mu_of_x

        '''
        encoder_mu, log_var = self.encoder(x)
        z = self.encoder.calculate_z(encoder_mu, log_var) 
        decoder_mu, log_sigma = self.decoder(z)

        if random_sampling:
            return self.decoder.get_x(decoder_mu, log_sigma)
        
        # want the expectation for sharper image == default
        else:
            return decoder_mu

