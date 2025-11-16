# import packages
import tensorflow as tf
import numpy as np

# import my own modules
from encoder import Encoder
from decoder import Decoder
from training_logging import LogTraining

# nice docstring
import argparse
import textwrap
parser = argparse.ArgumentParser(prog='class VAE',
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=textwrap.dedent('''\
                                    Subclass VAE (superclass = tf.keras.Model)
                                ---------------------------------------------------
                                Variational Autoencoder that consists of encoder and decoder.
                                It builds the whole model (encoder and decoder neural networks,
                                objective function ELBO), trains the model and carries out
                                downstream tasks: generates latent space z, generates new images
                                sampling z from prior distribution p(z) and generates new images 
                                sampling z from posterior distribution q(z|x). This VAE model works 
                                for input data consisting of colored or black and white images,
                                the image type ('color' or 'bw') should be specified when instance
                                of VAE is constructed.                                
                                                        
                                Methods:
                                1)  call: Takes the input data and runs it through the VAE 
                                    in order to build the objective function ELBO.
                                    @param x = input data to be used to train the VAE
                                    @return -elbo = (-1) * the objective function
                                                            
                                2)  train: Uses the input data and the objective function from
                                    call (1) to train the VAE.
                                    @param x = data to be used to train the VAE
                                    @param optimizer = optimizer to be used for the VAE training                     
                                    @return training loss
                                                            
                                3)  generate_latent_space_z: Uses the inputed data and the
                                    trained VAE model to generate latent space z.                          
                                    @param x = data to be used to generate the latent space z
                                    @return multidimensional latent space z
                                                        
                                4)  generate_new_images_from_prior: Uses the trained VAE model to
                                    generate new images from prior distribution of z (specifcally
                                    multivariate Gaussian). 
                                    @param number_of_images to be generated (default = 100)
                                    @param random_sampling = boolean parameter, if True the
                                    output is randomly sampled from the VAE decoder output
                                    distribution, if False the output is the expectation
                                    resulting in sharper images (default False)
                                    @return images VAE was able to generate from prior
                                
                                5)  generate_new_images_from_posterior: Uses the trained VAE model
                                    to generate images from the inputed data. If training was sucesful
                                    it should more or less return the inputed images.
                                    @param random_sampling = boolean parameter, if True the
                                    output is randomly sampled from the VAE decoder output
                                    distribution, if False the output is the expectation
                                    resulting in sharper images (default False)
                                    @return images VAE was able to generate from posterior
                                '''),

                                epilog=textwrap.dedent('''\
                                    Subclass VAE Usage
                                -----------------------------------
                                For usage see file 'train_vae.py'
                                ''')
                )



class VAE(tf.keras.Model):

    # need all the encoder/decoder params to be input here too since I'm constructing encoder/decoder in the vae constructor here
    def __init__(self, img_type, latent_dim = None, units = None, activation = 'relu',  # BiCoder params
                input_shape = None,                                                     # extra Encoder params
                output_dim = 28*28, target_shape = (4,4,128), channel_out = 3,          # extra Decoder params
                kernel_size = 3, strides = 2, filters = 32):                            # optional BiCoder params only for color (but no * cause everything before this "required" for VAE)

        # tf.keras.Model constructor so that VAE can inherit from there
        super().__init__()

        # construct encoder and decoder objects -> stable neural netoworks over life of VAE instance
        self.encoder = Encoder(img_type, latent_dim , units, activation, input_shape,
                                kernel_size = kernel_size, strides = strides, filters = filters)
        self.decoder = Decoder(img_type, latent_dim, units, activation, output_dim, target_shape, channel_out, 
                                kernel_size = kernel_size, strides = strides, filters = filters)

        # instance variable for the docstring
        self.parser_VAE = parser

    # Prints the nicely formatted docstring
    @property
    def help(self):
        self.parser_VAE.print_help()

    # private method, mu and log_var from encoder
    def _kl_divergence(self, mu, log_var):
        return 0.5 * tf.reduce_sum(tf.square(mu) + tf.exp(log_var) - log_var - 1, axis=-1) 

    # private method, mu, log_sigma from decoder (decoder already returns log_sigma even if we have fixed std)
    def _log_diag_mvn(self, x, mu, log_sigma):
        sum_axes = tf.range(1, tf.rank(mu))
        k = tf.cast(tf.reduce_prod(tf.shape(mu)[1:]), x.dtype)
        logp =  - 0.5 * k * tf.math.log(2*np.pi) \
                - log_sigma \
                - 0.5*tf.reduce_sum(tf.square(x - mu)/tf.math.exp(2.*log_sigma),axis=sum_axes)
        return logp

    # call function that builds the objective f to be optimized using only data as input
    def call(self, x):

        # get encoder output, calculate z from it, pass the z into decoder to get decoder output
        encoder_mu, log_var = self.encoder(x)
        z = self.encoder.calculate_z(encoder_mu, log_var)
        decoder_mu, log_sigma = self.decoder(z)

        # build the objective f (elbo) from the encoder/decoder inputs 
        kl_div = self._kl_divergence(encoder_mu, log_var)
        log_diag_mnv = self._log_diag_mvn(x, decoder_mu, log_sigma)
        elbo = log_diag_mnv - kl_div
        
        # return the objective f to be maximized 
        return - elbo    # -elbo since we wanna min elbo (loss) but the machine runs optimization always as max so flip sign
    
    # basically just Rogelio's train method, 
    @tf.function
    @LogTraining    # but we add our custom decorator LogTraining to log the training loss
    def train(self, x, optimizer):
        with tf.GradientTape() as tape:
            loss = self.call(x)
        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return loss

    # we use our calculate_z encoder method to generate latent space z from the data
    def generate_latent_space_z(self, x):
        encoder_mu, log_var = self.encoder(x)
        z = self.encoder.calculate_z(encoder_mu, log_var)  
        return z

    # we sample z from prior (multivariate Gaussian) and then pass this random z to the trained decoder to generate new images
    # using 100 images as the default since we plotting grid of 100 typically in testing
    def generate_new_images_from_prior(self, number_of_images = 100, random_sampling = False):
        
        # the dimensions of z must match my usual image vector size so that encoder works (it's set up for that size) 
        z = tf.random.normal((number_of_images, self.encoder._latent_dim))
        decoder_mu, log_sigma = self.decoder(z)

        # sample from the whole dist we get from the decoder
        if random_sampling:
            return self.decoder.get_x(decoder_mu, log_sigma)
        
        # or get directly the expectation for sharper image == default
        else:
            return decoder_mu


    # pass (test) data to our trained encoder, get z from the results of that and then pass 
    # this z to our trained decoder to hopefully get back the original pictures from the input data
    def generate_new_images_from_posterior(self, x, random_sampling = False):

        # run the data through the whole VAE
        encoder_mu, log_var = self.encoder(x)
        z = self.encoder.calculate_z(encoder_mu, log_var) 
        decoder_mu, log_sigma = self.decoder(z)

        # randomly sample from the decoder dist
        if random_sampling:
            return self.decoder.get_x(decoder_mu, log_sigma)
        
        # or get directly the expectation for sharper image == default
        else:
            return decoder_mu

