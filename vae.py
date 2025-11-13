import tensorflow as tf
import numpy as np

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4) 


# import BiCoder stuff

'''
but all the outputs (mu, log_var, z...) will be instance variables here for sure but idk if also in bicoder, 
maybe not even here??? gotta think abt this a bit more how it works with the train



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

    def __init__(self):

        # tf.keras.Model constructor so that VAE can inherit from there
        super().__init__()

    # mu and log_var from encoder
    def kl_divergence(self, mu, log_var):
        return 0.5 * tf.reduce_sum(tf.square(mu) + tf.exp(log_var) - log_var - 1, axis=-1) 

    # mu, log_sigma from decoder, but our sigma fixed so gotta take log 1st (maybe output log_sigma already from decoder? wouldn't that be how the output is if actually trained?) 
    def log_diag_mvn(self, x, mu, log_sigma):
        sum_axes = tf.range(1, tf.rank(mu))
        k = tf.cast(tf.reduce_prod(tf.shape(mu)[1:]), x.dtype)
        logp =  - 0.5 * k * tf.math.log(2*np.pi) \
                - log_sigma \
                - 0.5*tf.reduce_sum(tf.square(x - mu)/tf.math.exp(2.*log_sigma),axis=sum_axes)
        return logp

    # should return loss based on Rogelio's train?
    # if I chnage the bicoder methods to call then I can just dircetly use it as decoder(x) etc -> implement in bicoder and then change here too
    def call(self, x):
        mu, log_var = encoder.get_network_output(x)
        z = encoder.calculate_z(mu, log_var)
        mu_of_x, std = decoder.get_network_output(z)
        kl_div = self.kl_divergence(mu, log_var)
        log_diag_mnv = self.log_diag_mvn(x, mu, log_sigma)
        elbo = log_diag_mnv - kl_div
        loss = - elbo
        return loss
    
    # here the returned loss is for monitoring of the training (how many epochs u need) -> from Rogelio
    # but also vae should have instance variable vae_loss I guess
    @tf.function 
    def train(self, x):
        with tf.GradientTape() as tape:
            loss = self.call(x)
        gradients = tape.gradient(self.vae_loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return loss


    def generate_new_images_from_posterior(self):
        '''
        x = decoder.get_x(mu_of_x)
        '''