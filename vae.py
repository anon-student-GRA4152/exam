import tensorflow as tf

# import BiCoder stuff

'''
but all the outputs (mu, log_var, z...) will be instance variables here for sure but idk if also in bicoder

'''



class VAE(tf.keras.Model):



    def call(self, x):
        '''
        mu, log_var = encoder.get_network_output(x)
        z = encoder.calculate_z(mu, log_var)
        mu_of_x = decoder.get_network_output(z)
        '''
    
    def generate_new_images_from_posterior(self):
        '''
        x = decoder.get_x(mu_of_x)
        '''