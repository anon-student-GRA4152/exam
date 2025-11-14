import tensorflow as tf
import argparse

from vae import VAE
from dataloader import DataLoader, BW_DataLoader, Color_DataLoader



'''
- load data 
    - make some url-dataset mapping so that I can load the correct datasets (test, train, labels) by just passing in dataset name
- make VAE object -> color vs bw based on data
- train VAE
- do the downstream tasks (with argparse add that just 1 will be carried out depending on what the user inputs)
    - generate latent space z
    - generate new images from prior
    - generate new images from posterior

- visualize the output of the downstream tasks (using f Rogelio provided for the grid stuff)


add the argparse functionality


'''

# data loading 

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4) 

