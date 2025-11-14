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


'''


# setup the parser
parser = argparse.ArgumentParser(description = 'Training and testing my VAE')
parser.add_argument('--dset', type = str, choices = ['mnist_bw', 'mnist_color'])
parser.add_argument('--epochs', type = int, default = 50)       # Rogelio used 50 so default
parser.add_argument('--visualize_latent', action = 'store_true')
parser.add_argument('--generate_from_prior', action = 'store_true')
parser.add_argument('--generate_from_posterior', action = 'store_true')
args = parser.parse_args()


# make a mapping for the datasets and the urls needed for their testing, training, label data, plus add bw vs color as img_type
# dataset name = key of the main dict, then smaller dict inside saying what each urls is
datasets_urls = {'mnist_bw' : {
                                'train' : 'https://www.dropbox.com/scl/fi/fjye8km5530t9981ulrll/mnist_bw.npy?rlkey=ou7nt8t88wx1z38nodjjx6lch&st=5swdpnbr&dl=0',
                                'test' : 'https://www.dropbox.com/scl/fi/dj8vbkfpf5ey523z6ro43/mnist_bw_te.npy?rlkey=5msedqw3dhv0s8za976qlaoir&st=nmu00cvk&dl=0',
                                'labels' : 'https://www.dropbox.com/scl/fi/8kmcsy9otcxg8dbi5cqd4/mnist_bw_y_te.npy?rlkey=atou1x07fnna5sgu6vrrgt9j1&st=m05mfkwb&dl=0'},
                
                'mnist_color' : {
                                'train' : 'https://www.dropbox.com/scl/fi/w7hjg8ucehnjfv1re5wzm/mnist_color.pkl?rlkey=ya9cpgr2chxt017c4lg52yqs9&st=ev984mfc&dl=0',
                                'test' : 'https://www.dropbox.com/scl/fi/w08xctj7iou6lqvdkdtzh/mnist_color_te.pkl?rlkey=xntuty30shu76kazwhb440abj&st=u0hd2nym&dl=0',
                                'labels' : 'https://www.dropbox.com/scl/fi/fkf20sjci5ojhuftc0ro0/mnist_color_y_te.npy?rlkey=fshs83hd5pvo81ag3z209tf6v&st=99z1o18q&dl=0'}}

# from the user inputed dset extract bw vs color
img_type = args.dset.split('_')[-1]

# now I want to extract only the dict for the dataset user picked
urls = datasets_urls[img_type]






optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4) 

