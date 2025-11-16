# import packages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import argparse
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import json

# import my own modules
from dataloader import DataLoader
from bw_dataloader import BW_DataLoader
from color_dataloader import Color_DataLoader
from vae import VAE

'''
This is a testing file for my project. 
It allows users to interactively in the command line pick which data set 
they want to use (black and white images from mnist_bw or colored images
from mnist_color). The data from the chosen dataset is downloaded and loaded
into the tensorflow object. Then the script constructs my VAE and trains it.
The training is done for 50 epochs by default but the user can also choose
this parameter. Lastly the script carries out downstream tasks. The user can
choose whether any of them should be executed with the input parameters. The
tasks that can be done are 1) Generating and visualizing the latent space z, 
2) Generating new images by sampling z from the prior distribution p(z), and
3) Generating new images by sampling z from the posterior distribution q(z|x).
The outputs of the downstream tasks are saved as pdf documents.
'''
# ----------------- function to plot grids -----------------------

# function from Rogelio for plotting grids
def plot_grid(images, dataset_name, downstream_task, N=10, C=10, figsize=(24., 28.)):
    fig = plt.figure(figsize=figsize)
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                    nrows_ncols=(N, C),  
                    axes_pad=0,  # pad between Axes in inch.
                    )
    for ax, im in zip(grid, images):
        # Iterating over the grid returns the Axes.
        ax.imshow(im)
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(dataset_name + '-' + downstream_task + '.pdf')
    plt.close()


# ----------------- user input parser -----------------------

# setup the parser
parser = argparse.ArgumentParser(description = 'Training and testing my VAE')
parser.add_argument('--dset', type = str, choices = ['mnist_bw', 'mnist_color'])
parser.add_argument('--epochs', type = int, default = 50)       # Rogelio used 50 so default
parser.add_argument('--visualize_latent', action = 'store_true')
parser.add_argument('--generate_from_prior', action = 'store_true')
parser.add_argument('--generate_from_posterior', action = 'store_true')
args = parser.parse_args()

# save the user input arguments in a dictionary in a separte file to have a log of what has been already called
with open ('commandline_args.txt', 'a') as f:
    json.dump(args.__dict__, f, indent = 2)

# ----------------- loading data -----------------------

# make a mapping for the datasets and the urls needed for their testing, training, label data
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
urls = datasets_urls[args.dset]

# construct correct data loader
if img_type == 'bw':
    data_loader = BW_DataLoader() 
elif img_type == 'color':
    data_loader = Color_DataLoader('m2')

# construct data loader for labels
labels_data_loader = DataLoader()

# load train, test data and labels, with active debugging in case download or loading fails
try:
    x_train = data_loader.load_data(urls['train'])
    x_test = data_loader.load_data(urls['test'])
    labels = labels_data_loader.load_data(urls['labels'])
    print('Data from {} dataset has been sucesfully loaded.'.format(args.dset))
except:
    print('Error loading the {} data.'.format(args.dset))

# -------- VAE construction and training -------------------

# # construct the VAE object
# vae = VAE(img_type)

# # 1. Train the VAE model
# optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4) 
# batch_size = 100

# for epoch in range(args.epochs):
#     for batch_x in x_train.batch(batch_size):
#         vae.train(batch_x, optimizer)

# ---------- downstream tasks based on which ones the user picked ------------

# 2. Generate and visualize the latent space z
if args.visualize_latent:
    
    # fixing the 'Inputs to a layer should be tensors' error
    x_test_array = np.array(list(x_test.as_numpy_iterator()))

    # generate z
    z = vae.generate_latent_space_z(x_test_array)

    # reduce z to 2D (for scatter) using TSNE
    tsne = TSNE(n_components = 2)
    z_2d = tsne.fit_transform(z)

    # plot 2D z as scatterplot, using the labels to color the scatters
    fig = plt.figure(figsize=(24., 28.))
    scatter = plt.scatter(z_2d[:, 0], z_2d[:, 1], c = labels, cmap = 'tab10')
    plt.savefig(args.dset + '-generate_latent_space_z' + '.pdf')
    plt.close()

# ----------------------------------------------------------
# 3. Generate new images sampling z from the prior p(z)
if args.generate_from_prior:

    # as a default we sample the expectation (mu of x) for clearer pictures (random_sampling = False) 
    # and 100 images which is the default grid dimensions from Rogelio
    x_hat = vae.generate_new_images_from_prior(number_of_images = 100, random_sampling = False)

    # need to use the clipping as Rogelio said
    img = tf.clip_by_value(255*x_hat, clip_value_min=0, clip_value_max=255).numpy().astype(np.uint8)

    # fixing the shape error for bw, we have flattened it to be only a vector and now need to make it 28*28 again
    if img_type == 'bw':
        img = img.reshape(-1, 28, 28)

    plot_grid(img, args.dset, 'prior')

# -----------------------------------------------------------
# 4. Generate new images sampling z from the posterior q(z|x)
if args.generate_from_posterior:

    # fixing the Inputs to a layer should be tensors error
    x_test_array = np.array(list(x_test.as_numpy_iterator()))

    # as a default we sample the expectation (mu of x) for clearer pictures (random_sampling = False) 
    x_hat = vae.generate_new_images_from_posterior(x_test_array, random_sampling = False)

    # need to use the clipping as Rogelio said
    img = tf.clip_by_value(255*x_hat, clip_value_min=0, clip_value_max=255).numpy().astype(np.uint8)

    # fixing the shape error for bw, we have flattened it to be only a vector and now need to make it 28*28 again
    if img_type == 'bw':
        img = img.reshape(-1, 28, 28)
    
    plot_grid(img, args.dset, 'posterior')


