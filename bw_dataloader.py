# import packages
import os
import subprocess
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

# import my own module
from dataloader import DataLoader

# nice docstring
import argparse
import textwrap
parser = argparse.ArgumentParser(prog='Subclass BW_DataLoader',
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=textwrap.dedent('''\
                                        Subclass Data Loader for BW Images
                                -----------------------------------------------
                                Data Loader that downloads datasets with black and white images from 
                                online links, loads the data from their file, preprocesses them with 
                                scaling and vectorization and returns the preprocessed data. 
                                Made particularly for the mnist black and white datasets.
                                                            
                                Methods:
                                Inherited from superclass DataLoader (methods download and load_data)
                                Method prep_data extending superclass as described below
                                                            
                                1)  prep_data: Loads data from a file (this behaviour inherited from 
                                    superclass DataLoader). Scales (dividing by 255) and vectorizes 
                                    (flatens) them. Returns the preprocessed data.
                                    @param file_name = file name of where the bw images data is stored
                                    @return preprocessed (scaled and vectorized) black and white image 
                                    data from the file in format tf.data.Dataset.from_tensor_slices() 
                                                        
                                '''),

                                epilog=textwrap.dedent('''\
                                        Subclass BW_DataLoader Usage
                                ---------------------------------------------
                                bw_dataloader = BW_DataLoader()                                       # initialize an object
                                file_name = bw_dataloader.download('https://www.dropbox.com/...')     # download data from the dropbox link and get the file name 
                                data = bw_dataloader.prep_data('mnist_bw.npy')                        # preprocess and load data from 'mnist_bw.npy' into data (as the tf object)
                                data = bw_dataloader.load_data('https://www.dropbox.com/...')         # download data from the dropbox link and then load the preprocessed data
                                                                                                        into variable data
                                ''')
                )


# ----------------------------------- subclass class BW_DataLoader ------------------------------------


class BW_DataLoader(DataLoader):

    def __init__(self):
        super().__init__()

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

    # preprocessing: 1) scaling, 2) vectorization, incl check that input img dimension as expected (28x28)
    def prep_data(self, data_file):
        data = super().prep_data(data_file)

        # data should have shape: (number of images, 28, 28) since image dimensions are supposed to be 28x28
        # the number of images is flexible so we only want to make sure the image dimensions are as expected
        assert data.shape[1:] == (28, 28), 'You are trying to load data consisting of images with unexpected dimensions. The expected image dimensions: 28x28.'

        # scaling
        data = data.astype(np.float32)
        data /= 255

        # vectorization -> each image should be flattened therefore the resulting shape should be (number of images, 28*28)
        img_dimensions = 28 * 28
        img_number = data.shape[0]
        preprocessed_data = data.reshape(img_number, img_dimensions)

        # return preprocessed_data in the tf format 
        return tf.data.Dataset.from_tensor_slices(preprocessed_data)