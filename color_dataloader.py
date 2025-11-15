import os
import subprocess
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

from dataloader import DataLoader

import argparse
import textwrap
parser = argparse.ArgumentParser(prog='Subclass Color_DataLoader',
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=textwrap.dedent('''\
                                        Subclass Data Loader for Color Images
                                --------------------------------------------------
                                Data Loader that downloads datasets with colored images from 
                                online links, loads the data from their file, unzips them, extracts
                                only a single version of the data according to user instructions and   
                                returns the prepared data. 
                                Made particularly for the mnist color datasets.
                                                            
                                Methods:
                                Inherited from superclass DataLoader (methods download and load_data)
                                Method prep_data extending superclass as decsribed below
                                                            
                                1)  prep_data: Loads data from a file (this behaviour inherited from 
                                    superclass DataLoader). Extracts a single version (single color palet)
                                    of the colored images from the bulk of data loaded from the file. Which
                                    version it is depends on the user input when constructing an instance of 
                                    the Color_DataLoader. Returns the prepared data.
                                    @param file_name = file name of where the colored images data is stored
                                    @return one version of the colored images data from the file in format
                                    tf.data.Dataset.from_tensor_slices() 
                                                        
                                '''),

                                epilog=textwrap.dedent('''\
                                        Subclass Color_DataLoader Usage
                                ---------------------------------------------
                                color_dataloader = Color_DataLoader('m1')                                # initialize an object picking color version m1 
                                file_name = color_dataloader.download('https://www.dropbox.com/...')     # download data from the dropbox link and get the file name 
                                data = color_dataloader.prep_data('mnist_color.pkl')                     # load color version m1 from 'mnist_color.pkl' into data (as the tf object)
                                data = color_dataloader.load_data('https://www.dropbox.com/...')         # download data from the dropbox link and then load the m1 version
                                                                                                            into variable data
                                ''')
                )

# -------------------------------- subclass class Color_DataLoader ----------------------------------------

class Color_DataLoader(DataLoader):

    # color has instance variable for version = can have multiple color dataloaders at once for different versions
    def __init__(self, version = 'm0'):

        super().__init__()

        # extra instance variable just for this subclass + active debugging of input
        possible_versions = ['m0', 'm1', 'm2', 'm3', 'm4']
        assert version in possible_versions, 'Color version you have picked is not valid. It must be one of the following options: {}'.format(possible_versions)
        self._key = version

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

    # isolate only 1 of the 5 dicts using the key user chose, default = 'm0' (specified in constructor)
    def prep_data(self, file_name):
        data = super().prep_data(file_name)
        preprocessed_data = data[self._key]

        # return preprocessed_data in the tf format 
        return tf.data.Dataset.from_tensor_slices(preprocessed_data)

color = Color_DataLoader('m4')
color.help