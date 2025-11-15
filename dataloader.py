import os
import subprocess
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

import argparse
import textwrap
parser = argparse.ArgumentParser(prog='Superclass DataLoader',
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=textwrap.dedent('''\
                                        Superclass Data Loader
                                ----------------------------------------
                                Data Loader that downloads datasets from online links,
                                loads the data from their file and returns the prepared data. 
                                Made particularly for the mnist datasets.
                                                            
                                Methods:
                                1)  download: Extracts file name from the url, checks whether the file 
                                    we are trying to download already exists and if it does, skipping 
                                    redownloading again, returns the file name. If the file has not yet
                                    been downloaded, downloads the file using wget and returns the file name.
                                    @param url = the url we want to download the dataset from
                                    @return the file name of the dataset we wanted to download
                                                            
                                2)  prep_data: Loads data from a file.
                                    @param file_name = file name of where the data is stored
                                    @return data from the file in numpy format
                                                            
                                3)  load_data: Summary method that first downloads the data from a url 
                                    according to the download method (1) and then loads the downaloded
                                    data according to prep_data (2).
                                    @param url = the url we want to download the dataset from
                                    @return downloaded data in numpy format  
                                '''),

                                epilog=textwrap.dedent('''\
                                    Superclass DataLoader Usage
                                -----------------------------------
                                dataloader = DataLoader()                                           # initialize an object
                                file_name = dataloader.download('https://www.dropbox.com/...')      # download data from the dropbox link and get the file name 
                                data = dataloader.prep_data('mnist_color.pkl')                      # load data from file 'mnist_color.pkl' into variable data
                                data = dataloader.load_data('https://www.dropbox.com/...')          # download data from the dropbox link and then load them into variable data
                                ''')
                )


# -------------------------------------- superclass class DataLoader ------------------------------------------

class DataLoader:

    # construct a DataLoader
    def __init__(self):
        self.parser_superclass = parser

    # Prints the nicely formatted docstring
    @property
    def help(self):
        self.parser_superclass.print_help()

    # use wget to download data from a given link
    # 1st check whether it's already downloaded and if yes don't redownload, just return already existing file name
    def download(self, url):
        
        # this might be a bit too specific, working only for this application 
        # but here the pattern of the urls is always the same = .../file_name?... 
        # so extract the file_name by splitting the string to keep only part between the only ? and the 1st / before that
        file_name = url.split('?')[0].split('/')[-1]

        if os.path.exists(file_name):
            print('File {} has already been downloaded. The existing version will be used. ' \
            'If you want to redownload the file, you first need to delete the preexisting one.'.format(file_name))
            return file_name
        
        subprocess.run(['wget', '-O', file_name, url])
        return file_name

    # min implementation common across all subclasses -> load file
    # will be extended for subclasses to do the necessary preprocessing steps
    def prep_data(self, file_name):

        # allow_pickle = True since the color dataset threw err 'Cannot load file containing 
        # pickled data when allow_pickle=False' due to it being .pkl
        data = np.load(file_name, allow_pickle = True)

        # return data in normal np format so that this parent class DataLoader can be used for labels
        # subclasses modify this and return the preprocessed data in the tf format they should be
        return data

    # summary method -> download the data, preprocess and return
    def load_data(self, url):
        data_file = self.download(url)
        preprocessed_data = self.prep_data(data_file)
        return preprocessed_data
    

loader = DataLoader()
loader.help