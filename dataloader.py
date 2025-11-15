import os
import subprocess
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

'''

change the architecture to be more similar to bicoder? with the img_type input?

'''

# -------------------------------------- superclass class DataLoader ------------------------------------------

class DataLoader:

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

        # allow_pickle = True since the color dataset threw err 'Cannot load file containing pickled data when allow_pickle=False' due to it being .pkl
        data = np.load(file_name, allow_pickle = True)

        # return data in normal np format so that this parent class DataLoader can be used for labels
        # subclasses modify this and return the preprocessed data in the tf format they should be
        return data

    # summary method -> download the data, preprocess and return
    def load_data(self, url):
        data_file = self.download(url)
        preprocessed_data = self.prep_data(data_file)
        return preprocessed_data


# -------------------------------- subclass class Color_DataLoader ----------------------------------------


class Color_DataLoader(DataLoader):

    # color has instance variable for version = can have multiple color dataloaders at once for different versions
    def __init__(self, version = 'm0'):

        # extra instance variable just for this subclass + active debugging of input
        possible_versions = ['m0', 'm1', 'm2', 'm3', 'm4']
        assert version in possible_versions, 'Color version you have picked is not valid. It must be one of the following options: {}'.format(possible_versions)
        self._key = version

    # isolate only 1 of the 5 dicts using the key user chose, default = 'm0' (specified in constructor)
    def prep_data(self, data_file):
        data = super().prep_data(data_file)
        preprocessed_data = data[self._key]

        # return preprocessed_data in the tf format 
        return tf.data.Dataset.from_tensor_slices(preprocessed_data)


# ----------------------------------- subclass class BW_DataLoader ------------------------------------


class BW_DataLoader(DataLoader):

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
