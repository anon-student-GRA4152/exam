import tensorflow as tf
import wget
import numpy as np
import os


class DataLoader:

    # use wget to download data from a given link, but 1st check whether it's already downloaded and then don't redownload, just return the already existing file name
    def download(self, url):
        # there is always only 1 ? in the url, right after the file name so can just find the ? and extract everything before it and up until '/' using string operations
        # extract everything before ? so the modified str will end with the file name, then split based on / and extract the last element -> file name
        file_name = url.split('?')[0].split('/')[-1]

        if os.path.exists(file_name):
            print('File {} has already been downloaded. The existing version will be used. If you want to redownload the file, you first need to delete the preexisting one.'.format(file_name))
            return file_name
        else:
            return wget.download(url)

    # min implementation common across all subclasses -> load file, will be extended for subclasses to do the necessary preprocessing steps
    def prep_data(self, data_file):
        # for the color data set I've got error saying Cannot load file containing pickled data when allow_pickle=False so I've set it to True
        data = np.load(data_file, allow_pickle = True)
        return data

    # summary method -> download the data, preprocess and return in the final tf format
    def load_data(self, url):
        data_file = self.download(url)
        preprocessed_data = self.prep_data(data_file)
        return tf.data.Dataset.from_tensor_slices(preprocessed_data)


class Color_DataLoader(DataLoader):

    # the color has instance variable for version, so that we can have multiple color object instances at once for different versions + easier implementation
    def __init__(self, version = 'm0'):

        # extra instance variable just for this subclass + active debugging of input
        possible_versions = ['m0', 'm1', 'm2', 'm3', 'm4']
        assert version in possible_versions, 'Color version you have picked is not valid. It must be one of the following options: {}'.format(possible_versions)
        self._key = version

    # isolate only 1 of the 5 dictionaires using the key user chose, default = 'm0' version (specified above)
    def prep_data(self, data_file):
        data = super().prep_data(data_file)
        preprocessed_data = data[self._key]

        return preprocessed_data

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

        return preprocessed_data


# color_dataloader = Color_DataLoader('m2')
# mnist_color_train = color_dataloader.load_data('https://www.dropbox.com/scl/fi/w7hjg8ucehnjfv1re5wzm/mnist_color.pkl?rlkey=ya9cpgr2chxt017c4lg52yqs9&st=ev984mfc&dl=0')
# print(mnist_color_train.element_spec)

# bw_dataloader = BW_DataLoader()
# mnist_bw_train = bw_dataloader.load_data('https://www.dropbox.com/scl/fi/fjye8km5530t9981ulrll/mnist_bw.npy?rlkey=ou7nt8t88wx1z38nodjjx6lch&st=5swdpnbr&dl=0')
# print(mnist_bw_train.element_spec)

# labels_dataloader = DataLoader()
# mnist_bw_labels = labels_dataloader.load_data('https://www.dropbox.com/scl/fi/8kmcsy9otcxg8dbi5cqd4/mnist_bw_y_te.npy?rlkey=atou1x07fnna5sgu6vrrgt9j1&st=m05mfkwb&dl=0')
# print(mnist_bw_labels.element_spec)

