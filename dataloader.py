import tensorflow as tf


class DataLoader:
    def __init__(self, url):
        # def all instnce variables
        self._url = url

    def download(self):
        # use wget to download data from a given link
        pass

    def prep_data(self):
        # abstract method
        raise NotImplementedError

    def load_into_tf_dataset(self):
        # load preprocessed data into tf object that can be directly used in VAE
        pass


class Color_DataLoader(DataLoader):
    def __init__(self, url, version = 'm0'):
        # get same instance variables as superclass
        super().__init__(url)

        # implement extra instance variable just for this extension, add active debugging of input
        possible_versions = ['m0', 'm1', 'm2', 'm3', 'm4']
        assert version in possible_versions, 'Color version you have picked is not valid. It must be one of the following options: {}'.format(possible_versions)
        self._key = version

    def prep_data(self):
        # unzip the data -> isolate only 1 of the 5 dictionaires
        pass


class BW_DataLoader(DataLoader):
    def __init__(self, url):
        super().__init__(url)

    def prep_data(self):
        # preprocessing: 1) scaling, 2) vectorization
        pass