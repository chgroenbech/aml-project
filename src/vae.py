#!/usr/bin/env python3

import numpy
from scipy.io import loadmat

from matplotlib import pyplot as plt

from aux import data_path, fig_path, script_directory

def main():
    
    # Data
    
    data = Data(data_path("mnist_all.mat"))
    
    # Sample
    
    sample = data.sample(0, 0)
    
    # Plotting
    
    plt.imshow(data.plot(sample), cmap = "binary")
    plt.savefig(fig_path("sample.pdf"))

class Data(object):
    def __init__(self, data_file_path):
        super(Data, self).__init__()
        self.data = loadmat(data_file_path)
        self.D = numpy.sqrt(len(self.data["train0"][0]))
    
    def sample(self, n, i, colour = "B&W", mode = None):
        sample = self.data["train{:d}".format(n)][0]
        
        if colour == "B&W":
            # Use matplotlib.colors.Normalize
            sample = (sample - numpy.min(sample))/numpy.max(sample)
            sample = sample.astype(bool).astype(int)
        
        return sample
    
    def plot(self, sample):
        return sample.reshape(self.D, self.D)
    

if __name__ == '__main__':
    script_directory()
    main()
