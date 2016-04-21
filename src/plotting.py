#!/usr/bin/env python3

from matplotlib import pyplot

from scipy.stats import norm
from scipy.stats import invgauss

from aux import data_path, fig_path, script_directory

def main():
    
    # Data
    
    data = loadmat(data_path("mnist_all.mat"))
    
    print(sorted(data.keys()))
    
    D = sqrt(len(data["train0"][0]))
    
    # Reshaping data
    
    sample = data["train0"][0].reshape(D, D)
    
    # Plotting
    
    plt.imshow(sample)
    plt.savefig(fig_path("sample.pdf"))

if __name__ == '__main__':
    script_directory()
    main()
