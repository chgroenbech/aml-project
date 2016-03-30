#!/usr/bin/env python3

import data
import numpy

from matplotlib import pyplot as plt

from aux import fig_path, script_directory

def main():
    
    # Data
    
    X_train, y_train, X_val, y_val, X_test, y_test = data.load()
    
    C = len(X_train[0]) # number of channels
    K = 10 # number of classes
    D = len(X_train[0, 0]) # dimensions: D^2
    
    # Plotting
    
    plt.imshow(X_train[0, 0, :, :], cmap = "binary")
    plt.savefig(fig_path("sample.pdf"))
    
    # Setup
    
    M = 100 # mini-batch size
    H = 200 # number of hidden units
    

if __name__ == '__main__':
    script_directory()
    main()
