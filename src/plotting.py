#!/usr/bin/env python

from matplotlib import pyplot

import numpy
import theano

import data
import pickle

from pprint import pprint

from aux import data_path, figure_path, script_directory

def main():
    
    # Setup
    
    downsampling_factor = 2
    latent_size = 2
    N_epochs = 20
    
    # Fix random seed for reproducibility
    numpy.random.seed(1234)
    
    # Data
    
    specifications = "ds{}_l{}_e{}".format(downsampling_factor, latent_size, N_epochs)
    file_name = "results_" + specifications + ".pkl"
    
    with open(data_path(file_name), "r") as f:
        setup_and_results = pickle.load(f)
    
    setup = setup_and_results["setup"]
    C, H, W = setup["image size"]
    h = H / downsampling_factor
    w = W / downsampling_factor
    
    pprint(setup)
    print
    
    results = setup_and_results["results"]
    
    ## Learning curves
    
    learning_curve = results["learning curve"]
    epochs = learning_curve["epochs"]
    cost_train = learning_curve["training cost function"]
    cost_test = learning_curve["test cost function"]
    
    figure = pyplot.figure()
    axis = figure.add_subplot(1, 1, 1)
    
    axis.plot(epochs, cost_train, label = 'Training data')
    axis.plot(epochs, cost_test, label = 'Test data')
    
    pyplot.legend(loc = "best")
    
    axis.set_ylabel("log-likelihood")
    axis.set_xlabel('Epochs')
    
    plot_name = figure_path("learning_curve_" + specifications + ".pdf")
    pyplot.savefig(plot_name)
    print("Learning curve saved as {}.".format(plot_name))
    
    ## Reconstruction
    
    reconstructions = results["reconstructions"]
    
    x = reconstructions["originals"]
    x_reconstructed = reconstructions["reconstructions"]
    
    N_reconstructions = len(x_reconstructed)
    
    image = numpy.zeros((H * 2, W * N_reconstructions))

    for i in range(N_reconstructions):
        x_a, x_b = 0 * H, 1 * H
        x_recon_a, x_recon_b = 1 * H, 2 * H
        y_a, y_b = i * W, (i + 1) * W
        image_i = x[i].reshape((H, W))
        image_i_reconstructed = x_reconstructed[i].reshape((H, W))
        image[x_a: x_b, y_a: y_b] = image_i
        image[x_recon_a: x_recon_b, y_a: y_b] = image_i_reconstructed

    figure = pyplot.figure()
    axis = figure.add_subplot(1, 1, 1)

    axis.imshow(image, cmap = 'gray')

    axis.set_xticks(numpy.array([]))
    axis.set_yticks(numpy.array([]))

    plot_name = figure_path("reconstructions_" + specifications + ".pdf")
    pyplot.savefig(plot_name)
    print("Reconstructions saved as {}.".format(plot_name))

    ## Manifold
    
    samples = results["manifold"]["samples"]
    
    if latent_size == 2:
    
        idx = 0
        canvas = numpy.zeros((H * 20, 20 * W))
        for i in range(20):
            for j in range(20):
                canvas[i*H: (i + 1) * H, j * W: (j + 1) * W] = samples[idx].reshape((H, W))
                idx += 1
        
        figure = pyplot.figure()
        axis = figure.add_subplot(1, 1, 1)
        
        pyplot.imshow(canvas, cmap = "binary")
        
        pyplot.title('MNIST handwritten digits')
        axis.set_xticks(numpy.array([]))
        axis.set_yticks(numpy.array([]))
        
        plot_name = figure_path("manifold_" + specifications + ".pdf")
        pyplot.savefig(plot_name)
        print("Manifold saved as {}.".format(plot_name))
    
    ## Reconstructions of homemade numbers
    
    reconstructions_homemade = results["reconstructed homemade numbers"]
    
    if downsampling_factor == 2:
        
        x = reconstructions_homemade["originals"]
        x_reconstructed = reconstructions_homemade["reconstructions"]
    
        N_reconstructions = len(x_reconstructed)
    
        image = numpy.zeros((H * 2, W * N_reconstructions))
    
        for i in range(N_reconstructions):
            x_a, x_b = 0 * H + h/2, 1 * H - h/2
            y_a, y_b = i * W + w/2, (i + 1) * W - w/2
            x_recon_a, x_recon_b = 1 * H, 2 * H
            y_recon_a, y_recon_b = i * W, (i + 1) * W
            image_i = x[i].reshape((h, w))
            image_i_reconstructed = x_reconstructed[i].reshape((H, W))
            image[x_a: x_b, y_a: y_b] = image_i
            image[x_recon_a: x_recon_b, y_recon_a: y_recon_b] = image_i_reconstructed
    
        figure = pyplot.figure()
        axis = figure.add_subplot(1, 1, 1)
    
        axis.imshow(image, cmap = 'gray')
    
        axis.set_xticks(numpy.array([]))
        axis.set_yticks(numpy.array([]))
    
        plot_name = figure_path("reconstructions_homemade_" + specifications + ".pdf")
        pyplot.savefig(plot_name)
        print("Reconstructions of homemade numbers saved as {}.".format(plot_name))

if __name__ == '__main__':
    script_directory()
    main()
