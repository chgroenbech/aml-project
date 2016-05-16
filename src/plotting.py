#!/usr/bin/env python

from __future__ import division

from matplotlib import pyplot

import numpy
import theano

import data
import pickle

from scipy.misc import imresize

from pprint import pprint

from aux import data_path, figure_path, script_directory

def main():
    
    # Setup
    
    downsampling_factors = [2, 4]
    latent_sizes = [2, 10, 30]
    N_epochs = 20
    
    # Fix random seed for reproducibility
    numpy.random.seed(1234)
    
    for latent_size in latent_sizes:
        
        learning_curves = []
        
        for downsampling_factor in downsampling_factors:
            
            # Data
    
            specifications = "ds{}_l{}_e{}".format(downsampling_factor, latent_size, N_epochs)
            file_name = "results_" + specifications + ".pkl"
    
            try:
                with open(data_path(file_name), "r") as f:
                    setup_and_results = pickle.load(f)
            except Exception as e:
                continue
    
            setup = setup_and_results["setup"]
            C, H, W = setup["image size"]
            h = int(H / downsampling_factor)
            w = int(W / downsampling_factor)
    
            pprint(setup)
            print
    
            results = setup_and_results["results"]
    
            ## Learning curves
    
            learning_curve = results["learning curve"]
            learning_curves.append({"data": learning_curve, "downsampling factor": downsampling_factor})
            
            epochs = learning_curve["epochs"]
            cost_train = learning_curve["training cost function"]
            cost_test = learning_curve["test cost function"]
    
            figure = pyplot.figure()
            axis = figure.add_subplot(1, 1, 1)
    
            axis.plot(epochs, cost_train, label = 'Training data')
            axis.plot(epochs, cost_test, label = 'Test data')
    
            pyplot.legend(loc = "best")
    
            axis.set_ylabel("Variational lower bound")
            axis.set_xlabel('Epochs')
    
            plot_name = figure_path("learning_curve_" + specifications + ".pdf")
            pyplot.savefig(plot_name)
            print("Learning curve saved as {}.".format(plot_name))
    
            ## Reconstruction
    
            reconstructions = results["reconstructions"]
    
            x = reconstructions["originals"]
            x_LR = reconstructions["downsampled"]
            x_reconstructed = reconstructions["reconstructions"]
    
            N_reconstructions = len(x_reconstructed)
    
            image = numpy.zeros((H * 4, W * N_reconstructions))

            for i in range(N_reconstructions):
            
                image_i = x[i].reshape((H, W))
                x_a, x_b = 0 * H, 1 * H
                y_a, y_b = i * W, (i + 1) * W
                image[x_a: x_b, y_a: y_b] = image_i
            
                image_i_LR = x_LR[i].reshape((h, w))
                x_LR_a, x_LR_b = int(1.5 * H - numpy.ceil(h/2.)), int(1.5 * H + numpy.floor(h/2))
                y_LR_a, y_LR_b = int((i + 0.5) * W - numpy.ceil(w/2)), int((i + 0.5) * W + numpy.floor(w/2))
                image[x_LR_a: x_LR_b, y_LR_a: y_LR_b] = image_i_LR
            
                image_i_upscaled = imresize(image_i_LR, (H, W), interp = "bicubic") / 255.
                x_up_a, x_up_b = 2 * H, 3 * H
                y_up_a, y_up_b = i * W, (i + 1) * W
                image[x_up_a: x_up_b, y_up_a: y_up_b] = image_i_upscaled
            
                image_i_reconstructed = x_reconstructed[i].reshape((H, W))
                x_recon_a, x_recon_b = 3 * H, 4 * H
                y_recon_a, y_recon_b = i * W, (i + 1) * W
                image[x_recon_a: x_recon_b, y_recon_a: y_recon_b] = image_i_reconstructed

            figure = pyplot.figure()
            axis = figure.add_subplot(1, 1, 1)

            axis.imshow(image, cmap = 'binary')

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
    
                image = numpy.zeros((H * 3, W * N_reconstructions))
    
                for i in range(N_reconstructions):
                
                    image_i = x[i].reshape((h, w))
                    x_a, x_b = int(0.5 * H - numpy.ceil(h/2.)), int(0.5 * H + numpy.floor(h/2))
                    y_a, y_b = int((i + 0.5) * W - numpy.ceil(w/2)), int((i + 0.5) * W + numpy.floor(w/2))
                    image[x_a: x_b, y_a: y_b] = image_i
                
                    image_i_upscaled = imresize(image_i, (H, W), interp = "bicubic") / 255.
                    x_up_a, x_up_b = 1 * H, 2 * H
                    y_up_a, y_up_b = i * W, (i + 1) * W
                    image[x_up_a: x_up_b, y_up_a: y_up_b] = image_i_upscaled
                
                    image_i_reconstructed = x_reconstructed[i].reshape((H, W))
                    x_recon_a, x_recon_b = 2 * H, 3 * H
                    y_recon_a, y_recon_b = i * W, (i + 1) * W
                    image[x_recon_a: x_recon_b, y_recon_a: y_recon_b] = image_i_reconstructed
    
                figure = pyplot.figure()
                axis = figure.add_subplot(1, 1, 1)
    
                axis.imshow(image, cmap = 'binary')
    
                axis.set_xticks(numpy.array([]))
                axis.set_yticks(numpy.array([]))
    
                plot_name = figure_path("reconstructions_homemade_" + specifications + ".pdf")
                pyplot.savefig(plot_name)
                print("Reconstructions of homemade numbers saved as {}.".format(plot_name))
            
        figure = pyplot.figure()
        axis = figure.add_subplot(1, 1, 1)
        
        for learning_curve in learning_curves:
            
            epochs = learning_curve["data"]["epochs"]
            cost_train = learning_curve["data"]["training cost function"]
            downsampling_factor = learning_curve["downsampling factor"]
            
            axis.plot(epochs, cost_train, label = "$d = {}$".format(downsampling_factor))
        
        pyplot.legend(loc = "best")
        
        axis.set_ylabel("Variational lower bound")
        axis.set_xlabel('Epochs')
        
        specifications = "l{}_e{}".format(latent_size, N_epochs)
        plot_name = figure_path("learning_curves_" + specifications + ".pdf")
        pyplot.savefig(plot_name)
        print("Learning curves for different downsampling factors saved as {}.".format(plot_name))
        
        print

if __name__ == '__main__':
    script_directory()
    main()
