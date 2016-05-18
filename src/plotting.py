#!/usr/bin/env python

from __future__ import division

from matplotlib import pyplot

import numpy
import theano

import data
import pickle

from scipy.misc import imresize, imsave

from pprint import pprint

from cycler import cycler

from aux import data_path, figure_path, script_directory, colours

# TODO Use TeX
TeX = False

pyplot.rc("axes", prop_cycle = cycler("color", colours))
pyplot.rc("figure", figsize = (10, 4))

if TeX:
    pyplot.rc("text", usetex = True)
    pyplot.rc('font', **{'family':"sans-serif"})

    params = {'text.latex.preamble': [r'\usepackage{siunitx}', 
        r'\usepackage{sfmath}', r'\sisetup{detect-family = true}',
        r'\usepackage{amsmath}'], }
    pyplot.rcParams.update(params)

def main():
    
    # Setup
    
    downsampling_factors = [1, 2, 4]
    latent_sizes = [2, 30]
    # latent_sizes = [2, 5, 10, 30]
    N_epochs = 50
    binarise_downsampling = False
    bernoulli_sampling = True
    
    N_reconstructions_max = 6
    
    # Fix random seed for reproducibility
    numpy.random.seed(1234)
    
    # Table for results
    table = []
    
    for latent_size in latent_sizes:
        
        learning_curves = []
        reconstructions_set = []
        
        for downsampling_factor in downsampling_factors:
            
            # Data
    
            specifications = "{}ds{}{}_l{}_e{}".format("b_" if bernoulli_sampling else "", downsampling_factor, "b" if binarise_downsampling else "", latent_size, N_epochs)
            file_name = "results_" + specifications + ".pkl"
    
            try:
                with open(data_path(file_name), "r") as f:
                    setup_and_results = pickle.load(f)
            except Exception as e:
                print(e)
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
            
            epochs = numpy.array(learning_curve["epochs"])
            cost_train = numpy.array(learning_curve["training cost function"])
            cost_test = numpy.array(learning_curve["test cost function"])
            
            print("Training variational lower bound: {:.2f}.".format(cost_train[-1]))
            print("Test variational lower bound: {:.2f}.".format(cost_test[-1]))
            
            if downsampling_factor == 1:
                cost_test_ds1 = cost_test[-1]
                cost_train_ds1 = cost_train[-1]
            elif downsampling_factor > 1:
                table_row = "& {:d} & {:d} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\\\\n".format(latent_size, downsampling_factor, cost_train[-1], cost_train[-1] - cost_train_ds1, cost_test[-1], cost_test[-1] - cost_test_ds1)
                table.append(table_row)
            
            figure = pyplot.figure()
            axis = figure.add_subplot(1, 1, 1)
    
            axis.plot(epochs, cost_train, color = colours[0], label = 'Training data')
            axis.plot(epochs, cost_test, color = colours[0], label = 'Test data')
    
            pyplot.legend(loc = "best")
    
            axis.set_ylabel("Variational lower bound")
            axis.set_xlabel('Epochs')
    
            plot_name = figure_path("learning_curve_" + specifications + ".pdf")
            pyplot.savefig(plot_name, bbox_inches='tight')
            print("Learning curve saved as {}.".format(plot_name))
    
            ## Reconstruction
    
            reconstructions = results["reconstructions"]
            
            if downsampling_factor > 1:
                reconstructions_set.append({"data": reconstructions, "downsampling factor": downsampling_factor})
            
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
            
            alignment = {"horizontalalignment": "right", "verticalalignment": "center"}
            
            axis.text(-5, 0.5 * H, "Originals", fontdict = alignment)
            axis.text(-5, 1.5 * H, "Downsampled", fontdict = alignment)
            axis.text(-5, 2.5 * H, "Upscaled", fontdict = alignment)
            axis.text(-5, 3.5 * H, "VAE", fontdict = alignment)
            
            axis.set_xticks(numpy.array([]))
            axis.set_yticks(numpy.array([]))
            
            plot_name = figure_path("reconstructions_" + specifications + ".pdf")
            pyplot.savefig(plot_name, bbox_inches = 'tight')
            plot_name = figure_path("reconstructions_" + specifications + ".png")
            imsave(plot_name, 1 - image)
            print("Reconstructions saved as {}.".format(plot_name))
            
            # Single Reconstruction example
            
            image_original = x[0].reshape((H, W))
            plot_name = figure_path("example_original_" + specifications + ".png")
            imsave(plot_name, 1 - image_original)
            
            image_LR = x_LR[0].reshape((h, w))
            plot_name = figure_path("example_downsampled_" + specifications + ".png")
            imsave(plot_name, 1 - image_LR)
            
            image_reconstructed = x_reconstructed[0].reshape((H, W))
            plot_name = figure_path("example_reconstructed_" + specifications + ".png")
            imsave(plot_name, 1 - image_reconstructed)
            
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
                pyplot.savefig(plot_name, bbox_inches='tight')
                plot_name = figure_path("manifold_" + specifications + ".png")
                imsave(plot_name, 1 - canvas)
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
                pyplot.savefig(plot_name, bbox_inches='tight')
                print("Reconstructions of homemade numbers saved as {}.".format(plot_name))
            
            print
        
        specifications = "{}ds{}_l{}_e{}".format("b_" if bernoulli_sampling else "", "b" if binarise_downsampling else "", latent_size, N_epochs)
        
        if len(learning_curves) > 1:
            
            figure = pyplot.figure()
            axis = figure.add_subplot(1, 1, 1)
            
            for c, learning_curve in enumerate(learning_curves):
                
                epochs = numpy.array(learning_curve["data"]["epochs"])
                cost_train = numpy.array(learning_curve["data"]["training cost function"])
                cost_test = numpy.array(learning_curve["data"]["test cost function"])
                downsampling_factor = learning_curve["downsampling factor"]
                
                label_train = "$d = {}$, training set".format(downsampling_factor)
                label_test = "$d = {}$, test set".format(downsampling_factor)
                
                axis.plot(epochs, cost_train, color = colours[c], label = label_train)
                axis.plot(epochs, cost_test, linestyle = 'dashed', color = colours[c], label = label_test)
        
            pyplot.legend(loc = "best")
        
            axis.set_ylabel("Variational lower bound")
            axis.set_xlabel('Epochs')
        
            plot_name = figure_path("learning_curves_" + specifications + ".pdf")
            pyplot.savefig(plot_name, bbox_inches='tight')
            print("Learning curves for different downsampling factors saved as {}.".format(plot_name))
            
            print
        
        if len(reconstructions_set) > 1:
            
            N_reconstructions_actual = len(reconstructions_set[0]["data"]["reconstructions"])
            
            N_reconstructions = min(N_reconstructions_max, len(x_reconstructed))
            
            space = 1
            
            image = numpy.zeros((H * 4, (W * N_reconstructions + space) * len(reconstructions_set) - space))
            
            for k, reconstructions in enumerate(reconstructions_set):
                
                x = reconstructions["data"]["originals"]
                x_LR = reconstructions["data"]["downsampled"]
                x_reconstructed = reconstructions["data"]["reconstructions"]
                
                downsampling_factor = reconstructions["downsampling factor"]
                
                h = int(H / downsampling_factor)
                w = int(W / downsampling_factor)
                
                for i in range(N_reconstructions):
            
                    image_i = x[i].reshape((H, W))
                    x_a, x_b = 0 * H, 1 * H
                    y_a, y_b = i * W + (W * N_reconstructions + space) * k, (i + 1) * W + (W * N_reconstructions + space) * k
                    image[x_a: x_b, y_a: y_b] = image_i
                    
                    image_i_LR = x_LR[i].reshape((h, w))
                    x_LR_a, x_LR_b = int(1.5 * H - numpy.ceil(h/2.)), int(1.5 * H + numpy.floor(h/2))
                    y_LR_a, y_LR_b = int((i + 0.5) * W - numpy.ceil(w/2)) + (W * N_reconstructions + space) * k, int((i + 0.5) * W + numpy.floor(w/2)) + (W * N_reconstructions + space) * k
                    image[x_LR_a: x_LR_b, y_LR_a: y_LR_b] = image_i_LR
                
                    image_i_upscaled = imresize(image_i_LR, (H, W), interp = "bicubic") / 255.
                    x_up_a, x_up_b = 2 * H, 3 * H
                    y_up_a, y_up_b = i * W + (W * N_reconstructions + space) * k, (i + 1) * W + (W * N_reconstructions + space) * k
                    image[x_up_a: x_up_b, y_up_a: y_up_b] = image_i_upscaled
                
                    image_i_reconstructed = x_reconstructed[i].reshape((H, W))
                    x_recon_a, x_recon_b = 3 * H, 4 * H
                    y_recon_a, y_recon_b = i * W + (W * N_reconstructions + space) * k, (i + 1) * W + (W * N_reconstructions + space) * k
                    image[x_recon_a: x_recon_b, y_recon_a: y_recon_b] = image_i_reconstructed
                
                if k < len(reconstructions_set) - 1:
                    image[:, (W * N_reconstructions) * (k + 1)] = 1
            
            figure = pyplot.figure()
            axis = figure.add_subplot(1, 1, 1)
            
            axis.imshow(image, cmap = 'binary')
            
            row_alignment = {"horizontalalignment": "right", "verticalalignment": "center"}
            column_alignment = {"horizontalalignment": "center", "verticalalignment": "baseline", "size": "x-large"}
            
            axis.text(-5, 0.5 * H, "Originals", fontdict = row_alignment)
            axis.text(-5, 1.5 * H, "Downsampled", fontdict = row_alignment)
            axis.text(-5, 2.5 * H, "Upscaled\nBicubic", fontdict = row_alignment)
            axis.text(-5, 3.5 * H, "Upscaled\nVAE", fontdict = row_alignment)
            
            for k, reconstructions in enumerate(reconstructions_set):
                axis.text(W * N_reconstructions / 2 + (W * N_reconstructions + space) * k, -5, "$\\times 1/{}$".format(reconstructions["downsampling factor"]), fontdict = column_alignment)
            
            axis.set_xticks(numpy.array([]))
            axis.set_yticks(numpy.array([]))
            
            plot_name = figure_path("reconstructions_set_" + specifications + ".pdf")
            pyplot.savefig(plot_name, bbox_inches = 'tight')
            plot_name = figure_path("reconstructions_set_" + specifications + ".png")
            imsave(plot_name, 1 - image)
            print("Reconstructions saved as {}.".format(plot_name))
            
        
        print
    
    with open(data_path("results_variational_lower_bound.tex"), "w") as results_file:
        for table_row in table:
            results_file.write(table_row)

if __name__ == '__main__':
    script_directory()
    main()
