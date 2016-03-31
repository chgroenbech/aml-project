#!/usr/bin/env python3

import data
from vae import VAE

import numpy
from matplotlib import pyplot as plt
from lasagne.nonlinearities import rectify
from lasagne.updates import adam

from aux import fig_path, script_directory

def main():
    
    # Data
    
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = data.load()
    
    # C, D, D = X_train[0].shape.eval() # shape (channels, dimensions)
    N = X_train[0].shape.eval()
    C = 1 # number of channels
    D = int(numpy.sqrt(N))
    K = 10 # number of classes
    
    # Setup
    
    M = 500 # mini-batch size
    E = 1 # number of epochs
    H = 200 # number of hidden units
    L = 2 # number of latent variables (z)

    # Initialize model
    # model = VAE([C,D,D],[H,H],L,trans_func=rectify,batch_size=M)
    model = VAE(N,[H, H],L,trans_func=rectify,batch_size=M)

    # Train model
    
    train_model, test_model, valid_model = model.build_model(X_train, X_test, X_val, adam, update_args=(1e-3,))
    
    epochs = numpy.linspace(0, E-1, E)
    eval_train = numpy.empty(E)
    eval_test = numpy.empty(E)
    eval_valid = numpy.empty(E)
    n_train_batches = X_train.get_value(borrow=True).shape[0] / M
    n_test_batches = X_test.get_value(borrow=True).shape[0] / M
    n_valid_batches = X_val.get_value(borrow=True).shape[0] / M
    
    print("Epoch    Train Loss    Test Loss    Validation Loss")
    print("---------------------------------------------------")
    for e in range(E):
        avg_costs = []
        for minibatch_index in range(int(n_train_batches)):
            minibatch_avg_cost = train_model(minibatch_index)
            avg_costs.append(minibatch_avg_cost)
        eval_train[e] = numpy.mean(avg_costs)
        test_losses = [test_model(i) for i in range(int(n_test_batches))]
        valid_losses = [valid_model(i) for i in range(int(n_valid_batches))]
        eval_test[e] = numpy.mean(test_losses)
        eval_valid[e] = numpy.mean(valid_losses)
        print(" {:2d}      {:.3e}    {:.3e}    {:.3e}".format(e, eval_train[e], eval_valid[e], eval_test[e]))
        log_pz, log_qz_given_x, log_px_given_z = model.model.get_log_distributions(X_test)
    
    # Plotting
    
   ## Learning curves
    
    plt.figure()
    
    plt.plot(epochs, eval_train, label = "Training data")
    plt.plot(epochs, eval_test, label = "Test data")
    plt.plot(epochs, eval_valid, label = "Validation data")
    
    plt.legend()
    
    plt.xlabel("Epoch")
    plt.ylabel("log-likelihood")
    
    plt.savefig(fig_path("learning_curve.pdf"))
    
    ## Reconstructions
    
    test_x_eval = X_test.eval()
    subset = numpy.random.randint(0, len(test_x_eval), size=50)
    x = numpy.array(test_x_eval)[numpy.array(subset)]
    z = model.get_output(x)
    x_recon = model.get_reconstruction(z).eval()
    
    plt.figure()
    
    img_out = numpy.zeros([2 * D, len(subset) * D])
    
    for i in range(len(subset)):
        y_a, y_b = i * D, (i + 1) * D
        
        x_a, x_b = 0, D
        img = x[i].reshape(D, D)
        img_out[x_a:x_b, y_a:y_b] = img
        
        x_recon_a, x_recon_b = D, 2 * D
        img_recon = x_recon[i].reshape(D, D)
        img_out[x_recon_a:x_recon_b, y_a:y_b] = img_recon
        
    m = plt.matshow(img_out, cmap = "binary")
    plt.xticks([])
    plt.yticks([])
    
    plt.savefig(fig_path("reconstruction.pdf"))

if __name__ == '__main__':
    script_directory()
    main()
