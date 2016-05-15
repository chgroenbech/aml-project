#!/usr/bin/env python

import data

import numpy

import theano
import theano.tensor as T

from lasagne import init, updates

from lasagne.nonlinearities import identity, sigmoid, rectify, softmax, softplus

from lasagne.layers import (
    InputLayer, DenseLayer,
    Pool2DLayer, ReshapeLayer, DimshuffleLayer,
    get_output,
    get_all_params
)
from parmesan.layers.sample import SimpleSampleLayer, SampleLayer

from parmesan.distributions import (
    log_stdnormal, log_normal2, log_bernoulli,
    kl_normal2_stdnormal
)

import time

from matplotlib import pyplot

from scipy.stats import norm as gaussian

from aux import data_path, figure_path, script_directory, enumerate_reversed

def main():
    
    # Setup
    
    C = 1 # number of channels in image
    H = 28 # height of image
    W = 28 # width of image
    # K = 10 # number of classes
    
    downsampling_factor = 2
    
    h = H / downsampling_factor
    w = W / downsampling_factor
    
    hidden_sizes = [200, 200]
    latent_size = 10
    batch_size = 100
    
    analytic_kl_term = True
    learning_rate = 0.01 #0.0003
    
    N_epochs = 30 # 1000
    
    # shape = [H, W, C]
    shape = [H * W * C]
    
    # Symbolic variables
    symbolic_x_LR = T.matrix()
    symbolic_x_HR = T.matrix()
    symbolic_z = T.matrix()
    symbolic_learning_rate = T.scalar('learning_rate')
    
    # Fix random seed for reproducibility
    numpy.random.seed(1234)
    
    # Data
    
    file_name = "mnist.pkl.gz"
    file_path = data_path(file_name)
    
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = data.load(file_path, shape)
    
    X_train = numpy.concatenate([X_train, X_valid])
    
    X_train = X_train.astype(theano.config.floatX)
    X_test = X_test.astype(theano.config.floatX)

    N_train_batches = X_train.shape[0] / batch_size
    N_test_batches = X_test.shape[0] / batch_size

    # Setup shared variables
    X_train_shared = theano.shared(X_train, borrow = True)
    X_test_shared = theano.shared(X_test, borrow = True)
    
    # Models
    
    ## Recognition model q(z|x)
    
    l_enc_HR_in = InputLayer((None, H * W * C), name = "ENC_HR_INPUT")
    
    l_enc_HR_downsample = l_enc_HR_in
    
    l_enc_HR_downsample = ReshapeLayer(l_enc_HR_downsample, (-1, C, H, W))
    if downsampling_factor == 1:
        l_enc_HR_downsample = Pool2DLayer(l_enc_HR_downsample, pool_size = downsampling_factor, mode = "average_exc_pad")
    l_enc_HR_downsample = ReshapeLayer(l_enc_HR_downsample, (-1, h * w * C))
    
    l_enc_LR_in = InputLayer((None, h * w * C), name = "ENC_LR_INPUT")
    
    l_enc = l_enc_LR_in
    
    for i, hidden_size in enumerate(hidden_sizes, start = 1):
        l_enc = DenseLayer(l_enc, num_units = hidden_size, nonlinearity = softplus, name = 'ENC_DENSE{:d}'.format(i))
    
    l_z_mu = DenseLayer(l_enc, num_units = latent_size, nonlinearity = identity, name = 'ENC_Z_MU')
    l_z_log_var = DenseLayer(l_enc, num_units = latent_size, nonlinearity = identity, name = 'ENC_Z_LOG_VAR')
    
    # Sample the latent variables using mu(x) and log(sigma^2(x))
    l_z = SimpleSampleLayer(mean = l_z_mu, log_var = l_z_log_var) # as Kingma
    # l_z = SampleLayer(mean = l_z_mu, log_var = l_z_log_var)

    ## Generative model p(x|z)
    
    l_dec_in = InputLayer((None, latent_size), name = "DEC_INPUT")
    
    l_dec = l_dec_in
    
    for i, hidden_size in enumerate_reversed(hidden_sizes, start = 0):
        l_dec = DenseLayer(l_dec, num_units = hidden_size, nonlinearity = softplus, name = 'DEC_DENSE{:d}'.format(i))
    
    l_dec_x_mu = DenseLayer(l_dec, num_units = H * W * C, nonlinearity = sigmoid, name = 'DEC_X_MU')
    
    # TRY relu instead of softplus (maybe with more hidden units)
    # TRY softmax instead of sigmoid
    # PROBLEM with this is that we have several pixels activated.

    ## Get outputs from models
    
    # With noise
    x_LR_train = get_output(l_enc_HR_downsample, symbolic_x_HR, deterministic = False)
    z_train, z_mu_train, z_log_var_train = get_output(
        [l_z, l_z_mu, l_z_log_var], {l_enc_LR_in: x_LR_train}, deterministic = False
    )
    x_mu_train = get_output(l_dec_x_mu, {l_dec_in: z_train}, deterministic = False)
    
    # Without noise
    x_LR_eval = get_output(l_enc_HR_downsample, symbolic_x_HR, deterministic = True)
    z_eval, z_mu_eval, z_log_var_eval = get_output(
        [l_z, l_z_mu, l_z_log_var], {l_enc_LR_in: x_LR_eval}, deterministic = True
    )
    x_mu_eval = get_output(l_dec_x_mu, {l_dec_in: z_eval}, deterministic = True)
    
    # Sampling
    x_mu_sample = get_output(l_dec_x_mu, {l_dec_in: symbolic_z},
        deterministic = True)
    
    # Likelihood
    
    # Calculate the loglikelihood(x) = E_q[ log p(x|z) + log p(z) - log q(z|x)]
    def log_likelihood(z, z_mu, z_log_var, x_mu, x, analytic_kl_term):
        if analytic_kl_term:
            kl_term = kl_normal2_stdnormal(z_mu, z_log_var).sum(axis = 1)
            log_px_given_z = log_bernoulli(x, x_mu,  eps = 1e-6).sum(axis = 1)
            LL = T.mean(-kl_term + log_px_given_z)
        else:
            log_qz_given_x = log_normal2(z, z_mu, z_log_var).sum(axis = 1)
            log_pz = log_stdnormal(z).sum(axis = 1)
            log_px_given_z = log_bernoulli(x, x_mu,  eps = 1e-6).sum(axis = 1)
            LL = T.mean(log_pz + log_px_given_z - log_qz_given_x)
        return LL

    # log-likelihood for training
    ll_train = log_likelihood(
        z_train, z_mu_train, z_log_var_train, x_mu_train, symbolic_x_HR, analytic_kl_term)

    # log-likelihood for evaluating
    ll_eval = log_likelihood(
        z_eval, z_mu_eval, z_log_var_eval, x_mu_eval, symbolic_x_HR, analytic_kl_term)
    
    # Parameters to train
    parameters = get_all_params([l_z, l_dec_x_mu], trainable = True)
    print("Parameters that will be trained:")
    for parameter in parameters:
        print("{}: {}".format(parameter, parameter.get_value().shape))

    ### Take gradient of negative log-likelihood
    gradients = T.grad(-ll_train, parameters)

    # Adding gradient clipping to reduce the effects of exploding gradients,
    # and hence speed up convergence
    gradient_clipping = 1
    gradient_norm_max = 5
    gradient_constrained = updates.total_norm_constraint(gradients,
        max_norm = gradient_norm_max)
    gradients_clipped = [T.clip(g,-gradient_clipping, gradient_clipping) for g in gradient_constrained]
    
    # Setting up functions for training
    
    symbolic_batch_index = T.iscalar('index')
    batch_slice = slice(symbolic_batch_index * batch_size, (symbolic_batch_index + 1) * batch_size)

    update_expressions = updates.adam(gradients_clipped, parameters,
        learning_rate = symbolic_learning_rate)

    train_model = theano.function(
        [symbolic_batch_index, symbolic_learning_rate], ll_train,
        updates = update_expressions, givens = {symbolic_x_HR: X_train_shared[batch_slice]}
    )

    test_model = theano.function(
        [symbolic_batch_index], ll_eval,
        givens = {symbolic_x_HR: X_test_shared[batch_slice]}
    )
    
    def train_epoch(learning_rate):
        costs = []
        for i in range(N_train_batches):
            cost_batch = train_model(i, learning_rate)
            costs += [cost_batch]
        return numpy.mean(costs)
    
    def test_epoch():
        costs = []
        for i in range(N_test_batches):
            cost_batch = test_model(i)
            costs += [cost_batch]
        return numpy.mean(costs)
    
    # Training
    
    epochs = []
    cost_train = []
    cost_test = []

    for epoch in range(N_epochs):
        
        start = time.time()
        
        # Shuffle train data
        numpy.random.shuffle(X_train)
        X_train_shared.set_value(X_train)
        
        # TODO: Using dynamically changed learning rate
        train_cost = train_epoch(learning_rate)
        test_cost = test_epoch()
        
        duration = time.time() - start
        
        epochs.append(epoch)
        cost_train.append(train_cost)
        cost_test.append(test_cost)
        
        # line = "Epoch: %i\tTime: %0.2f\tLR: %0.5f\tLL Train: %0.3f\tLL test: %0.3f\t" % ( epoch, t, learning_rate, train_cost, test_cost)
        print("Epoch {:d} (duration: {:.2f} s, learning rate: {:.1e}):".format(epoch + 1, duration, learning_rate))
        print("    log-likelihood: {:.3f} (training set), {:.3f} (test set)".format(train_cost, test_cost))
    
    # N_epochs, latent_size
    # epochs, cost_train, cost_test
    # X_test_shared, l_enc_HR_downsample, l_z, x_mu_sample


if __name__ == '__main__':
    script_directory()
    main()
