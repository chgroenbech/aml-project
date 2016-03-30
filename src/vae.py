import numpy
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from lasagne.layers.base import Layer
from lasagne import init
from lasagne.nonlinearities import rectify

import lasagne

class VAE:
    def __init__(self, n_in, n_hidden_enc, n_out, n_hidden_dec=None, trans_func=rectify, batch_size=200):
        super(VAE, self).__init__()
        self.batch_size = batch_size

        self.srng = RandomStreams()
        
        shape = [batch_size, *n_in]
        l_in_enc = lasagne.layers.InputLayer(shape=shape)
        l_prev_enc = l_in_enc

        for i in range(len(n_hidden_enc)):
            l_tmp_enc = lasagne.layers.DenseLayer(l_prev_enc,
                                                      num_units=n_hidden_enc[i],
                                                      W=lasagne.init.Uniform(),
                                                      nonlinearity=trans_func)
            l_prev_enc = l_tmp_enc
        

        l_in_dec = lasagne.layers.InputLayer(shape=(batch_size, n_out))
        l_prev_dec = l_in_dec
        if n_hidden_dec is None:
            n_hidden_dec = n_hidden_enc
        
        for i in range(len(n_hidden_dec)):
            l_tmp_dec = lasagne.layers.DenseLayer(l_prev_dec,
                                                      num_units=n_hidden_dec[-(i + 1)],
                                                      W=lasagne.init.Uniform(),
                                                      nonlinearity=trans_func)
            l_prev_dec = l_tmp_dec


        l_in = lasagne.layers.InputLayer(shape=shape)
        self.model = VAELayer(l_in,
                                  enc=l_prev_enc,
                                  dec=l_prev_dec,
                                  latent_size=n_out,
                                  x_distribution='bernoulli',
                                  qz_distribution='gaussianmarg',
                                  pz_distribution='gaussianmarg')
        self.x = T.matrix('x')

    # Build and train the model
    def build_model(self, train_x, test_x, valid_x, update, update_args):
        index = T.iscalar('index')
        batch_slice = slice(index * self.batch_size, (index + 1) * self.batch_size)

        x = self.srng.binomial(size=self.x.shape, n=1, p=self.x)
        log_pz, log_qz_given_x, log_px_given_z = self.model.get_log_distributions(self.x)
        loss_eval = (log_pz + log_px_given_z - log_qz_given_x).sum()
        loss_eval /= self.batch_size

        all_params = get_all_params(self.model)
        updates = update(-loss_eval, all_params, *update_args)

        train_model = theano.function([index], loss_eval, updates=updates,
                                      givens={self.x: train_x[batch_slice], },)

        test_model = theano.function([index], loss_eval,
                                     givens={self.x: test_x[batch_slice], },)

        validate_model = theano.function([index], loss_eval,
                                         givens={self.x: validation_x[batch_slice], },)

        return train_model, test_model, validate_model

    def draw_sample(self, z):
        return self.model.draw_sample(z)

    def get_output(self, dat):
        z, _, _ = self.model.get_z_mu_sigma(dat)
        return z

    def get_reconstruction(self, z):
        return self.model.dec_output(z)



class VAELayer(Layer):
    def __init__(self, incoming, enc, dec,
                 x_distribution='bernoulli',
                 pz_distribution='gaussian',
                 qz_distribution='gaussian',
                 latent_size=50,
                 W=init.Normal(0.01),
                 b=init.Normal(0.01),
                 **kwargs):
        super(VAELayer, self).__init__(incoming, **kwargs)
        num_batch, n_channels, n_dim1, n_dim2 = self.input_shape
        self.num_batch = num_batch
        self.n_features = n_channels*n_dim1*n_dim2
        self.x_distribution = x_distribution
        self.pz_distribution = pz_distribution
        self.qz_distribution = qz_distribution
        self.enc = enc
        self.dec = dec
        self._srng = RandomStreams()

        if self.x_distribution not in ['gaussian', 'bernoulli']:
            raise NotImplementedError
        if self.pz_distribution not in ['gaussian', 'gaussianmarg']:
            raise NotImplementedError
        if self.qz_distribution not in ['gaussian', 'gaussianmarg']:
            raise NotImplementedError

        self.params_enc = lasagne.layers.get_all_params(enc)
        self.params_dec = lasagne.layers.get_all_params(dec)
        for p in self.params_enc:
            p.name = "VAELayer enc :" + p.name
        for p in self.params_dec:
            p.name = "VAELayer dec :" + p.name

        self.num_hid_enc = enc.output_shape[1]
        self.num_hid_dec = dec.output_shape[1]
        self.latent_size = latent_size

        self.W_enc_to_z_mu = self.add_param(W, (self.num_hid_enc, latent_size))
        self.b_enc_to_z_mu = self.add_param(b, (latent_size,))

        self.W_enc_to_z_logsigma = self.add_param(W, (self.num_hid_enc, self.latent_size))
        self.b_enc_to_z_logsigma = self.add_param(b, (latent_size,))

        self.W_dec_to_x_mu = self.add_param(W, (self.num_hid_dec, self.n_features))
        self.b_dec_to_x_mu = self.add_param(b, (self.n_features,))

        self.W_params = [self.W_enc_to_z_mu,
                         self.W_enc_to_z_logsigma,
                         self.W_dec_to_x_mu] + self.params_enc + self.params_dec
        self.bias_params = [self.b_enc_to_z_mu,
                            self.b_enc_to_z_logsigma,
                            self.b_dec_to_x_mu]

        params_tmp = []
        if self.x_distribution == 'gaussian':
            self.W_dec_to_x_logsigma = self.add_param(W, (self.num_hid_dec, self.n_features))
            self.b_dec_to_x_logsigma = self.add_param(b, (self.n_features,))
            self.W_params += [self.W_dec_to_x_logsigma]
            self.bias_params += [self.b_dec_to_x_logsigma]
            self.W_dec_to_x_logsigma.name = "VAE: W_dec_to_x_logsigma"
            self.b_dec_to_x_logsigma.name = "VAE: b_dec_to_x_logsigma"
            params_tmp = [self.W_dec_to_x_logsigma, self.b_dec_to_x_logsigma]

        self.params = self.params_enc + [self.W_enc_to_z_mu,
                                             self.b_enc_to_z_mu,
                                             self.W_enc_to_z_logsigma,
                                             self.b_enc_to_z_logsigma] + self.params_dec + \
                      [self.W_dec_to_x_mu, self.b_dec_to_x_mu] + params_tmp

        self.W_enc_to_z_mu.name = "VAELayer: W_enc_to_z_mu"
        self.W_enc_to_z_logsigma.name = "VAELayer: W_enc_to_z_logsigma"
        self.W_dec_to_x_mu.name = "VAELayer: W_dec_to_x_mu"
        self.b_enc_to_z_mu.name = "VAELayer: b_enc_to_z_mu"
        self.b_enc_to_z_logsigma.name = "VAELayer: b_enc_to_z_logsigma"
        self.b_dec_to_x_mu.name = "VAELayer: b_dec_to_x_mu"

    def get_params(self):
        return self.params

    def get_output_shape_for(self, input_shape):
        dec_out_shp = self.dec.get_output_shape_for(
            (self.num_batch, self.num_hid_dec))
        if self.x_distribution == 'bernoulli':
            return dec_out_shp
        elif self.x_distribution == 'gaussian':
            return [dec_out_shp, dec_out_shp]

    def enc_output(self, x, *args, **kwargs):
        return lasagne.layers.get_output(self.enc, x, **kwargs)

    def dec_output(self, z, *args, **kwargs):
        h_dec = lasagne.layers.get_output(self.dec, z, **kwargs)
        if self.x_distribution == 'gaussian':
            mu_dec = T.dot(h_dec, self.W_dec_to_x_mu) + self.b_dec_to_x_mu
            log_sigma_dec = T.dot(h_dec, self.W_dec_to_x_logsigma) + self.b_dec_to_x_logsigma
            dec_out = mu_dec, log_sigma_dec
        elif self.x_distribution == 'bernoulli':
            # TODO: Finish writing the output of the dec for a bernoulli distributed x.
            dec_out = T.nnet.sigmoid(T.dot(h_dec, self.W_dec_to_x_mu) + self.b_dec_to_x_mu)
        else:
            raise NotImplementedError
        return dec_out

    def get_z_mu_sigma(self, x, *args, **kwargs):
        h_enc = self.enc_output(x, *args, **kwargs)
        mu_enc = T.dot(h_enc, self.W_enc_to_z_mu) + self.b_enc_to_z_mu
        log_sigma_enc = (T.dot(h_enc, self.W_enc_to_z_logsigma) +
                             self.b_enc_to_z_logsigma)
        eps = self._srng.normal(log_sigma_enc.shape)
        # TODO: Calculate the sampled z. 
        z = mu_enc + T.exp(0.5 * log_sigma_enc) * eps
        return z, mu_enc, log_sigma_enc

    def get_log_distributions(self, x, *args, **kwargs):
        # sample z from q(z|x).
        h_enc = self.enc_output(x, *args, **kwargs)
        mu_enc = T.dot(h_enc, self.W_enc_to_z_mu) + self.b_enc_to_z_mu
        log_sigma_enc = (T.dot(h_enc, self.W_enc_to_z_logsigma) +
                             self.b_enc_to_z_logsigma)
        eps = self._srng.normal(log_sigma_enc.shape)
        z = mu_enc + T.exp(0.5 * log_sigma_enc) * eps

        # forward pass z through dec to generate p(x|z).
        dec_out = self.dec_output(z, *args, **kwargs)
        if self.x_distribution == 'bernoulli':
            x_mu = dec_out
            log_px_given_z = -T.nnet.binary_crossentropy(x_mu, x)
        elif self.x_distribution == 'gaussian':
            x_mu, x_logsigma = dec_out
            log_px_given_z = normal2(x, x_mu, x_logsigma)

        # sample prior distribution p(z).
        if self.pz_distribution == 'gaussian':
            log_pz = standard_normal(z)
        elif self.pz_distribution == 'gaussianmarg':
            log_pz = -0.5 * (T.log(2 * np.pi) + (T.sqr(mu_enc) + T.exp(log_sigma_enc)))

        # variational approximation distribution q(z|x)
        if self.qz_distribution == 'gaussian':
            log_qz_given_x = normal2(z, mu_enc, log_sigma_enc)
        elif self.qz_distribution == 'gaussianmarg':
            log_qz_given_x = - 0.5 * (T.log(2 * np.pi) + 1 + log_sigma_enc)

        # sum over dim 1 to get shape (,batch_size)
        log_px_given_z = log_px_given_z.sum(axis=1, dtype=theano.config.floatX)  # sum over x
        log_pz = log_pz.sum(axis=1, dtype=theano.config.floatX)  # sum over latent vars
        log_qz_given_x = log_qz_given_x.sum(axis=1, dtype=theano.config.floatX)  # sum over latent vars

        return log_pz, log_qz_given_x, log_px_given_z

    def draw_sample(self, z=None, *args, **kwargs):
        if z is None:  # draw random z
            z = self._srng.normal((self.num_batch, self.latent_size))
        return self.dec_output(z, *args, **kwargs)
        
        
