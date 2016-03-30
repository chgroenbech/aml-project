import numpy
import theano
import theano.tensor as T

from lasagne.layers.base import Layer



class VAELayer(Layer):
	"""docstring for VAELayer"""
	def __init__(self, arg):
		super(VAELayer, self).__init__()
		self.arg = arg
		

class VAE:
	"""docstring for VAE"""
	def __init__(self, n_in, n_hidden_enc, n_out, n_hidden_dec=None, trans_func=rectify, batch_size=200):
		super(VAE, self).__init__()
        self.batch_size = batch_size
		
		self.srng = RandomStreams()
		
		shape = n_in.insert(0,batch_size)
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


		l_in = lasagne.layers.InputLayer(shape=(batch_size, n_in))
	    self.model = VAELayer(l_in,
	                              enc=l_prev_enc,
	                              dec=l_prev_dec,
	                              latent_size=n_out,
	                              x_distribution='bernoulli',
	                              qz_distribution='gaussianmarg', #gaussianmarg
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
