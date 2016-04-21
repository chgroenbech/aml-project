class VariationalAutoEncdoer:
    def __init__(self, shape_in, shape_hidden_enc, shape_out, shape_hidden_dec = None, sample_size = 1, batch_size = 200):
        super(VariationalAutoencdoer, self).__init__()
        self.batch_size = batch_size
        self.sample_size = sample_size
        
        shape_data = [batch_size, *shape_in]
        
        # l_in_enc = lasagne.layers.InputLayer(shape=shape_data)
#         l_prev_enc = l_in_enc
#
#         for i in range(len(shape_hidden_enc)):
#             l_tmp_enc = lasagne.layers.DenseLayer(l_prev_enc,
#                                                       num_units=shape_hidden_enc[i],
#                                                       W=lasagne.init.Uniform(),
#                                                       nonlinearity=transfer_function)
#             l_prev_enc = l_tmp_enc
#
#
#         l_in_dec = lasagne.layers.InputLayer(shape=(batch_size, shape_out))
#         l_prev_dec = l_in_dec
#
#         if shape_hidden_dec is None:
#             shape_hidden_dec = shape_hidden_enc
#
#         for i in range(len(shape_hidden_dec)):
#             l_tmp_dec = lasagne.layers.DenseLayer(l_prev_dec,
#                                                       num_units=shape_hidden_dec[-(i + 1)],
#                                                       W=lasagne.init.Uniform(),
#                                                       nonlinearity=transfer_function)
#             l_prev_dec = l_tmp_dec
#
#
#         l_in = lasagne.layers.InputLayer(shape=shape_data)
#
#         # self.model = VAELayer(l_in,
#         #                           enc=l_prev_enc,
#         #                           dec=l_prev_dec,
#         #                           latent_size=shape_out,
#         #                           x_distribution='bernoulli',
#         #                           qz_distribution='gaussianmarg',
#         #                           pz_distribution='gaussianmarg')
#
#         self.x = T.matrix('x')

    
        
        
