#!/usr/bin/env python3

from matplotlib import pyplot

from scipy.stats import norm
from scipy.stats import invgauss

from aux import data_path, fig_path, script_directory

def main():
    
    kind = " ({} epochs, {} latent neurons)".format(N_epochs, latent_size)
    learning_curve()

def learning_curve():
    
    figure = pyplot.figure()
    axis = figure.add_subplot(1, 1, 1)
    
    axis.plot(epochs, cost_train, label = 'training data')
    axis.plot(epochs, cost_test, label = 'testing data')
    
    pyplot.legend(loc = "best")
    
    axis.set_ylabel("Log Likelihood")
    axis.set_xlabel('Epochs')
    
    pyplot.savefig(figure_path("Learning curve" + kind + ".pdf"))

def reconstructions():
    
    N_reconstructions = 50
    
    X_test_eval = X_test_shared.eval()
    subset = numpy.random.randint(0, len(X_test_eval), size = N_reconstructions)
    
    x = X_test_eval[numpy.array(subset)]
    x_LR = get_output(l_enc_HR_downsample, x)
    z = get_output(l_z, x_LR).eval()
    x_reconstructed = x_mu_sample.eval({symbolic_z: z})
    
    image = numpy.zeros((H * 2, W * len(subset)))
    
    for i in range(len(subset)):
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
    
    pyplot.savefig(figure_path("Reconstructions" + kind + ".pdf"))

def manifold():

    x = numpy.linspace(0.1,0.9, 20)
    # TODO: Ideally sample from the real p(z)
    v = gaussian.ppf(x)
    z = numpy.zeros((20**2, 2))

    i = 0
    for a in v:
        for b in v:
            z[i,0] = a
            z[i,1] = b
            i += 1
    z = z.astype('float32')
    
    samples = x_mu_sample.eval({symbolic_z: z})

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
    
    pyplot.savefig(figure_path("Distribution" + kind + ".pdf"))

def reconstruct_homemade_number():
    for i in range(1,5):
        X_LR_HM = numpy.loadtxt("../data/hm_7_{}.txt".format(i)).reshape(-1, h**2)
        # X_LR_HM = theano.shared(X_HM_1, borrow = True)
        # print(X_LR_HM.shape)
        # print(X_LR_HM)
        z = get_output(l_z, X_LR_HM).eval()
        X_HM_reconstructed = x_mu_sample.eval({symbolic_z: z})[0]
        image = X_HM_reconstructed.reshape((H, W))
        figure = pyplot.figure()
        axis = figure.add_subplot(1, 1, 1)
        
        axis.imshow(image, cmap = 'gray')
        
        axis.set_xticks(numpy.array([]))
        axis.set_yticks(numpy.array([]))
        
        pyplot.savefig(figure_path("Homemade \#{} (reconstructed) ".format(i) + kind + ".pdf"))

        figure = pyplot.figure()
        axis = figure.add_subplot(1, 1, 1)
        
        axis.imshow(X_LR_HM[0].reshape((h, w)), cmap = 'gray')
        
        axis.set_xticks(numpy.array([]))
        axis.set_yticks(numpy.array([]))
        
        pyplot.savefig(figure_path("Homemade \#{} ".format(i) + kind + ".pdf"))

if __name__ == '__main__':
    script_directory()
    main()
