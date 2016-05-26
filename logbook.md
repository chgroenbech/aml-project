\#02460: Advanced Machine Learning

# Logbook

* Christopher Heje Grønbech (s152421)
* Maximillian Fornitz Vording (s147246)
* (Supervisor: Ole Winther)


## Project goals ##

### Learning objectives ###

* Perform and understand generative modeling with variational deep learning
    * Understand and investigate the theory, applications and limitations of variational auto-encoders.
    * Understand super-resolution imaging and investigate how VAE can be used for this. 
    * Understand and Theano and Lasagne modules in Python for implementing a variational auto-encoder model.

### Delimitations, hypotheses and aims ###

#### Hypothesis:

Variational auto-encoders can perform super-resolution tasks better than traditional methods by statistical inference.

#### Delimitations:

Time is a major delimiter for this project. This is due to the time needed for the 3 following steps: 
* understanding the theory of variational auto-encoders includes reading several articles and understanding variational inference and deep learning models.
* implementation: Deep learning models build on several libraries. 
* investigation and computation: Optimization of a variational auto-encoder takes a long time. 

## Meetings ##

### Week 1 ###

* Questions
* Reading, who and what
* Implementation, who and what
* Results, who and what
* Decisions, who and what, what do you don alone, what do you do together
* Presentation of results since last meeting
* Action points for next week

### Week 1 ###
joint meeting on background material:

All read chapter in Bishop:

* §§10--10.1: variational learning.
* Chapter 12, especially §12.2: non-linear latent variable models.

as well as original article on VAE (Kingma and Welling (2013)).

Went through the benchmarks in Deep Learning:

* ImageNet
* CNN
* Speech Recognition on spectrum
* Future: Text

Our work: Generative model p(x|z), generate images of digits. 

Problems with more complex images.
