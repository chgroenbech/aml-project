\#02460: Advanced Machine Learning

# Logbook

* Christopher Heje Grønbech (s152421)
* Maximillian Fornitz Vording (s147246)

## Project goals ##

### Learning objectives ###

* Understand and use generative modelling with variational deep learning.
* Understand and investigate the theory, applications and limitations of variational auto-encoders.
* Understand super-resolution imaging and investigate how VAE can be used for this. 
* Understand Theano and Lasagne modules for Python for implementing a variational auto-encoder model.

### Hypothesis

Variational auto-encoders can perform super-resolution tasks better than traditional methods by statistical inference.

### Delimitations

Time is a major delimiter for this project. The reason is threefold:

* Understanding the theory of variational auto-encoders includes reading several articles and understanding variational inference and deep-learning models;
* implementing a variational auto-encoder requires using several libraries, which we need to become familiar with; and
* investigating values for hyperparameters and training the variational auto-encoder takes a lot of time.

## Project work ##

All through the project, we had weekly joint meetings with our supervisor and his other groups presenting and discussing our findings.

We have also used Git version control to synchronise and keep track of our progress. As such, a detailed account of all our contributions to the implementation, poster, and report can be seen on our [GitHub page](https://github.com/chgroenbech/aml-project).

### Week 1: 4--10 March ###

For the group meeting we both read and discussed the background material in Bishop:

* §§10--10.1: variational learning.
* §12.2: non-linear latent variable models.

### Week 2: 11--17 March ###

We both read and discussed the original article on VAE (Kingma and Welling (2013)) for the group meeting. We also brainstormed ideas for applications.

### Week 3: 29--31 March ###

We started looking at ways to implement a variational auto-encoder using the Python module Lasagne and got it somewhat working. We also considered what to use the variational auto-encoder for.

### Week 4: 1--7 April ###

We changed our implementation to also use the Parmesan Python module, but still only got it working in the base case. We also decided on the subject of our project: super-resolution.

### Week 5: 8--14 April ###

We read the paper on semi-supervised variational auto-encoding (Kingma et al (2014)), a paper on per-feature loss (Johnson et al (2016)) as well as an introduction to convolutional neural networks for a Stanford course. We also continued our work on the variational auto-encoder, but had trouble getting the dimensions match up using convolutional neural networks.

### Week 6: 15--21 April ###

We continued working on getting convolutional neural networks to work with the variational auto-encoder. Christopher also reworked the implementation to be more clear and concise as well as enabling multiple inputs (low- and high-resolutions).

### Week 7: 22--28 April ###

Maximillian used our implementation to investigate different latent sizes, learning rates, and number of epochs. He also "drew" binarised, low-resolution numbers as text files for testing. Christopher continued work on the convolutional neural networks.

We both prepared a presentation of our status and progress and presented it at the joint meeting.

### Week 8: 29 April -- 4 May ###

We talked about using other loss functions and Maximillian started writing about the theory.

### Week 9: 5--12 May ###

Christopher got the variational auto-encoder working with convolutional neural networks, but it was really slow and gave worse results with a minimal setup. Max continued work on the theory section. We also made a detailed outline for the poster.

### Week 10: 13--19 May ###

We ran the variational auto-encoder multiple times for different latent sizes and downsampling factors, while Christopher prepared plots for the poster, wrote bullet points for the discussion and conclusion, and Max finished the theory and introduction for the poster. We also switched it up and rewrote each other's sections.

Finally, we presented our poster.

### Week 11: 20--26 May ###

We wrote the report based on the material for the poster as well as the feedback from the poster presentation. Max focused again on fleshing out the theory section, while Christopher focused on the other sections. Again, we switched it up by rewriting each other's parts.
