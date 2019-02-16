# Improved-WGAN
WGAN-GP Implemented in Python3 with PyTorch 0.4.0

I got frustrated that there weren't really any baseline implementations for the wgan-gp in PyTorch-0.4.0, much less in Python3, So here's one I made on a nice summer morning. 

This should work **out of the box** with just a couple packages:

* PyTorch 0.4.1
* Scipy
* Numpy


It should do MNIST and CIFAR without much trouble. I'll get to working on some agnostic one which will just use the `ImageFolder` function. It seems like a lot of people just want to run these on their own images. But to do that I'll need to implement the ResNet generator from the paper, maybe for another weekend unless someone wants it right now

# MNIST

![results](images/mnist_results.png)

# CIFAR10

![results](images/cifar_results.png)
