# Improved-WGAN
WGAN-GP Implemented in Python3 with PyTorch 0.4.0

I got frustrated that there weren't really any baseline implementations for the wgan-gp in PyTorch-0.4.0, much less in Python3. So here's one I made on a nice summer morning. 


It should do MNIST and CIFAR without much trouble. I'll get to working on some agnostic one with will just use the `ImageFolder` function. It seems like a lot of people just want to run these on their own images. 

# MNIST

![results](images/mnist_results.png)




## Notes


If you have PyTorch 0.4.1+, there is a weird bug/feature with the mean() function creating a 1-dim tensor that gets broadcasted to 0-dim. To avoid it just replace instances of `.mean()` with `.mean(0, keepdim=True)`
