# Self-Supervision for Transfer Learning

The purpose of the is repo is to:

1. Train an autoencoder (AE), or variational autoencoder (VAE) or GAN on data in a self-supervised manner
2. Fine tune the learned features on a task with very few labels, to test transfer learning 
   
To run:

    python train.py --config config/config.json

To install:

Create the docker image and container 

    docker build -f Dockerfile.gpu -t ss4tl .
    docker run -v /home/username:/home/username --name ss4tl -it ss4tl bash

This repo was started from [victoresque's PyTorch Template Project](https://github.com/victoresque/pytorch-template)

    