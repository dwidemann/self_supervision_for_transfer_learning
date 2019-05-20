# Self-Supervision for Transfer Learning

The purpose of the is repo is to:

1. Train an autoencoder (AE), or variational autoencoder (VAE) or GAN on data in a self-supervised manner
2. Fine tune the learned features on a task with very few labels, to test transfer learning 

Here is an example of the current workflow:  

First train an autoencoder like model

    python train.py --config config/ae_config.json --device 0,1,2,3

This will save a model that did best on the validation data, e.g.

    /ae_results/models/ae_mnist/0518_185527/model_best.pth

Insert this path into the fine_tune_config.json file and train the fine tune model:

    python train.py --config config/fine_tune_config.json --device 0,1,2,3

to test:

    python test.py --resume path_to_best_fine_tune_model/model_best.pt --device 0,1,2,3

If you want to compare how much the self-supervised features helped, you can train and test using the fine_tune_null_config.json file. This creates the same model but does not use self-supervised features. 


To install:

Create the docker image and container 

    docker build -f Dockerfile.gpu -t ss4tl .
    docker run -v /home/username:/home/username --name ss4tl -it ss4tl bash

This repo was started from [victoresque's PyTorch Template Project](https://github.com/victoresque/pytorch-template)

    