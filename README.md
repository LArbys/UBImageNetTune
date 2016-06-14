# UBImageNetTune

This repository contains the files and scripts to train a neutrino+cosmic vs. cosmic-only model.

Our strategy is to use transfer learning and fine-tune the first version of googlenet that has been pre-trained on the ImageNet dataset. We use the BVLC googlenet model.

This code also serves as a tutorial to training network using images stored in our LArCV format and using our fork of Caffe. After training, we provide some scripts that demonstrate how to do some simple analyses with the network.

If you need a crash course in CNNs, take a look at this:

[Standards CS231N: convolutional neural networks for visual recognition](http://cs231n.github.io/)

I recommend listening to their lectures, which were [here](http://cs231n.stanford.edu/syllabus.html). Videos of the lectures are gone now, but speak to me about how one might get a copy.

note: if you don't have your own GPU to work with, you can use the AWS. They give you 750 hours of free time per month.  That should be enough time to do some prototyping and train a network or two per month. More serious work will require your own setup, though.  The class above has a [tutorial](http://cs231n.github.io/aws-tutorial/) on how to setup an AWS account and spin an instance with GPU access and other software.  I haven't looked into how to get LArCV and our custom caffe onto it though.

## Tutorial Instructions

1. First clone this directory
2. Download the training and validation data using get_data.sh
3. Train! (see more detailed instructions below)
4. While things train, monitor the loss and accuracies of the training and validation data sets
5. Plot the distribution of neutrino-like softmax score for the cosmic-only and neutrino+cosmics using pyROOT
6. Look at scores versus different truth quantities

## Training (inside of a screen session)

TL;DR

    screen
    source setup_env.sh
    caffe train -solver rmsprop.solver -weights bvlc_googlenet.caffemodel >& log.txt

Open another screen terminal (or VNC)

    python plot_training.py log.txt
    tail -n 50 log.txt




