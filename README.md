# UBImageNetTune

This repository contains the files and scripts to train a neutrino+cosmic vs. cosmic-only model.

Our strategy is to use transfer learning and fine-tune the first version of googlenet that has been pre-trained on the ImageNet dataset. We use the BVLC googlenet model.

This code also serves as a tutorial to training network using images stored in our LArCV format and using our fork of Caffe. After training, we provide some scripts that demonstrate how to do some simple analyses with the network.

If you need a crash course in CNNs, take a look at this:

[Standards CS231N: convolutional neural networks for visual recognition](http://cs231n.github.io/)
[Syllabus for above](http://cs231n.stanford.edu/syllabus.html)

I recommend listening to their lectures, which were [here](http://cs231n.stanford.edu/syllabus.html). Videos of the lectures are gone now, but speak to me about how one might get a copy.

note: if you don't have your own GPU to work with, you can use the AWS. They give you 750 hours of free time per month.  That should be enough time to do some prototyping and train a network or two per month. More serious work will require your own setup, though.  The class above has a [tutorial](http://cs231n.github.io/aws-tutorial/) on how to setup an AWS account and spin an instance with GPU access and other software.  I haven't looked into how to get LArCV and our custom caffe onto it though.

## Tutorial Instructions

There are a couple of [wiki pages](https://github.com/LArbys/UBImageNetTune/wiki) that provide brief tutorials on the basics of training and an analysis with the network output.

## Training (inside of a screen session)

TL;DR

    screen
    source setup_env.sh
    caffe train -solver rmsprop.solver -weights bvlc_googlenet.caffemodel >& log.txt

To monitor the training process, ppen another screen terminal (or VNC)

    python plot_training.py log.txt
    tail -n 50 log.txt




