# RNN module of grainLearning

This module uses recurrent neural networks (RNN) to predict a sequence of macroscopic observables (i.e. stress) of a granular material undergoing a given input sequence of strains.
We have tested this module for training on DEM simulations using [YADE](http://yade-dem.org/). In the dataset we consider triaxial compressions of samples with different contact parameters.

[here could go the illustration of the kind of time series]

## Installation

During the installation of grainLearning activate extra `rnn`:

`poetry install --extras "rnn"` or `pip install .[rnn]`

*Note:* This will install a version of tensorflow depending on your system. If gpu dependencies are not installed or activated, it will default to **CPU** version. Check the installation of tensorflow and/or re-install it following specific instructions for your hardware.

> For MacOS with arm64 processor, we recommend to install  tensorflow following [this](https://betterdatascience.com/install-tensorflow-2-7-on-macbook-pro-m1-pro/) and install grainLearning without `rnn` extra. In this case if you want to have [weights and biases](https://wandb.ai/site) you can install it via `pip install wandb`.

## How to use

There are three main usages of RNN module:

1. [Train a RNN with your own data.](/tutorials/data_driven/LSTM/train_rnn.ipynb)
2. [Make a prediction with a pre-trained model.](/tutorials/data_driven/LSTM/predict.ipynb)
3. [Use a trained RNN in grainLearning calibration process.](/tutorials/data_driven/LSTM/rnn_calibration_GL.ipynb)

For more details about how are these model built check the [documentation](https://grainlearning.readthedocs.io/en/latest).
