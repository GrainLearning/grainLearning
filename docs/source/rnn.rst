RNN Module
==========

We implemented a `Recurrent Neural Network (RNN) <https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-recurrent-neural-networks>`_ model in tensorflow framework.

The RNN takes as inputs a vector of *parameters* (i.g. contact parameters) and a sequence of size :math:`mathcal{N}` (i.g. applied strain). The model returns a sequence of size :math:`\mathcal{N}`

There are three main usages of RNN module:

1. `Train a RNN with your own data`_.
2. `Make a prediction with a pre-trained model`_.
3. `Use a trained RNN in grainLearning calibration process`_.
4. Train a RNN model during the grainLearning calibration process.

Train a RNN with your own data
------------------------------

Get your data
`````````````
The RNN model of this module considers a specific data format and organization. Our example of data consists of several DEM simulations of Triaxial Compressions of samples having different contact parameters. Such simulations were performed using `YADE <http://yade-dem.org/>`_ that outputs the simulation state to a .npy file every given amount of time steps.

* Run `rnn/data_parsing/parse_data_YADE.py` to read the .npy files in ``data_dir`` and create ``target_file`` with format hdf5. The ``target_file`` can be placed to folder `rnn/data`.
   
* ``pressures`` and ``experiment_types`` are subfolders of our database and they will become fields in ``target_file``. For more information about the parameters take a look at the API documentation. TODO

* If your data comes from another software or is stored differently please write your own parser such that the format of ``target_file`` has the same structure as the one given as example.

* Finally, copy or move the generated hdf5 file to the folder `rnn/data`, if it is not already there.

Configuration of your model and training procedure
``````````````````````````````````````````````````

In the main function of `train.py` there is a dictionary `defaults` containing several values that configure the model and training procedure. We encourage you to define them all, otherwise default values defined in each function will be used.
Check these API docs to understand the meaning of each entry. TODO

**Option 1:** Train using wandb
```````````````````````````````
`Weights a Biases <https://wandb.ai/site>`_ is an external platform that can be used for tracking experiments and hyperparameter tuning. It allows the user to gather training metrics, model configuration and system performance for different runs (i.e. training of your RNN).

To use it you have to create a free account. If you have installed grainLearning with rnn dependencies, ``wandb`` should be already in your system, otherwise, you can install it: ``pip install wandb``.

Experiment tracking: Single run
::::::::::::::::::::::::::::::::

- Check that in the main function of `train.py` the function ``train(config=defaults)`` is called and ``#train_without_wandb(config=defaults)`` is commented out.
- In the environment where grainLearning and rnn dependencies has been installed, run: ``python train.py``

If is the first time running wandb it will ask you to login (copy paste your API key that you'll find in your wandb profile).

Hyperparameter optimization: Sweep
:::::::::::::::::::::::::::::::::::

`Wandb Sweeps <https://wandb.ai/site/sweeps>`_ allows the user to train the model with different *hyperparameters combinations* gathering metrics in the wandb interface to facilitate the analysis and choice of the best model.

1. Configure your sweep:
   In folder *sweep* you can find an example of the configuration file (example_sweep.yaml) containing the sweep configuration values and/or range of values per each hyperparameter. Here you have complete freedom to chose how many values and in which ranges are you going to search for the optimal combination.
   Don't forget to put your own project and entity to get the results in your wandb dashboard.

   **Note:** The combination of values of the parameter that wandb is going to draw for each run will override those of the `default` dictionary in `train.py`.

   For more information about how to configure the .yaml file see `this <https://docs.wandb.ai/guides/sweeps/define-sweep-configuration>`_. 
2. Check that in the main function of `train.py` the function ``train(config=defaults)`` is called and ``#train_without_wandb(config=defaults)`` is commented out.
3. Open a terminal with your environment where grainLearning and rnn dependencies are installed.
4. If you are running the training in a supercomputer continue with the instructions in `Running a Sweep on HPC`_.
5. Create a sweep: ``wandb sweep example_sweep.yaml``
    This will print out in the console the sweep ID as well as the instructions to start an agent.

6. Run an agent: ``wandb agent <entity>/<project>/<sweep_id>``
    Running this command will start a training run with hyperparameters chosen according to `example_sweep.yaml`, will keep starting new runs, and will update your wandb dashboard. Models are saved both locally and also uploaded to wandb.

Running a Sweep on HPC
''''''''''''''''''''''
This instructions assume that your HPC platform uses job scheduler slurm. `run_sweep.sh` configures the job and loads modules from **Snellius**, these can be different in other supercomputers.

1. Install grainLearning and rnn dependencies.  
2. Check that `run_sweep.sh` load the correct modules. In this file the outputs of the job will be directed to `job_outputs`. It can be that in your HPC such folder is not automatically created and thus, you have to do it in advance.
3. Run your job: ``sbatch run_sweep.sh``
   This command will create the sweep, gather the sweep_id from the output that is printed on the terminal and then start an agent.

**Option 2:** Train using plain tensorflow 
``````````````````````````````````````````
- Check that in the main function of `train.py` the function   ``train_without_wandb(config=defaults)`` is called and ``#train(config=defaults)`` is commented out.
- In the environment where grainLearning and rnn dependencies has been installed, run: ``python train.py``

Make a prediction with a pre-trained model
------------------------------------------

Use a trained RNN in grainLearning calibration process
------------------------------------------------------