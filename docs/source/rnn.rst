RNN Module
==========

We implemented a `Recurrent Neural Network (RNN) <https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-recurrent-neural-networks>`_ model in tensorflow framework.

The RNN model
-------------


The RNN takes as inputs a vector of *parameters* (i.g. contact parameters) and a sequence of size :math:`\mathcal{N}` (i.g. applied strain). The model returns a sequence of size :math:`\mathcal{N}`.

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

For both single runs and sweeps, wandb will create a folder named `wandb` containing metadata and files generated during the run(s). In this same folder, per each run, you will find 3 files: `configuration.yaml`, `train_stats_npy` and `model-best.h5`. These files contain all the information required to load your model in the future. 

Experiment tracking: Single run
::::::::::::::::::::::::::::::::
Create `my_train.py` where you would like to run the training. Be aware to configure the data directory accordingly (See API docs for more information about the config keys). Avoid creating this file inside the grainlearning package nor rnn module.

.. code-block:: python
   :caption: my_train.py

   import grainlearning.rnn.train as train_rnn

   # 1. Create my dictionary of configuration
   my_config = train_rnn.get_default_dict()
   
   # 2. Run the training using bare tensorflow
   train_rnn.train(config=my_config)

Open a terminal where you have your file, activate the environment where grainLearning and rnn dependencies has been installed and run: ``python my_train.py``

If is the first time running wandb it will ask you to login (copy paste your API key that you'll find in your wandb profile).

Hyperparameter optimization: Sweep
:::::::::::::::::::::::::::::::::::

`Wandb Sweeps <https://wandb.ai/site/sweeps>`_ allows the user to train the model with different *hyperparameters combinations* gathering metrics in the wandb interface to facilitate the analysis and choice of the best model.

You can run your sweep:

- `From a python file`_.
- `From the command line`_.

From a python file
''''''''''''''''''

Create `my_sweep.py` where you would like to run the training. Configure the sweep parameters (See API docs for more information about the config keys). Avoid creating this file inside the grainlearning package nor rnn module. See `this <https://docs.wandb.ai/guides/sweeps/define-sweep-configuration>`_ for more information about sweep configuration, and `this wandb guide <https://docs.wandb.ai/guides/sweeps/quickstart>`_.

.. code-block:: python
   :caption: my_sweep.py

   import wandb
   import grainlearning.rnn.train as train_rnn

   wandb.login()
   sweep_configuration = {
    'method': 'bayes',
    'name': 'sweep',
    'metric': {'goal': 'maximize', 'name': 'val_acc'},
    'parameters':
      {
    	'raw_data': {'value': 'data/sequences.hdf5'},
    	'use_windows': {'value': True},
  		'window_size': {'values': [5, 10, 20, 30]},
  		'window_step': {'value': 1},
  		'conditional': {'value': True},
  		'standardize_outputs': {'value': True},
    	'lstm_units': {'values': [50, 100, 150, 200]},
    	'dense_units': {'values': [20, 50, 100, 150]},
        'batch_size': {'values': [128, 256, 512]},
        'epochs': {'value': 2},
        'learning_rate': {'max': 0.1, 'min': 0.0001},
        'patience': {'value': 5},
        'save_weights_only': {'value': False}
      }
   }
   
   # create a new sweep, here you can also configure your project and entity.
   sweep_id = wandb.sweep(sweep=sweep_configuration)

   # run an agent
   wandb.agent(sweep_id, function=train_rnn.train, count=4)

Open a terminal where you have your file, activate the environment where grainLearning and rnn dependencies has been installed and run: ``python my_sweep.py``.

If you want to run another agent or re-start the sweep you can replace the creation of a new step sweep for assigning the id of your sweep to the variable ``sweep_id``.

From the command line
'''''''''''''''''''''

1. Configure your sweep:
   In folder *sweep* you can find an example of the configuration file `example_sweep.yaml` containing the sweep configuration values and/or range of values per each hyperparameter. Here you have complete freedom to chose as many values and in which ranges wandb will search for the optimal combination.
   Don't forget to put your own project and entity to get the results in your wandb dashboard. For more information about how to configure the .yaml file see `this <https://docs.wandb.ai/guides/sweeps/define-sweep-configuration>`_. 

   .. note:: The combination of values of the parameter that wandb is going to draw for each run will override those of the `default` dictionary in `train.py`.
2. Create a copy of this file outside grainlearning package and rnn module, in the folder where you want to run your sweep. `wandb`` folder containing the runs information an model data will be automatically created in this folder. Change ``raw_data`` value accordingly.  
3. Create python file `my_sweep_CL.py` and in `example_sweep.yaml` set ``program: my_sweep_CL.py``.
    
.. _my sweep CL:
.. code-block:: python
   :caption: my_sweep_CL.py

   import grainlearning.rnn.train as train_rnn
   train_rnn.train()

4. Open a terminal and activate the environment where grainLearning and rnn dependencies are installed.
5. If you are running the training in a supercomputer continue with the instructions in `Running a Sweep on HPC`_.
6. Create a sweep: ``wandb sweep example_sweep.yaml``.
   
   This will print out in the console the sweep ID as well as the instructions to start an agent.
7. Run an agent: ``wandb agent <entity>/<project>/<sweep_id>``.
   
   Running this command will start a training run with hyperparameters chosen according to `example_sweep.yaml`, will keep starting new runs, and will update your wandb dashboard. Models are saved both locally and also uploaded to wandb.

Running a Sweep on HPC
''''''''''''''''''''''
This instructions assume that your HPC platform uses job scheduler slurm. `run_sweep.sh` configures the job and loads modules from **Snellius**, these can be different in other supercomputers.

1. Install grainLearning and rnn dependencies.  
2. Create the folder containing your data, `run_sweep.sh`, file :ref:`my_sweep_CL.py <my sweep CL>` and `example_sweep.yaml`, make sure to modify the last one accordingly.
3. Check that `run_sweep.sh` load the correct modules. In this file the outputs of the job will be directed to `job_outputs`. It can be that in your HPC such folder is not automatically created and thus, you have to do it in advance.
4. Run your job: ``sbatch run_sweep.sh``
   This command will create the sweep, gather the sweep_id from the output that is printed on the terminal and then start an agent.

**Option 2:** Train using plain tensorflow 
``````````````````````````````````````````
Create `my_train.py` where you would like to run the training. Be aware to configure the data directory accordingly. Avoid creating this file inside the grainlearning package nor rnn module.

.. code-block:: python
   :caption: my_train.py

   import grainlearning.rnn.train as train_rnn

   # 1. Create my dictionary of configuration
   my_config = train_rnn.get_default_dict()
   
   # 2. Run the training using bare tensorflow
   train_rnn.train_without_wandb(config=my_config)

Open a terminal where you have your file, activate the environment where grainLearning and rnn dependencies has been installed and run: ``python my_train.py``

The folder `outputs` is created and contains `config.npy`, `train_Stats.npy` and  either `saved_model.pb` or `weights.h5` depending if you choose to save the entire model or only its weights. The contents of this directory will be necessary to load the trained model in the future.

.. warning:: Every time you run a new experiment  the files in `outputs` will be override. If you want to save them, copy them to another location once the run is finished.
  
Make a prediction with a pre-trained model
------------------------------------------

You can load a pre-trained model from:

- `Saved model`_. 
- `A wandb sweep`_.

Saved model
```````````

You can find some pre-trained models in in `rnn/train_models` and you can also load a model that you have trained. The function ``get_pretrained_model()`` will take care of checking if your model was trained via wandb or outside of it, as well as if only the weights were saved or the entire model.

In this example, we are going to load the same dataset that we used for training, but we are going to predict from the `test` sub-dataset. Here you're free to pass any data having the same format (tf.data.Dataset) and input dimensions to the model: () 

.. code-block:: python
   :caption: predict_from_pre-trained.py

   from pathlib import Path

   import grainlearning.rnn.predict as predict_rnn
   from grainlearning.rnn.preprocessing import prepare_datasets

   # 1. Define the location of the model to use
   path_to_trained_model = Path('C:/GrainLearning/grainLearning/grainlearning/rnn/trained_models/My_model_1')

   # 2. Get the model information
   model, train_stats, config = predict_rnn.get_pretrained_model(path_to_trained_model)

   # 3. Load input data to predict from
   config['raw_data'] = '../train/data/sequences.hdf5'
   data, _ = prepare_datasets(**config)

   #4. Make a prediction
   predictions = predict_rnn.predict_macroscopics(model, data['test'], train_stats, config,batch_size=256, single_batch=True)

If the model was trained with ``standardize_outputs = True``, ``predictions`` are going to be unstandardized (i.e. no values between [0,1] but with the original scale). 
In our example, ``predictions`` is a tensorflow tensor of size ``(batch_size, length_sequences - window_size, 7)``.

A wandb sweep
`````````````
You need to have access to the sweep and know its ID.
Often this looks like `<entity>/<project>/<sweep_id>`.

.. code-block:: python
   :caption: predict_from_sweep.py

   from pathlib import Path

   import grainlearning.rnn.predict as predict_rnn
   from grainlearning.rnn.preprocessing import prepare_datasets

   # 1. Define which sweep to look into
   entity_project_sweep_id = 'grainlearning-escience/grainLearning-grainlearning_rnn/6zrc0vjb'

   # 2. Chose the best model from a sweep, and get the model information
   model, data, train_stats, config = predict_rnn.get_best_run_from_sweep(entity_project_sweep_id)

   # 3. Load input data to predict from
   config['raw_data'] = '../train/data/sequences.hdf5'
   data, _ = prepare_datasets(**config)

   #4. Make a prediction
   predictions = predict_rnn.predict_macroscopics(model, data['test'], train_stats, config,batch_size=256, single_batch=True)

This can fail if you have deleted some runs or if your wandb folder is not present in this folder. We advise to copy `config.yaml`, `train_stats.py` and `model_best.h5` from `wandb/runXXX/files` to another location and follow `Saved model`_ instructions. These files can also be downloaded from the wandb dashboard.

Use a trained RNN in grainLearning calibration process
------------------------------------------------------