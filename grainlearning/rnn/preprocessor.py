import h5py
import numpy as np
import tensorflow as tf
import warnings

from abc import ABC, abstractmethod

import grainlearning.rnn.windows as windows

class Preprocessor(ABC):

    def __init__(self):
        pass


    @abstractmethod
    def prepare_datasets(self, **kwargs):
        """
        Abstract class to be implemented in each of the child classes.
        Must return a tuple containing:

          * ``data_split``: Dictionary with keys `'train'`, `'val'`, `'test'`, and values of the corresponding tensorflow Datasets.
          * ``train_stats``: Dictionary containing the shape of the data:
                  ``sequence_length``, ``num_input_features``, ``num_contact_params``, ``num_labels``,
                  and `'mean'` and `'std'` of the training set, in case ``standardize_outputs`` is True.
        """
        # Throw an error because this should be always implemented in the child class (?)
        pass

    @classmethod
    def warning_config_field(cls, key: str, config: dict, default, add_default_to_config: bool = False):
        """
        Raises a warning if key is not included in config dictionary.
        Also informs the default value that will be used.
        If add_default_to_config=True, then it adds the key and its default value to config.

        :param key: string of the key in dictionary ``config`` to look for.
        :param config: Dictionary containing keys and values
        :param default: default value associated to the key in config.
        :param add_default_to_config: Wether to add the default value and the key to config in case it does not exists.

        :return: Updated ``config`` dictionary.
        """
        # customized warning to print -only- the warning message
        def _custom_format_warning(msg, *_):
            return str(msg) + '\n' # ignore everything except the message

        warnings.formatwarning = _custom_format_warning

        if key not in config:
            if add_default_to_config: config[key] = default
            warnings.warn(f"No {key} specified in config, using default {default}.")

        return config

    @classmethod
    def make_splits(cls, dataset: tuple, train_frac: float, val_frac: float, seed: int):
        """
        Split data into training, validation, and test sets.
        The split is done on a sample by sample basis, so sequences are not broken up.
        It is done randomly across all pressures and experiment types present.

        :param dataset: Full dataset to split on, in form of a tuple (inputs, outputs),
                where inputs is a dictionary and its keys and the outputs are numpy arrays.
        :param train_frac: Fraction of data used for training set.
        :param val_frac: Fraction of data used for validation set. Test fraction is the remaining.
        :param seed: Random seed used to make the split.

        :return: Dictionary containing `'train'`, `'val'`, and `'test'` datasets.
        """
        n_tot = dataset[1].shape[0]
        n_train = int(train_frac * n_tot)
        n_val = int(val_frac * n_tot)

        if train_frac + val_frac > 1:
            raise ValueError(f"Fractions of training {train_frac} and validation {val_frac} are together bigger than 1.")
        if n_val <= 0: # error if not enough samples in validation dataset
            raise ValueError(f"Fractions of training and validation lead to have {n_val} samples in validation dataset.")
        if n_train + n_val >= n_tot:
            raise ValueError(f"Fractions of training {train_frac} and validation {val_frac}"
                             f"lead to have {n_tot - n_train - n_val} samples in test dataset.")

        np.random.seed(seed=seed)
        inds = np.random.permutation(np.arange(n_tot))
        i_train, i_val, i_test = inds[:n_train], inds[n_train:n_train + n_val], inds[n_train + n_val:]
        split_data = {
                     'train': cls.get_split(dataset, i_train),
                     'val': cls.get_split(dataset, i_val),
                     'test': cls.get_split(dataset, i_test),
                     }
        return split_data

    @classmethod
    def get_split(cls, dataset:tuple, inds: np.array):
        """
        Given a dataset and the indexes, return  the sub-dataset containing only the elements of indexes in ``inds``.
        The returned dataset respects the format of ``inputs`` and ``outputs`` (see more info in return).

        :param dataset: Full dataset to split on, in form of a tuple ``(inputs, outputs)``,
          where ``inputs`` is a dictionary and its keys and ``outputs`` are numpy arrays.
        :param inds: Indexes of the elements in ``dataset`` that are going to be gathered in this specific split.

        :return: tuple of the split, 2 dimensions:
          * inputs: dictionary containing `'load_sequence'` and `'contact_params'`.
          * outputs: Labels
        """
        X = {key: tf.gather(val, inds) for key, val in dataset[0].items()}
        y = tf.gather(dataset[1], inds)

        return X, y

    @classmethod
    def pad_initial(cls, array: np.array, pad_length: int, axis=1):
        """
        Add ``pad_length`` copies of the initial state in the sequence to the start.
        This is used to be able to predict also the first timestep from a window
        of the same size.

        :param array: Array that is going to be modified
        :param pad_lenght: number of copies of the initial state to be added at the beggining of ``array``.

        :return: Modified array
        """
        starts = array[:, :1, :]
        padding = tf.repeat(starts, pad_length, axis=axis)
        padded_array = tf.concat([padding, array], axis)

        return padded_array

    @classmethod
    def standardize_outputs(cls, split_data: dict):
        """
        Standardize outputs of ``split_data`` using the mean and std of the training data
        taken over both the samples and the timesteps. The 3 datasets `'train'`, `'val'`, `'test'` will be standardized.

        :param split_data: dictionary containing `'train'`, `'val'`, `'test'` keys and the respective datasets.

        :return: Tuple containing:

          * ``standardized_splits``: Dictionary containing the standardized datasets.
          * ``train_stats``: Dictionary with the metrics (mean and standard deviation)
            used to standardize the data.
        """
        train_outputs = split_data['train'][1]
        mean = np.mean(train_outputs, axis=(0, 1))
        std = np.std(train_outputs, axis=(0, 1))
        train_stats = {'mean': mean, 'std': std}

        def _standardize(x, y):
            return x, (y - mean) / std

        standardized_splits = split_data
        for split in ['train', 'val', 'test']:
            standardized_splits[split] = _standardize(*split_data[split])

        return standardized_splits, train_stats

    @classmethod
    def get_dimensions(cls, data: tf.data.Dataset):
        """
        Extract dimensions of sample from a tensorflow dataset.

        :param data: The dataset to extract from.

        :return: Dictionary containing:
                sequence_length, num_load_features, num_contact_params, num_labels
        """
        train_sample = next(iter(data))  # just to extract a single batch
        sequence_length, num_load_features = train_sample[0]['load_sequence'].shape
        num_contact_params = train_sample[0]['contact_parameters'].shape[0]
        num_labels = train_sample[1].shape[-1]

        return {'sequence_length': sequence_length,
                'num_load_features': num_load_features,
                'num_contact_params': num_contact_params,
                'num_labels': num_labels,
                }


class Preprocessor_Triaxial_Compression(Preprocessor):
    """
    Class to Preprocess data of triaxial compresion experiments, inheriting from abstract class :class:`Preprocessor`

    Attributes: see :meth:`Preprocessor_Triaxial_Compression.get_default_config`

    """
    def __init__(self, **kwargs):
        """
        Constructor of the preprocessor for Triaxial Compression experiment.
        Initializes the values of required attributes to preprocess the data, from kwargs:

        :param raw_data: Path to hdf5 file containing the data.
        :param pressure: Experiment confining Pressure as a string or `'All'`.
        :param experiment_type: Either `'drained'`, `'undrained'` or `'All'`.
        :param train_frac: Fraction of data used in the training set.
        :param val_frac: Fraction of the data used in the validation set.
        :param pad_length: Amount by which to pad the sequences from the start.
        :param window_size: Number of timesteps to include in a window.
        :param window_step: Offset between subsequent windows.
        :param standardize_outputs: Whether to transform the training set labels
                                    to have zero mean and unit variance.
        :param add_e0: Whether to add the initial void ratio as a contact parameter.
        :param add_pressure: Wheter to add the pressure to contact parameters.
          If True, the pressure is normalized by 10**6.

        .. note:: Teh parameters that will not be explicitly specified with its value in ``kwargs`` will take the defaults from :meth:`Preprocessor_Triaxial_Compression.get_default_config`.

        :param add_experiment_type: Wheter to add the experiment type to contact parameters.
        """
        config = self.check_config(kwargs)

        # creates an attribute per each element in config
        for key, value in config.items():
            setattr(self, key, value)

    @classmethod
    def get_default_config(cls):
        """
        Returns a dictionary with default values for the configuration of data preparation. Possible fields are:

        * `'raw_data'`: Path to hdf5 file generated using parse_data_YADE.py
        * `'pressure'` and `'experiment_type'`: Name of the subfield of dataset to consider. It can also be 'All'.
        * `'standardize_outputs'`: If True transform the data labels to have zero mean and unit variance.
          Also, in train_stats the mean and variance of each label will be stored,
          so that can be used to transform predicitons.
          (This is very usful if the labels are not between [0,1])
        * `'add_e0'`: Whether to add the initial void ratio (output) as a contact parameter.
        * `'add_pressure'`: Wether to add the pressure to contact_parameters.
        * `'add_experiment_type'`: Wether to add the experiment type to contact_parameters.
        * `'train_frac'`: Fraction of the data used for training, between [0,1].
        * `'val_frac'`: Fraction of the data used for validation, between [0,1].
          The fraction of the data used for test is then ``1 - train_frac - val_frac``.
        * `'window_size'`: int, number of steps composing a window.
        * `'window_step'`: int, number of steps between consecutive windows (default = 1).
        * `'pad_length'`: int, equals to ``window_size``. Length of the sequence that with be pad at the start.

        :return: Dictionary containing default values of the arguments that the user can set.
        """
        return {
            'raw_data': 'data/sequences.hdf5',
            'pressure': 'All',
            'experiment_type': 'All',
            'add_e0': False,
            'add_pressure': True,
            'add_experiment_type': True,
            'train_frac': 0.7,
            'val_frac': 0.15,
            'window_size': 10,
            'window_step': 1,
            'pad_length': 0,
            'standardize_outputs': True
        }

    @classmethod
    def check_config(cls, config:dict):
        """
        Checks that values requiring an input from the user would be specified in config.

        :param config: Dictionary containing the values of different arguments.

        :return: Updated config dictionary.
        """
        # Note: I systematically use config.keys() instead of in config, because config can be a dict from wandb.
        # This object behaves differently than python dict (might be just the version), but this solves it.
        if 'raw_data' not in config.keys(): raise ValueError("raw_data has not been defined in config")

        # Warning that defaults would be used if not defined and adding them to config because is required in other functions.
        keys_to_check = ['pressure', 'experiment_type', 'standardize_outputs',
                         'add_e0', 'add_pressure', 'add_experiment_type',
                         'window_size', 'pad_length', 'train_frac', 'val_frac']
        defaults = cls.get_default_config()
        for key in keys_to_check:
            config = super().warning_config_field(key, config, defaults[key], add_default_to_config=True)

        # Deleting elements that are not in default
        keys_to_delete = set(config) - set(defaults)
        for key in keys_to_delete: del config[key]

        return config


    def prepare_datasets(self, seed: int = 42):
        """
        Convert raw data into preprocessed split datasets.
        First split the data into `train`, `val` and `test` datasets
        and then apply the `Sliding windows` transformation.
        This is to avoid having some parts of a dataset in `train` and some in `val` and/or in `test` (i.e. data leak).

        :param seed: Random seed used to split the datasets.

        :return: Tuple (split_data, train_stats)

          * ``split_data``: Dictionary with keys `'train'`, `'val'`, `'test'`, and values the
            corresponding tensorflow Datasets.

          * ``train_stats``: Dictionary containing the shape of the data:
            ``sequence_length``, ``num_load_features``, ``num_contact_params``, ``num_labels``,
            and `'mean'` and `'std'` of the training set, in case ``standardize_outputs`` is True.
        """
        with h5py.File(self.raw_data, 'r') as datafile: # Will raise an exception in File doesn't exists
            inputs, outputs, contacts = self._merge_datasets(datafile)

        if self.add_e0:
            contacts = self._add_e0_to_contacts(contacts, outputs)

        if self.pad_length > 0:
            inputs = super().pad_initial(inputs, self.pad_length)
            outputs = super().pad_initial(outputs, self.pad_length)

        dataset = ({'load_sequence': inputs, 'contact_parameters': contacts}, outputs)
        split_data = super().make_splits(dataset, self.train_frac, self.val_frac, seed)

        if self.standardize_outputs:
            split_data, train_stats = super().standardize_outputs(split_data)
        else:
            train_stats = {}

        split_data = {key: tf.data.Dataset.from_tensor_slices(val) for key, val in split_data.items()}
        train_stats.update(super().get_dimensions(split_data['train']))
        split_data = windows.windowize_train_val_test(split_data, self.window_size, self.window_step)

        return split_data, train_stats


    def _merge_datasets(self, datafile: h5py._hl.files.File):
        """
        Merge the datasets with different pressures and experiment types.
        If ``pressure`` or ``experiment_type`` is `'All'`.
        Otherwise just return the inputs, outputs and contact_params for the given pressure and experimen_type.

        :param datafile: h5py file containing the dataset.

        :return: input, output and contact_params arrays merged for the given pressures and expriment_types.
        """
        if self.pressure == 'All': pressures = list(datafile.keys()) # this considers pressure as the first group of the dataset.
        else: pressures = [self.pressure]

        input_sequences, output_sequences, contact_params = ([] for _ in range(3))
        for pres in pressures:
            if self.experiment_type == 'All': experiment_types = list(datafile[pres].keys())
            else: experiment_types = [self.experiment_type]

            for exp_type in experiment_types:
                data = datafile[pres][exp_type]
                input_sequences.append(data['inputs'][:])
                output_sequences.append(data['outputs'][:])

                cps = data['contact_params'][:]
                cps = self._augment_contact_params(cps, pres, exp_type)
                contact_params.append(cps)

        input_sequences = np.concatenate(input_sequences, axis=0)
        output_sequences = np.concatenate(output_sequences, axis=0)
        contact_params = np.concatenate(contact_params, axis=0)

        return input_sequences, output_sequences, contact_params


    def _add_e0_to_contacts(self, contacts: np.array, outputs: np.array):
        """
        Add the initial void ratio e_0 as an extra contact parameter at the end.

        :param contacts: List of contact parameters
        :param inputs: List of input parameters

        :return: Modified contacts list with e_0 added at the end.
        """
        e0s = outputs[:, 0, 0]  # first element in series, 0th feature == e_0
        e0s = np.expand_dims(e0s, axis=1)
        contacts = np.concatenate([contacts, e0s], axis=1)

        return contacts


    def _augment_contact_params(self, contact_params: np.array, pressure: str, experiment_type: str):
        """
        Add the pressure and the experiment type as contact parameters.
        Pressure is divided by 10**6, i.e. '0.3e6' becomes 0.3.
        Experiment type is converted to 1 for drained and 0 for undrained.

        :param contact_params: Array containing contact parameters for all the
                samples with the given pressure and experiment type.
        :param pressure: The corresponding pressure.
        :param experiment_type: The corresponding experiment type: `'drained'` or `'undrained'`.

        :return: Numpy array containing augmented contact parameters.
        """
        new_info = []
        pres_num = float(pressure) / 10**6
        if self.add_pressure: new_info.append(pres_num)
        if self.add_experiment_type: new_info.append(self.experiment_type == 'drained')

        num_samples = contact_params.shape[0]
        new_info = np.expand_dims(new_info, 0)
        new_info = np.repeat(new_info, num_samples, axis=0)

        return np.concatenate([contact_params, new_info], axis=1)
