"""
Script to preprocess DEM data so that it can be used as input in an RNN.
In YADE data is stored to .npy files every interval of time steps.

Data consists of:

* contact_params (samples, c).
* input_params (samples, sequence_length, x).
* output_params (samples, sequence_length, y).

This example considers DEM simulations of Triaxial compressions
at different confinments (pressures), in drain and undrained conditions.
"""
import os, h5py
import numpy as np

CONTACT_KEYS = [
    'E',   # young's modulus  = log_10(E) with E in Pa
    'v',   # poisson's ratio
    'kr',  # rolling stiffness
    'eta', # rolling friction
    'mu'  # sliding friction
    ]

INPUT_KEYS_UNDRAINED = [
    'e_x', # strains in 3 directions
    'e_y',
    'e_z' # the axial direction
]

INPUT_KEYS_DRAINED = [
    'e_z' # the axial direction
]

OUTPUT_KEYS = [
    'e',   # void ratio
    'p',   # mean stress
    'q',   # deviatoric stress
    'f0',  # average contact normal force
    'a_c', # fabric anisotropy
    'a_n', # mechanical anisotropy
    'a_t' # mechanical anisotropy due to tangential forces
]

UNUSED_KEYS_SEQUENCE = [
    'dt',  # size of timestep taken at this iteration
    'numIter', # iteration number in the simulation at which equilibrium is
               # reached at the current loading
    'K',   # mysterious
    'A'   # also mysterious
]

UNUSED_KEYS_CONSTANT = [
    'conf',# confining pressure (stored as group name) = log_10(confinement_pressure)
    'mode',# drained/undrained (stored as group name)
    'num' # number of particles
]

def convert_all_to_hdf5(
        pressures: list,
        experiment_types: list,
        data_dir: str,
        target_file: str,
        sequence_length: int,
        stored_in_subfolders: bool
        ):
    """
    Merge data of experiments of different pressures and types into a single
    hdf5 file.

    :param pressures: List of strings of pressures available.
    :param experiment_types: List of strings of experiment types available.
    :param data_dir: Root directory containing all the data.
    :param target_file: Path to hdf5 file to be created.
    :param sequence_length: Expected number of time steps in sequences. If the sequence is
      longer, only the first ``sequence_length`` element will be considered.
    :param stored_in_subfolders: True if yade .npy files are stored in pressure/experiment_type
      tree structure. False if all files are in a single folder `data_dir`.

    .. warning:: Will remove `target_file` if already exists.
    """
    if os.path.exists(target_file):
        os.remove(target_file)

    with h5py.File(target_file, 'a') as f:
        f.attrs['inputs_drained'] = INPUT_KEYS_DRAINED
        f.attrs['inputs_undrained'] = INPUT_KEYS_UNDRAINED
        f.attrs['outputs'] = OUTPUT_KEYS
        f.attrs['contact_params'] = CONTACT_KEYS
        f.attrs['unused_keys_sequence'] = UNUSED_KEYS_SEQUENCE
        f.attrs['unused_keys_constant'] = UNUSED_KEYS_CONSTANT

        for pressure in pressures:
            for experiment_type in experiment_types:
                grp = f.require_group(f'{pressure}/{experiment_type}')
                inputs_tensor, contact_tensor, outputs_tensor = \
                        convert_to_arrays(pressure, experiment_type,
                                sequence_length, data_dir, stored_in_subfolders)
                grp['contact_params'] = contact_tensor
                grp['inputs'] = inputs_tensor
                grp['outputs'] = outputs_tensor

    print(f"Added all data to {target_file}")


def convert_to_arrays(
        pressure: str,
        experiment_type: str,
        sequence_length: int,
        data_dir: str,
        stored_in_subfolders: bool
        ):
    """
    For a given pressure and experiment type, read all the files in the corresponding
    directory and concatenate those with the expected sequence length together
    into numpy arrays.

    :param pressure: String indicating the pressure used.
    :param experiment_type: String indicating the experiment type ('drained', 'undrained')
    :param sequence_length: Expected number of timesteps in sequence. If the sequence is
      longer, only the first ``sequence_length`` element will be considered.
    :param data_dir: The root directory of the data.
    :param stored_in_subfolders: True if yade .npy files are stored in pressure/experiment_type
      tree structure. False if all files are in a single folder `data_dir`.

    :return: Tuple of arrays of inputs, contacts, outputs

    .. warning:: sequences longer and shorter than `sequence_length` are ignored.
    """
    if stored_in_subfolders: data_dir = data_dir + f'{pressure}/{experiment_type}/'
    if not os.listdir(data_dir):
        print(f"Directory {data_dir} is empty.")
        return

    file_names = [fn for fn in os.listdir(data_dir) if fn.endswith('.npy')]

    # rescale pressures by 10**6 to make it order 1.
    scalings = {key: 1. for key in OUTPUT_KEYS}
    scalings['p'] = scalings['q'] = 1.e6

    contact_list, inputs_list, outputs_list, other_lengths = ([] for i in range(4))
    for f in file_names:
        try:
            sim_params, sim_features = np.load(data_dir + f, allow_pickle=True)
        except:
            print('IOError', f, pressure)
            continue

        if not stored_in_subfolders:
            if (str(10**sim_params['conf']) != pressure) or (sim_params['mode'] != experiment_type):
                continue

        # test if sequence is of full length
        test_features = sim_features[OUTPUT_KEYS[0]]
        if len(test_features) >= sequence_length:
            contact_params = [sim_params[key] for key in CONTACT_KEYS]
            contact_list.append(contact_params)
            if experiment_type == 'drained':
                inputs_list.append([sim_features[key][:sequence_length] for key in INPUT_KEYS_DRAINED])
            elif experiment_type == 'undrained':
                inputs_list.append([sim_features[key][:sequence_length] for key in INPUT_KEYS_UNDRAINED])
            else: raise ValueError(f"experiment type must be drained or undrained but got {experiment_type}")
            outputs_list.append([np.array(sim_features[key][:sequence_length]) / scalings[key] for key in OUTPUT_KEYS])
        else:
            other_lengths.append(len(test_features))

    print(f'At confining pressure {pressure}, for the {experiment_type} case, '
          f'there are {len(other_lengths)} samples with a different sequence length.')
    print(f'lengths: ')
    print(other_lengths)

    inputs_array = np.array(inputs_list)
    contact_array = np.array(contact_list)
    outputs_array = np.array(outputs_list)

    # keras requires (batch, sequence_length, features) shape, so transpose axes
    inputs_array = np.transpose(inputs_array, (0, 2, 1))
    outputs_array = np.transpose(outputs_array, (0, 2, 1))

    print(f'Created array of {outputs_array.shape[0]} samples,')

    return inputs_array, contact_array, outputs_array


def get_pressures(data_dir: str):
    """
    From a list of .npy YADE files get the confinement pressures, existing in the dataset.

    :param data_dir: Path to the data (YADE .npy files)

    :return: List of confinement pressures present in the dataset.
    """
    if not os.listdir(data_dir):
        print(f"Directory {data_dir} is empty.")
        return

    pressures = []
    file_names = [fn for fn in os.listdir(data_dir) if fn.endswith('.npy')]
    for f in file_names:
        try:
            sim_params, sim_features = np.load(data_dir + f, allow_pickle=True)
        except:
            print('IOError', f, pressure)
            continue
        pressure = str(10**sim_params['conf'])
        if not (pressure in pressures): pressures.append(pressure)

    return pressures


def main():
    # Option 1: YADE .npy files stored in subfolders pressure/experiment_type
    pressures = ['0.2e6', '0.5e6', '1.0e6']
    experiment_types = ['undrained','drained']
    stored_in_subfolders = True

    # Option 2: YADE .npy files are stored in a single folder.
    pressures = get_pressures('./')
    experiment_types = ['drained']
    stored_in_subfolders = False

    # Call main function
    convert_all_to_hdf5(
            pressures=pressures,
            experiment_types=experiment_types,
            data_dir='/Users/aronjansen/Documents/grainsData/TriaxialCompression/',
            sequence_length=200,
            target_file='data/sequences.hdf5',
            stored_in_subfolders=stored_in_subfolders
        )


if __name__ == '__main__':
    main()