"""Test the Bayesian claibration module."""
from os import path
from math import floor, log
from shutil import rmtree
import numpy as np
from grainlearning import BayesianCalibration, DynamicSystem, IODynamicSystem, IterativeBayesianFilter
from grainlearning.tools import write_dict_to_file

PATH = path.abspath(path.dirname(__file__))
GL_PATH = path.abspath(path.join(__file__, "../../.."))


def test_run_with_dynamic_system():
    """Test if the callback function works as expected"""

    def run_sim(calib):
        calib.system.sim_data = [
            [[12, 3, 4, 4]],
            [[11, 24, 4, 3]],
        ]  # must be of shape (num_samples,num_obs,num_steps), thus adding a dummy dimension

    config = {
        "param_min": [1, 2],
        "param_max": [3, 4],
        "obs_data": [2, 4, 6, 7],
        "ctrl_data": [1, 2, 3, 4],
        "inv_obs_weigh": [0.5, 0.25],
        "num_samples": 2,
    }
    system_cls = DynamicSystem.from_dict(config)

    bayesian_calibration_cls = BayesianCalibration(system_cls, IterativeBayesianFilter(), callback=run_sim)

    bayesian_calibration_cls.run_callback()

    np.testing.assert_almost_equal(
        bayesian_calibration_cls.system.sim_data,
        [
            [[12, 3, 4, 4]],
            [[11, 24, 4, 3]],
        ],
    )


def test_run_with_io_dynamic_system():
    """Test if the callback function works as expected"""

    def run_sim(calib):
        """Run the simulation"""
        # Initialize the data list
        data = []
        system = calib.system
        # Get the ctrl data
        x = system.ctrl_data
        # Get the simulation name
        sim_name = system.sim_name
        # Loop over the parameter samples
        for i, param in enumerate(system.param_data):
            # Get the description of the current sample
            mag = floor(log(system.num_samples, 10)) + 1
            description = 'Iter' + str(system.curr_iter) + '_Sample' + str(i).zfill(mag)
            # Run the model
            y = param[0] + param[1] * x + param[2] * x ** 2 + param[3] * x ** 3
            # Append the data to the list
            data.append(np.array(y, ndmin=2))
            # Write the data to a file
            data_file_name = f'{sim_name}_' + description + '_sim.txt'
            write_dict_to_file({'f': list(y)}, data_file_name)
            # Write the parameters to a file
            data_param_name = f'{sim_name}_' + description + '_param.txt'
            param_data = {'a': [param[0]], 'b': [param[1]], 'c': [param[2]], 'd': [param[3]]}
            write_dict_to_file(param_data, data_param_name)
        # Set the simulation data
        system.set_sim_data(data)

    system_cls = IODynamicSystem(sim_name='test', sim_data_dir=PATH + '/sim_data/', sim_data_file_ext='.txt',
                                 obs_data_file=path.abspath(
                                     path.join(__file__, "../..")) + '/data/linear_sim_data/linear_obs.dat',
                                 obs_names=['f'], ctrl_name='u', num_samples=10, param_min=[None, None, None, None],
                                 param_max=[None, None, None, None], param_names=['a', 'b', 'c', 'd'])

    system_cls.param_data = np.arange(1, system_cls.num_samples * 4 + 1, dtype=float).reshape(
        system_cls.num_samples, 4)

    bayesian_calibration_cls = BayesianCalibration(system_cls, IterativeBayesianFilter(), callback=run_sim)

    bayesian_calibration_cls.run_callback()

    # check if the file that contains the parameter data has the right name
    assert path.normpath(bayesian_calibration_cls.system.param_data_file) == path.normpath(
        path.join(GL_PATH, 'tests', 'unit', 'sim_data', 'iter0', 'test_Iter0_Samples.txt'))

    # check if the simulations data are stored with the right name
    bayesian_calibration_cls.system.get_sim_data_files()
    mag = floor(log(bayesian_calibration_cls.system.num_samples, 10)) + 1
    for i, f in enumerate(bayesian_calibration_cls.system.sim_data_files):
        description = 'Iter' + str(0) + '_Sample' + str(i).zfill(mag)
        assert path.normpath(f) == path.normpath(
            path.join(GL_PATH, 'tests', 'unit', 'sim_data', 'iter0', f'test_{description}_sim.txt'))

    # check if the parameter data are correctly stored
    param_data_backup = np.copy(bayesian_calibration_cls.system.param_data)
    bayesian_calibration_cls.system.load_param_data()
    np.testing.assert_array_equal(param_data_backup, bayesian_calibration_cls.system.param_data)

    # check if the simulation data are correctly stored
    sim_data_backup = np.copy(bayesian_calibration_cls.system.sim_data)
    bayesian_calibration_cls.system.load_sim_data()
    np.testing.assert_array_equal(sim_data_backup, bayesian_calibration_cls.system.sim_data)

    # remove the temporary dumpy data directory
    rmtree(GL_PATH + '/tests/unit/sim_data/')
