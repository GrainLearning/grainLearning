"""Test the dynamic system module."""
from os import path
import numpy as np
from grainlearning import DynamicSystem, IODynamicSystem


class TestDynamicSystem:
    """The class that contains all tests for a dynamic system."""

    def test_init(self):
        """Test if the class is initialized correctly"""
        system_cls = DynamicSystem(
            param_min=[1, 2],
            param_max=[3, 4],
            obs_data=[[12, 3, 4, 4], [12, 4, 5, 4]],
            ctrl_data=[1, 2, 3, 4],
            num_samples=10,
        )

        assert isinstance(system_cls, DynamicSystem)

        config = {
            "param_min": [1, 2],
            "param_max": [3, 4],
            "obs_data": [[12, 3, 4, 4], [12, 4, 5, 4]],
            "ctrl_data": [1, 2, 3, 4],
            "num_samples": 10,
        }
        system_dct = DynamicSystem.from_dict(config)

        np.testing.assert_equal(system_cls.__dict__, system_dct.__dict__)
        np.testing.assert_array_almost_equal(
            system_cls.get_inv_normalized_sigma(),
            [
                [
                    1.41421356,
                    0.0,
                ],
                [0.0, 0.70710678],
            ],
            0.00001,
        )

    def test_compute_inv_normalized_sigma(self):
        """Test if the inverse normalized sigma is computed correctly"""
        system_cls = DynamicSystem(
            param_min=[1, 2, 3],
            param_max=[4, 5, 6],
            obs_data=np.arange(12).reshape(3, 4),
            ctrl_data=np.arange(4),
            num_samples=10,
            inv_obs_weight=[0.5, 0.25, 0.25],
        )

        np.testing.assert_array_almost_equal(
            system_cls.get_inv_normalized_sigma(),
            [[1.58740105, 0., 0.],
             [0., 0.79370053, 0.],
             [0., 0., 0.79370053]],
            0.00001,
        )

        # set the inv_normalized_sigma to None to test the function
        system_cls.reset_inv_normalized_sigma()

        system_cls.inv_obs_weight = [0.4, 0.3, 0.2]

        # compute again the inverse normalized sigma with a new weighting
        system_cls.compute_inv_normalized_sigma()

        np.testing.assert_array_almost_equal(
            system_cls.get_inv_normalized_sigma(),
            [[1.38672255, 0., 0.],
             [0., 1.04004191, 0.],
             [0., 0., 0.69336127]],
            0.00001,
        )

    def test_compute_estimated_params(self):
        """Test if the estimated parameters are computed from the posterior distribution correctly"""
        system_cls = DynamicSystem(
            param_min=[1, 2, 3],
            param_max=[4, 5, 6],
            obs_data=np.arange(12).reshape(3, 4),
            num_samples=10,
        )

        # Generate dummy samples
        system_cls.param_data = np.array(
            [
                [1., 2., 3.],
                [2.5, 3., 3.6],
                [1.75, 4., 4.2],
                [3.25, 2.33333333, 4.8],
                [1.375, 3.33333333, 5.4],
                [2.875, 4.33333333, 3.12],
                [2.125, 2.66666667, 3.72],
                [3.625, 3.66666667, 4.32],
                [1.1875, 4.66666667, 4.92],
                [2.6875, 2.11111111, 5.52]
            ]
        )

        # Generate a dummy posterior distribution (created with np.random.rand)
        posteriors = np.array(
            [
                [0.15500157, 0.09624244, 0.4280316, 0.7363387, 0.38838718,
                 0.17697888, 0.28657835, 0.17211441, 0.94996649, 0.75067305],
                [0.50780152, 0.42213463, 0.15667768, 0.40388211, 0.71117121,
                 0.74100691, 0.21623105, 0.1059679, 0.51379481, 0.34010844],
                [0.65359258, 0.72757396, 0.94833572, 0.16832953, 0.63599938,
                 0.98435012, 0.37146998, 0.87515041, 0.94847439, 0.03849703],
                [0.11020751, 0.16237893, 0.83208346, 0.51149285, 0.24458735,
                 0.4775532, 0.77801339, 0.84111314, 0.33597833, 0.56797428]
            ]
        )

        # Normalize the posterior distribution at every time step
        posteriors = posteriors / posteriors.sum(axis=1).reshape(4, 1)

        estimated_params = np.array(
            [
                [2.16385183, 3.26152642, 4.69052907],
                [2.09430982, 3.2809723, 4.23899345],
                [2.1372394, 3.58458817, 4.0804544],
                [2.46260083, 3.2994312, 4.31809075]
            ]
        )

        estimated_params_cv = np.array(
            [
                [0.39162964, 0.31337427, 0.15403638],
                [0.39689874, 0.28818393, 0.22721853],
                [0.41518435, 0.23276729, 0.19188208],
                [0.32668404, 0.25540592, 0.16929316]
            ]
        )

        system_cls.compute_estimated_params(posteriors)

        np.testing.assert_array_almost_equal(
            system_cls.estimated_params,
            estimated_params,
            0.00001,
        )

        np.testing.assert_array_almost_equal(
            system_cls.estimated_params_cv,
            estimated_params_cv,
            0.00001,
        )


class TestIODynamicSystem:
    """The class that contains all tests for a IO dynamic system."""

    def test_init(self):
        """Test if the class is initialized correctly"""
        system_cls = IODynamicSystem(sim_name='linear',
                                     sim_data_dir=path.abspath(path.join(__file__, "../..")) + '/data/linear_sim_data/',
                                     sim_data_file_ext='.txt', obs_data_file=path.abspath(
                path.join(__file__, "../..")) + '/data/linear_sim_data/linear_obs.dat', obs_names=['f'], ctrl_name='u',
                                     num_samples=20, param_min=[0.001, 0.001], param_max=[1, 10],
                                     param_names=['a', 'b'])

        config = {
            "system_type": IODynamicSystem,
            "param_min": [0.001, 0.001],
            "param_max": [1, 10],
            "param_names": ['a', 'b'],
            "num_samples": 20,
            "obs_data_file": path.abspath(path.join(__file__, "../..")) + '/data/linear_sim_data/linear_obs.dat',
            "obs_names": ['f'],
            "ctrl_name": 'u',
            "sim_name": 'linear',
            "sim_data_dir": path.abspath(path.join(__file__, "../..")) + '/data/linear_sim_data/',
            "sim_data_file_ext": '.txt',
        }
        system_dct = IODynamicSystem.from_dict(config)

        np.testing.assert_equal(system_cls.__dict__, system_dct.__dict__)

    def test_get_obs_data(self):
        """Test if the observation data is read correctly"""
        config = {
            "system_type": IODynamicSystem,
            "param_min": [0.001, 0.001],
            "param_max": [1, 10],
            "obs_data_file": path.abspath(path.join(__file__, "../../data/linear_sim_data/linear_obs.dat")),
            "obs_names": ['f'],
            "ctrl_name": 'u',
            "sim_name": 'linear',
            "sim_data_dir": path.abspath(path.join(__file__, "../../data/linear_sim_data")),
            "param_names": ['a', 'b']
        }
        system_dct = IODynamicSystem.from_dict(config)

        obs_data = np.array(
            [
                [5., 5.2, 5.4, 5.6, 5.8, 6., 6.2, 6.4, 6.6, 6.8, 7.,
                 7.2, 7.4, 7.6, 7.8, 8., 8.2, 8.4, 8.6, 8.8, 9., 9.2,
                 9.4, 9.6, 9.8, 10., 10.2, 10.4, 10.6, 10.8, 11., 11.2, 11.4,
                 11.6, 11.8, 12., 12.2, 12.4, 12.6, 12.8, 13., 13.2, 13.4, 13.6,
                 13.8, 14., 14.2, 14.4, 14.6, 14.8, 15., 15.2, 15.4, 15.6, 15.8,
                 16., 16.2, 16.4, 16.6, 16.8, 17., 17.2, 17.4, 17.6, 17.8, 18.,
                 18.2, 18.4, 18.6, 18.8, 19., 19.2, 19.4, 19.6, 19.8, 20., 20.2,
                 20.4, 20.6, 20.8, 21., 21.2, 21.4, 21.6, 21.8, 22., 22.2, 22.4,
                 22.6, 22.8, 23., 23.2, 23.4, 23.6, 23.8, 24., 24.2, 24.4, 24.6,
                 24.8]
            ]
        )

        np.testing.assert_array_almost_equal(
            system_dct.obs_data,
            obs_data,
            0.00001,
        )


def test_dynamic_system():
    """Test the dynamic system class"""
    # run all tests
    test_cls = TestDynamicSystem()
    test_cls.test_init()
    test_cls.test_compute_inv_normalized_sigma()
    test_cls.test_compute_estimated_params()


def test_io_dynamic_system():
    """Test the IO dynamic system class"""
    # run all tests
    test_cls = TestIODynamicSystem()
    test_cls.test_init()
    test_cls.test_get_obs_data()
