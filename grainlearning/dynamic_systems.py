"""
This module contains various classes of state-space systems that describe
the time evolution of the system's hidden state based on partial observations from the real world.
"""
from typing import Type, List, Callable
import os
from math import floor, log
from glob import glob
import numpy as np
from grainlearning.tools import get_keys_and_data, write_to_table


class DynamicSystem:
    """This is the dynamical system class.

    A dynamical system (also known as a state-space system) describes
    the time evolution of the system's (hidden) state and observation using the following equations:

    .. math::
        x_t & = f(x_{t−1}) + q_{t−1}

        y_t & = h(x_t) + r_t

    where
    :math:`x_t` is the hidden state,
    :math:`y_t` is the observation, both represented as random processes,
    :math:`f` is the state transition function,
    :math:`h` is the observation function,
    :math:`q_{t−1}` is the process noise, and :math:`r_t` is the observation noise.

    In the context of Bayesian parameter estimation, :math:`f` is the model
    that describe the physical process and :math:`h` is the model that describe
    the relationship between the hidden state and observation.
    In the simplest case, :math:`h` is an identity matrix which indicate the one-to-one relationship
    between :math:`x_t` and :math:`y_t`.

    Therefore, the :class:`.DynamicSystem` class is used to encapsulate the observation data
    and the simulation data, which require specifying the number of samples,
    the lower and upper bound of the parameters, and the callback function
    that runs the forward predictions.

    There are two ways of initializing the class.

    Method 1 - dictionary style

    .. highlight:: python
    .. code-block:: python

        system_cls = DynamicSystem.from_dict(
            {
                "param_names": ["a", "b"],
                "param_min": [0, 0],
                "param_max": [1, 10],
                "num_samples": 14,
                "obs_data": y_obs,
                "ctrl_data": y_ctrl,
                "callback": run_sim
            }
        )

    or

    Method 2 - class style

    .. highlight:: python
    .. code-block:: python

        system_cls = DynamicSystem(
            param_names = ["a", "b"],
            param_min = [0, 0],
            param_max = [1, 10],
            num_samples = 14,
            obs_data = y_obs,
            ctrl_data = y_ctrl,
            callback = run_sim
        )

    You can pass the simulation data to the class using the :meth:`.DynamicSystem.set_sim_data` method.

    .. highlight:: python
    .. code-block:: python

        system_cls.set_sim_data(x)

    The simulation data is a numpy array of shape (num_samples, num_obs, num_steps).

    :param obs_data: observation or reference data
    :param num_samples: Sample size of the ensemble of model evaluations
    :param param_min: List of parameter lower bounds
    :param param_max: List of parameter Upper bounds
    :param callback: Callback function, defaults to None
    :param ctrl_data: control data (e.g, time), defaults to None, optional
    :param obs_names: Column names of the observation data, defaults to None, optional
    :param ctrl_name: Column name of the control data, defaults to None, optional
    :param inv_obs_weight: Inverse of the observation weight, defaults to None, optional
    :param param_data: Parameter data, defaults to None, optional
    :param param_names: Parameter names, defaults to None, optional
    :param sim_data: Simulation data, defaults to None, optional
    :param sigma_max: Maximum uncertainty, defaults to 1.0e6, optional
    :param sigma_tol: Tolerance of the estimated uncertainty, defaults to 1.0e-3, optional
    :param sim_name: Name of the simulation, defaults to 'sim', optional
    """

    ##### Parameters #####

    #: Parameter data of shape (num_samples, num_params)
    param_data: np.ndarray

    #: Parameter data of previous iteration
    param_data_prev: np.ndarray

    #: Number of unknown parameters
    num_params: int

    #: Lower bound of the parameters
    param_min: List

    #: Upper bound of the parameters
    param_max: List

    #: Names of the parameters
    param_names: List[str]

    ##### Observations #####

    #: Observation (or reference) data of shape (num_obs, num_steps)
    obs_data: np.ndarray

    #: Observation keys
    obs_names: List[str]

    #: Inverse of the observation weight
    inv_obs_weight: List[float]

    #: Number of steps or sequence size in the dataset
    num_steps: int

    #: Number of observations in the dataset
    num_obs: int

    #: Control dataset (num_ctrl, num_steps)
    ctrl_data: np.ndarray

    #: Observation control (e.g., time)
    ctrl_name: str

    #: Number of control data (identical between simulation and observation)
    num_ctrl: int

    ##### Simulations #####

    #: Name of the simulation (e.g., sim)
    sim_name: str = 'sim'

    #: Simulation data of shape (num_samples, num_obs, num_steps)
    sim_data: np.ndarray

    #: Number of samples (usually specified by user)
    num_samples: int

    #: Callback function to run the forward predictions
    callback: Callable

    ##### Uncertainty #####

    #: Minimum value of the uncertainty
    sigma_min: float = 1.0e-6

    #: Maximum value of the uncertainty
    sigma_max: float = 1.0e6

    #: Sigma tolerance
    sigma_tol: float = 1.0e-3

    #: Calculated normalized sigma to weigh the covariance matrix
    _inv_normalized_sigma: np.array

    #: Estimated parameter as the first moment of the distribution (:math:`x_\mu = \sum_i w_i * x_i`)
    estimated_params: np.ndarray = None

    #: Parameter uncertainty as the second moment of the distribution (:math:`x_\sigma = \sum_i w_i \cdot (x_i - x_\mu)^2 / x_\mu`)
    estimated_params_cv: np.ndarray = None

    def __init__(
        self,
        obs_data: np.ndarray,
        num_samples: int,
        param_min: List[float],
        param_max: List[float],
        ctrl_data: np.ndarray = None,
        obs_names: List[str] = None,
        ctrl_name: str = None,
        inv_obs_weight: List[float] = None,
        sim_name: str = None,
        sim_data: np.ndarray = None,
        callback: Callable = None,
        param_data: np.ndarray = None,
        param_names: List[str] = None,
        sigma_max: float = 1.0e6,
        sigma_tol: float = 1.0e-3
    ):
        """Initialize the dynamic system class"""
        #### Observations ####
        self.obs_data = np.array(
            obs_data, ndmin=2
        )  # ensure data is of shape (num_obs,num_step).

        if ctrl_data is not None:
            self.ctrl_data = ctrl_data

        self.obs_names = obs_names

        self.ctrl_name = ctrl_name

        self.num_obs, self.num_steps = self.obs_data.shape

        self.num_ctrl, _ = self.obs_data.shape

        if inv_obs_weight is None:
            self.inv_obs_weight = list(np.ones(self.num_obs))
        else:
            self.inv_obs_weight = inv_obs_weight

        #### Simulations ####

        self.num_samples = num_samples

        self.sim_name = sim_name

        self.sim_data = sim_data

        self.callback = callback

        self.param_min = param_min

        self.param_max = param_max

        #### Parameters ####

        if param_min:
            self.num_params = len(param_min)

        self.param_data = param_data

        self.param_names = param_names

        #### Uncertainty ####

        self.sigma_max = sigma_max

        self.sigma_tol = sigma_tol

        self.compute_inv_normalized_sigma()

    @classmethod
    def from_dict(cls: Type["DynamicSystem"], obj: dict):
        """ Initialize the class using a dictionary style

        :param obj: Dictionary object
        :return: DynamicSystem: DynamicSystem object
        """

        assert "obs_data" in obj.keys(), "Error no obs_data key found in input"
        assert "num_samples" in obj.keys(), "Error no num_samples key found in input"
        assert "param_min" in obj.keys(), "Error no param_min key found in input"
        assert "param_max" in obj.keys(), "Error no param_max key found in input"
        assert "callback" in obj.keys(), "Error no callback key found in input"

        return cls(
            obs_data=obj["obs_data"],
            num_samples=obj["num_samples"],
            param_min=obj["param_min"],
            param_max=obj["param_max"],
            ctrl_data=obj.get("ctrl_data", None),
            obs_names=obj.get("obs_names", None),
            ctrl_name=obj.get("ctrl_name", None),
            inv_obs_weight=obj.get("inv_obs_weight", None),
            sim_name=obj.get("sim_name", None),
            sim_data=obj.get("sim_data", None),
            callback=obj.get("callback", None),
            param_data=obj.get("param_data", None),
            param_names=obj.get("param_names", None),
            sigma_tol=obj.get("sigma_tol", 0.001),
        )

    def run(self, **kwargs):
        """Run the callback function

        TODO design a better wrapper to avoid kwargs?
        :param kwargs: keyword arguments to pass to the callback function
        """

        if self.callback is None:
            raise ValueError("No callback function defined")

        self.callback(self, **kwargs)

    def set_sim_data(self, data: np.ndarray):
        """Set the simulation data

        :param data: simulation data of shape (num_samples, num_obs, num_steps)
        """
        self.sim_data = np.array(data)

    def compute_inv_normalized_sigma(self):
        """Compute the inverse of the matrix that apply different weights on the observables"""
        inv_obs_weight = np.diagflat(self.inv_obs_weight)
        self._inv_normalized_sigma = inv_obs_weight * np.linalg.det(inv_obs_weight) ** (
            -1.0 / inv_obs_weight.shape[0]
        )

    def get_inv_normalized_sigma(self):
        """Get the inverse of the matrix that apply different weights on the observables"""
        return self._inv_normalized_sigma

    def compute_estimated_params(self, posteriors: np.array):
        """Compute the estimated means and uncertainties for the parameters.

        This function is vectorized for all time steps

        :param posteriors: Posterior distribution of shape (num_steps, num_samples)
        """
        self.estimated_params = np.zeros((self.num_steps, self.num_params))
        self.estimated_params_cv = np.zeros((self.num_steps, self.num_params))

        for stp_id in range(self.num_steps):
            self.estimated_params[stp_id, :] = posteriors[stp_id, :] @ self.param_data

            self.estimated_params_cv[stp_id, :] = (
                posteriors[stp_id, :] @ (self.estimated_params[stp_id, :] - self.param_data) ** 2
            )

            self.estimated_params_cv[stp_id, :] = np.sqrt(
                self.estimated_params_cv[stp_id, :]) / self.estimated_params[stp_id, :]

    @classmethod
    def load_param_data(cls: Type["DynamicSystem"], curr_iter: int):
        """Virtual function to load param data from disk"""

    @classmethod
    def get_sim_data_files(cls: Type["DynamicSystem"], curr_iter: int):
        """Virtual function to get simulation data files from disk"""

    @classmethod
    def load_sim_data(cls: Type["DynamicSystem"]):
        """Virtual function to load simulation data"""

    @classmethod
    def write_to_table(cls: Type["DynamicSystem"], curr_iter: int):
        """Virtual function to write parameters into a text file"""


class IODynamicSystem(DynamicSystem):
    """
    This is the I/O dynamic system class derived from the dynamic system class.
    Extra functionalities are added to handle I/O operations.

    There are two ways of initializing the class.

    Method 1 - dictionary style

    .. highlight:: python
    .. code-block:: python

        system_cls = IODynamicSystem.from_dict(
            {
                "system_type": IODynamicSystem,
                "param_min": [0, 0],
                "param_max": [1, 10],
                "param_names": ['a', 'b'],
                "num_samples": 14,
                "obs_data_file": 'obs_data.txt',
                "obs_names": ['y_obs'],
                "ctrl_name": 'y_ctrl
                "sim_name": 'linear',
                "sim_data_file_ext": '.txt',
                "callback": run_sim
            }
        )

    or

    Method 2 - class style

    .. highlight:: python
    .. code-block:: python

        system_cls = IODynamicSystem(
                param_min = [0, 0],
                param_max = [1, 10],
                param_names = ['a', 'b'],
                num_samples = 14,
                obs_data_file = 'obs_data.txt',
                obs_names = ['y_obs'],
                ctrl_name = 'y_ctrl',
                sim_name = 'linear',
                sim_data_file_ext = '.txt',
                callback = run_sim
        )

    :param param_min: List of parameter lower bounds
    :param param_max: List of parameter Upper bounds
    :param param_names: Parameter names, defaults to None
    :param num_samples: Sample size of the ensemble of model evaluations
    :param obs_data_file: Observation data file, defaults to None
    :param obs_names: Column names of the observation data, defaults to None
    :param ctrl_name: Column name of the control data, defaults to None
    :param sim_name: Name of the simulation, defaults to 'sim'
    :param sim_data_dir: Simulation data directory, defaults to './sim_data'
    :param sim_data_file_ext: Simulation data file extension, defaults to '.npy'
    :param callback: Callback function, defaults to None
    :param param_data_file: Parameter data file, defaults to None, optional
    :param obs_data: observation or reference data, optional
    :param ctrl_data: Control data (e.g, time), defaults to None, optional
    :param inv_obs_weight: Inverse of the observation weight, defaults to None, optional
    :param param_data: Parameter data, defaults to None, optional
    :param sim_data: Simulation data, defaults to None, optional
    """

    ##### Parameters #####

    #: Name of the parameter data file
    param_data_file: str

    ##### Observations #####

    #: Name of the observation data file
    obs_data_file: str

    ##### Simulations #####

    #: Simulation data files (num_samples)
    sim_data_files: List[str]

    #: Name of the directory where simulation data is stored
    sim_data_dir: str = './sim_data'

    #: Extension of simulation data files
    sim_data_file_ext: str = '.npy'

    def __init__(
        self,
        sim_name: str,
        sim_data_dir: str,
        sim_data_file_ext: str,
        obs_data_file: str,
        obs_names: List[str],
        ctrl_name: str,
        param_data_file: str,
        obs_data: np.ndarray,
        num_samples: int,
        param_min: List[float],
        param_max: List[float],
        ctrl_data: np.ndarray = None,
        inv_obs_weight: List[float] = None,
        sim_data: np.ndarray = None,
        callback: Callable = None,
        param_data: np.ndarray = None,
        param_names: List[str] = None,
    ):
        """Initialize the IO dynamic system class"""

        #### Calling base constructor ####

        super().__init__(
            obs_data,
            num_samples,
            param_min,
            param_max,
            ctrl_data,
            obs_names,
            ctrl_name,
            inv_obs_weight,
            sim_name,
            sim_data,
            callback,
            param_data,
            param_names
        )

        ##### Parameters #####

        self.num_params = len(param_names)

        self.param_data_file = param_data_file

        #### Simulations ####

        self.sim_name = sim_name

        self.sim_data_dir = sim_data_dir

        self.sim_data_file_ext = sim_data_file_ext

        #### Observations ####

        self.obs_data_file = os.path.join(sim_data_dir, obs_data_file)

        self.ctrl_name = ctrl_name

        self.obs_names = obs_names

        self.get_obs_data()

        if inv_obs_weight is None:
            self.inv_obs_weight = list(np.ones(self.num_obs))
        else:
            self.inv_obs_weight = inv_obs_weight

        self.compute_inv_normalized_sigma()

    @classmethod
    def from_dict(cls: Type["IODynamicSystem"], obj: dict):
        """ Initialize the class using a dictionary style

        :param obj: Dictionary object
        :return IODynamicSystem: IODynamicSystem object
        """
        assert "param_names" in obj.keys(), "Error no param_names key found in input"
        assert "obs_data_file" in obj.keys(), "Error no obs_data_file key found in input"
        assert "obs_names" in obj.keys(), "Error no obs_names key found in input"
        assert "ctrl_name" in obj.keys(), "Error no ctrl_name key found in input"
        assert "sim_name" in obj.keys(), "Error no sim_name key found in input"
        assert "sim_data_dir" in obj.keys(), "Error no sim_data_dir key found in input"
        if "param_data_file" not in obj.keys():
            obj["param_data_file"] = None

        return cls(
            sim_name=obj["sim_name"],
            sim_data_dir=obj["sim_data_dir"],
            sim_data_file_ext=obj.get("sim_data_file_ext", '.npy'),
            obs_data_file=obj["obs_data_file"],
            obs_names=obj["obs_names"],
            ctrl_name=obj["ctrl_name"],
            param_data_file=obj["param_data_file"],
            obs_data=obj.get("obs_data", None),
            num_samples=obj.get("num_samples", None),
            param_min=obj.get("param_min", None),
            param_max=obj.get("param_max", None),
            ctrl_data=obj.get("ctrl_data", None),
            inv_obs_weight=obj.get("inv_obs_weight", None),
            sim_data=obj.get("sim_data", None),
            callback=obj.get("callback", None),
            param_data=obj.get("param_data", None),
            param_names=obj.get("param_names", None),
        )

    def get_obs_data(self):
        """Get the observation data from the observation data file.

        Separate the control data from the observation data if the name of control variable is given.
        Otherwise, the observation data is the entire data in the observation data file.
        """
        # if self.ctrl_name is given, separate the observation data into control and observation data
        if self.ctrl_name:
            keys_and_data = get_keys_and_data(self.obs_data_file)
            # separate the control data sequence from the observation data
            self.ctrl_data = keys_and_data.pop(self.ctrl_name)
            self.num_steps = len(self.ctrl_data)
            # remove the data not used by Bayesian filtering
            self.num_obs = len(self.obs_names)
            for key in keys_and_data:
                if key not in self.obs_names:
                    keys_and_data.pop(key)
            # assign the obs_data array
            self.obs_data = np.zeros([self.num_obs, self.num_steps])
            for i, key in enumerate(self.obs_names):
                self.obs_data[i, :] = keys_and_data[key]
        else:
            self.obs_data = np.genfromtxt(self.obs_data_file)
            # if only one observation data vector exists, reshape it with (1, num_steps)
            if len(self.obs_data) == 1:
                self.obs_data = self.obs_data.reshape([1, self.obs_data.shape[0]])

    def get_sim_data_files(self, curr_iter: int = 0):
        """Get the simulation data files from the simulation data directory.

        :param curr_iter: Current iteration number, default to 0.
        """
        mag = floor(log(self.num_samples, 10)) + 1
        self.sim_data_files = []

        for i in range(self.num_samples):
            if self.sim_data_file_ext != '.npy':
                sim_data_file_ext = '_sim' + self.sim_data_file_ext
            else:
                sim_data_file_ext = self.sim_data_file_ext
            file_name = self.sim_data_dir + f'/iter{curr_iter}/{self.sim_name}*Iter{curr_iter}*'\
                        + str(i).zfill(mag) + '*' + sim_data_file_ext
            files = glob(file_name)

            if not files:
                raise RuntimeError("No data files with name " + file_name + ' found')
            if len(files) > 1:
                raise RuntimeError("Found more than one files with the name " + file_name)
            self.sim_data_files.append(files[0])

    def load_sim_data(self):
        """Load the simulation data from the simulation data files.

        The function does the following:
        1. Load simulation data into an IO dynamic system object
        2. Check if parameter values read from the table matches those used to creat the simulation data
        """
        self.sim_data = np.zeros([self.num_samples, self.num_obs, self.num_steps])
        for i, sim_data_file in enumerate(self.sim_data_files):
            if self.sim_data_file_ext != '.npy':
                data = get_keys_and_data(sim_data_file)
                param_data = np.genfromtxt(sim_data_file.split('_sim')[0] + f'_param{self.sim_data_file_ext}')
                for j, key in enumerate(self.param_names):
                    data[key] = param_data[j]
            else:
                data = np.load(sim_data_file, allow_pickle=True).item()

            for j, key in enumerate(self.obs_names):
                self.sim_data[i, j, :] = data[key]

            params = np.array([data[key] for key in self.param_names])
            if not (np.abs((params - self.param_data[i, :])
                           / self.param_data[i, :] < 1e-5).all()):
                raise RuntimeError(
                    "Parameters [" + ", ".join(
                        [f"{v}" for v in self.param_data[i, :]])
                    + '] vs [' +
                    ", ".join(f"{v}" for v in params) +
                    f"] from the simulation data file {sim_data_file} and the parameter table do not match")

    def load_param_data(self, curr_iter: int = 0):
        """
        Load parameter data from a table written in a text file.

        :param curr_iter: Current iteration number, default to 0.
        """
        if os.path.exists(self.param_data_file):
            # we assume parameter data are always in the last columns.
            self.param_data = np.genfromtxt(self.param_data_file, comments='!')[:, -self.num_params:]
            self.num_samples = self.param_data.shape[0]
        else:
            # if param_data_file does not exit, get parameter daa from simulation data files
            files = glob(self.sim_data_dir + f'/iter{curr_iter}/{self.sim_name}*{self.sim_data_file_ext}')
            self.num_samples = len(files)
            self.sim_data_files = sorted(files)
            self.param_data = np.zeros([self.num_samples, self.num_params])
            for i, sim_data_file in enumerate(self.sim_data_files):
                data = np.load(sim_data_file, allow_pickle=True).item()
                params = [data[key] for key in self.param_names]
                self.param_data[i, :] = params

    def run(self, **kwargs):
        """Run the callback function

        TODO design a better wrapper to avoid kwargs?
        :param kwargs: keyword arguments to pass to the callback function
        """

        if self.callback is None:
            raise ValueError("No callback function defined")

        # create a directory to store simulation data
        curr_iter = kwargs['curr_iter']
        sim_data_sub_dir = f'{self.sim_data_dir}/iter{curr_iter}'
        if not os.path.exists(sim_data_sub_dir):
            os.makedirs(sim_data_sub_dir)
        else:
            input(f'Removing existing simulation data in {sim_data_sub_dir}?\n')
            files = glob(sim_data_sub_dir + '/*')
            for file in files:
                os.remove(file)

        # write the parameter data into a text file
        self.write_to_table(curr_iter)

        # run the callback function
        self.callback(self, **kwargs)

        # move simulation data files into the directory defined per iteration
        files = glob(f'{self.sim_name}_Iter{curr_iter}*{self.sim_data_file_ext}')
        for file in files:
            os.replace(f'./{file}', f'./{sim_data_sub_dir}/{file}')

    def write_to_table(self, curr_iter: int):
        """Write the parameter data into a text file.

        :param curr_iter: Current iteration number, default to 0.
        :return param_data_file: The name of the parameter data file
        """
        self.param_data_file = write_to_table(
            f'{self.sim_data_dir}/iter{curr_iter}/{self.sim_name}',
            self.param_data,
            self.param_names,
            curr_iter)
