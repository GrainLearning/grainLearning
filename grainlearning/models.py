from typing import Type, List, Callable, Tuple
import numpy as np
from .tools import get_keys_and_data, write_to_table


class Model:
    """
    This is the probabalistic model class. It contains information on the observation (or reference) data, simulation data, parameters and reference. It is also used to run a callback for the simulations.

    There are two ways of initializing the class.

    Method 1 - dictionary style

    .. highlight:: python
    .. code-block:: python

        model_cls = Model.from_dict(
            {

                "param_mins": [0, 0],
                "param_maxs": [1, 10],
                "num_samples": 14,
                "obs_data": y_obs,
                "ctrl_data": x_ctrl,
                "callback": run_sim
            }
        )

    or

    Method 2 - class style

    .. highlight:: python
    .. code-block:: python

        model_cls = Model(
                param_mins = [0, 0],
                param_maxs = [1, 10],
                num_samples = 14,
                obs_data = y_obs,
                ctrl_data = x_ctrl,
                callback = run_sim
        )

    y_obs is the observation data, x_ctrl is the control data. The callback function inputs the model as an argument, where one can modify the model.sim_data.

    :param obs_data: Observation or reference data
    :param num_samples: Sample size
    :param param_mins: List of parameter lower bounds
    :param param_maxs: List of parameter Upper bounds
    :param ctrl_data: Optional control data (e.g, time), defaults to None
    :param obs_names: Column names of the observation data, defaults to None
    :param ctrl_name: Coloumn name of the control data, defaults to None
    :param inv_obs_weight: Inverse of the observation weight, defaults to None
    :param param_data: Parameter data, defaults to None
    :param param_names: Parameter names, defaults to None
    :param sim_data: Simulation data, defaults to None
    :param callback: Callback function, defaults to None
    :param sigma_max: Uncertainty, defaults to 1.0e6
    """

    ##### Parameters #####

    #: Parameter data of shape (num_samples, num_params)
    param_data: np.ndarray

    #: Parameter data of previous iteration
    param_data_prev: np.ndarray

    #: Number of parameters
    num_params: int

    #: Minimum values of the parameters
    param_mins: List

    #: Maximum number of parameters
    param_maxs: List

    #: Names of the parameters.
    param_names: List[str]

    ##### Observations #####

    #: Observation (or reference) data of shape (num_obs,num_steps)
    obs_data: np.ndarray

    #: Observation keys
    obs_names: List[str]

    #: Inverse observation weight
    inv_obs_weight: List[float]

    #: Number of steps or sequence size in the dataset
    num_steps: int

    #: Number of observations in the dataset
    num_obs: int

    #: Control data (num_control,num_steps)
    ctrl_data = np.ndarray

    #: Observation control (e.g., time)
    ctrl_name: str

    #: Number of control data
    num_ctrl: int

    ##### Simulations #####

    #: Name of the simulation (e.g., sim)
    sim_name: str = 'sim'

    #: Simulation data of shape (num_samples,num_obs,num_steps)
    sim_data: np.ndarray

    #: Number of samples (usually specified by user)
    num_samples: int

    #: Callback function. The input arugment is the model where model.sim_data is modified
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

    def __init__(
        self,
        obs_data: np.ndarray,
        num_samples: int,
        param_mins: List[float],
        param_maxs: List[float],
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
        """Initialize the Model class"""
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

        self.param_mins = param_mins

        self.param_maxs = param_maxs

        #### Parameters ####

        if param_mins: self.num_params = len(param_mins)

        self.param_data = param_data

        self.param_names = param_names

        #### Uncertainty ####

        self.sigma_max = sigma_max

        self.sigma_tol = sigma_tol

        self.get_inv_normalized_sigma()

    @classmethod
    def from_dict(cls: Type["Model"], obj: dict) -> Type["Model"]:
        """ Initialize the class using a dictionary style"""

        # TODO do proper error checking on the input
        assert "obs_data" in obj.keys(), "Error no obs_data key found in input"
        assert "num_samples" in obj.keys(), "Error no num_samples key found in input"
        assert "param_mins" in obj.keys(), "Error no param_mins key found in input"
        assert "param_maxs" in obj.keys(), "Error no param_maxs key found in input"

        return cls(
            obs_data=obj["obs_data"],
            num_samples=obj["num_samples"],
            param_mins=obj["param_mins"],
            param_maxs=obj["param_maxs"],
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
        """This function runs the callback function"""

        if self.callback is None:
            raise ValueError("No callback function defined")

        self.callback(self, **kwargs)

    def get_inv_normalized_sigma(self):
        inv_obs_mat = np.diagflat(self.inv_obs_weight)
        self._inv_normalized_sigma = inv_obs_mat * np.linalg.det(inv_obs_mat) ** (
            -1.0 / inv_obs_mat.shape[0]
        )


class IOModel(Model):
    """
    This is the IOModel class to compute the posterior distribution from an existing dataset,
    and generate new parameter values for additional simulation runs.

    There are two ways of initializing the class.

    Method 1 - dictionary style

    .. highlight:: python
    .. code-block:: python

        model_cls = Model.from_dict(
            {
                "param_mins": [0, 0],
                "param_maxs": [1, 10],
                "num_samples": 14,
                "obs_data": y_obs,
                "ctrl_data": x_ctrl,
                "callback": run_sim
            }
        )

    or

    Method 2 - class style

    .. highlight:: python
    .. code-block:: python

        model_cls = Model(
                param_mins = [0, 0],
                param_maxs = [1, 10],
                num_samples = 14,
                obs_data = y_obs,
                ctrl_data = x_ctrl,
                callback = run_sim
        )

    y_obs is the observation data, x_ctrl is the control data. The callback function inputs the model as an argument, where one can modify the model.sim_data.

    :param obs_data: Observation or reference data
    :param num_samples: Sample size
    :param param_mins: List of parameter lower bounds
    :param param_maxs: List of parameter Upper bounds
    :param ctrl_data: Optional control data (e.g, time), defaults to None
    :param obs_names: Column names of the observation data, defaults to None
    :param ctrl_name: Coloumn name of the control data, defaults to None
    :param inv_obs_weight: Inverse of the observation weight, defaults to None
    :param param_data: Parameter data, defaults to None
    :param param_names: Parameter names, defaults to None
    :param sim_data_files: Simulation data files, defaults to None
    :param sim_data: Simulation data, defaults to None
    :param callback: Callback function, defaults to None
    :param sigma_max: Uncertainty, defaults to 1.0e6
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
        param_mins: List[float],
        param_maxs: List[float],
        ctrl_data: np.ndarray = None,
        inv_obs_weight: List[float] = None,
        sim_data: np.ndarray = None,
        callback: Callable = None,
        param_data: np.ndarray = None,
        param_names: List[str] = None,
    ):
        """Initialize the IOModel class"""

        #### Calling base constructor ####

        super().__init__(
            obs_data,
            num_samples,
            param_mins,
            param_maxs,
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

        self.obs_data_file = sim_data_dir + '/' + obs_data_file

        self.ctrl_name = ctrl_name

        self.obs_names = obs_names

        self.get_obs_data()

        if inv_obs_weight is None:
            self.inv_obs_weight = list(np.ones(self.num_obs))
        else:
            self.inv_obs_weight = inv_obs_weight

        self.get_inv_normalized_sigma()

    @classmethod
    def from_dict(cls: Type["IOModel"], obj: dict) -> Type["IOModel"]:
        """ Initialize the class using a dictionary style"""

        # TODO do proper error checking on the input
        assert "sim_name" in obj.keys(), "Error no sim_name key found in input"
        assert "sim_data_dir" in obj.keys(), "Error no sim_data_dir key found in input"
        assert "obs_data_file" in obj.keys(), "Error no obs_data_file key found in input"
        assert "obs_names" in obj.keys(), "Error no obs_names key found in input"
        assert "ctrl_name" in obj.keys(), "Error no ctrl_name key found in input"
        if "param_data_file" not in obj.keys(): obj["param_data_file"] = None

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
            param_mins=obj.get("param_mins", None),
            param_maxs=obj.get("param_maxs", None),
            ctrl_data=obj.get("ctrl_data", None),
            inv_obs_weight=obj.get("inv_obs_weight", None),
            sim_data=obj.get("sim_data", None),
            callback=obj.get("callback", None),
            param_data=obj.get("param_data", None),
            param_names=obj.get("param_names", None),
        )

    def get_obs_data(self):
        # if self.ctrl_name specifies the control variable during the observation
        if self.ctrl_name:
            keys_and_data = get_keys_and_data(self.obs_data_file)
            # separate the control data sequence from the observation data
            self.ctrl_data = keys_and_data.pop(self.ctrl_name)
            self.num_steps = len(self.ctrl_data)
            # remove data not used by the calibration
            self.num_obs = len(self.obs_names)
            for key in keys_and_data.keys():
                if key not in self.obs_names: keys_and_data.pop(key)
            # assign the obs_data array
            self.obs_data = np.zeros([self.num_obs, self.num_steps])
            for i, key in enumerate(self.obs_names):
                self.obs_data[i, :] = keys_and_data[key]
        else:
            self.obs_data = np.genfromtxt(self.obs_data_file)
            # if only one observation data vector exists, reshape it with (1, num_steps)
            if len(data) == 1:
                self.obs_data = self.obs_data.reshape([1, self.obs_data.shape[0]])

    def get_sim_data_files(self, curr_iter: int = 0):
        from math import floor, log
        from glob import glob

        magn = floor(log(self.num_samples, 10)) + 1
        self.sim_data_files = []

        for i in range(self.num_samples):
            if self.sim_data_file_ext != '.npy':
                sim_data_file_ext = '_sim*' + self.sim_data_file_ext
            else:
                sim_data_file_ext = self.sim_data_file_ext
            file_name = self.sim_data_dir + f'/iter{curr_iter}/{self.sim_name}*' \
                        + str(i).zfill(magn) + '*' + sim_data_file_ext
            files = glob(file_name)

            if not files:
                raise RuntimeError("No data files with name " + file_name + ' found')
            elif len(files) > 1:
                raise RuntimeError("Found more than one files with the name " + file_name)
            self.sim_data_files.append(files[0])

    def load_sim_data(self):
        """
        1. Read simulation data into self.model.sim_data and remove the observation data sequence
        2. Check if parameter values read from the table matches those used to creat the simulation data
        """
        self.sim_data = np.zeros([self.num_samples, self.num_obs, self.num_steps])
        for i, f in enumerate(self.sim_data_files):
            if self.sim_data_file_ext != '.npy':
                data = get_keys_and_data(f)
                param_data = np.genfromtxt(f.split('_sim')[0] + f'_param{self.sim_data_file_ext}')
                for j, key in enumerate(self.param_names):
                    data[key] = param_data[j]
            else:
                data = np.load(f, allow_pickle=True).item()

            for j, key in enumerate(self.obs_names):
                self.sim_data[i, j, :] = data[key]

            params = [data[key] for key in self.param_names]
            if not (np.abs((params - self.param_data[i, :])
                           / self.param_data[i, :] < 1e-5).all()):
                raise RuntimeError(
                    "Parameters [" + ", ".join(
                        ["%s" % v for v in self.param_data[i, :]])
                    + '] vs [' + \
                    ", ".join("%s" % v for v in params) + \
                    f"] from the simulation data file {f} and the parameter table do not match")

    def load_param_data(self, curr_iter: int = 0):
        """
        Load parameter data from a table written in a txt file
        """
        import os
        from glob import glob

        if os.path.exists(self.param_data_file):
            # we assumes parameter data in the last columns.
            self.param_data = np.genfromtxt(self.param_data_file, comments='!')[:, -self.num_params:]
            self.num_samples = self.param_data.shape[0]
        else:
            # get all simulation data files
            files = glob(self.sim_data_dir + f'/iter{curr_iter}/{self.sim_name}*{self.sim_data_file_ext}')
            self.num_samples = len(files)
            self.sim_data_files = sorted(files)
            self.param_data = np.zeros([self.num_samples, self.num_params])
            for i, f in enumerate(self.sim_data_files):
                data = np.load(f, allow_pickle=True).item()
                params = [data[key] for key in self.param_names]
                self.param_data[i, :] = params

    def run(self, **kwargs):
        """This function runs the callback function"""

        if self.callback is None:
            raise ValueError("No callback function defined")

        # create a directory to store simulation data
        import os
        from glob import glob
        curr_iter = kwargs['curr_iter']
        sim_data_sub_dir = f'{self.sim_data_dir}/iter{curr_iter}'
        if not os.path.exists(sim_data_sub_dir):
            os.makedirs(sim_data_sub_dir)
        else:
            input(f'Removing existing simulation data in {sim_data_sub_dir}?\n')
            files = glob(sim_data_sub_dir + '/*')
            for f in files: os.remove(f)

        # write the parameter table to a text file
        self.write_to_table(curr_iter)

        # run the callback function
        self.callback(self, **kwargs)

        # move simulation data files into the directory per iteration
        files = glob(f'{self.sim_name}_Iter{curr_iter}*{self.sim_data_file_ext}')
        for f in files: os.replace(f'./{f}', f'./{sim_data_sub_dir}/{f}')

    def write_to_table(self, curr_iter: int):
        self.param_data_file = write_to_table(
            f'{self.sim_data_dir}/iter{curr_iter}/{self.sim_name}', self.param_data, self.param_names, curr_iter)
