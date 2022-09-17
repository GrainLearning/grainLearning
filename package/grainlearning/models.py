from typing import Type, List, Callable, Tuple
import numpy as np


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
    :param ctrl_names: Coloumn names of the control data, defaults to None
    :param inv_obs_weight: Inverse of the observation weight, defaults to None
    :param param_data: Parameter data, defaults to None
    :param param_names: Parameternames, defaults to None
    :param sim_data: Simulation data, defaults to None
    :param callback: Callback function, defaults to None
    :param sigma_max: Uncertainty, defaults to 1.0e6
    """

    ##### Parameters #####

    #: Parameter data of shape (num_samples, num_params)
    param_data: np.ndarray

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
    ctrl_names: List[str]

    #: Number of control data
    num_ctrl: int

    ##### Simulations #####

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
            ctrl_names: List[str] = None,
            inv_obs_weight: List[float] = None,
            param_data: np.ndarray = None,
            param_names: List[str] = None,
            sim_data: np.ndarray = None,
            callback: Callable = None,
            sigma_max: float = 1.0e6,
    ):
        """Initialize the Model class"""
        #### Observations ####
        self.obs_data = np.array(
            obs_data, ndmin=2
        )  # ensure data is of shape (num_obs,num_step).

        if ctrl_data is not None:
            self.ctrl_data = np.array(
                ctrl_data, ndmin=2
            )  # ensure data is of shape (num_obs,num_step).

        self.obs_names = obs_names

        self.ctrl_names = ctrl_names

        self.num_obs, self.num_steps = self.obs_data.shape

        self.num_ctrl, _ = self.obs_data.shape

        if inv_obs_weight is None:
            self.inv_obs_weight = list(np.ones(self.num_obs))
        else:
            self.inv_obs_weight = inv_obs_weight

        inv_obs_mat = np.diagflat(self.inv_obs_weight)

        #### Simulations ####

        self.num_samples = num_samples

        self.sim_data = sim_data

        self.callback = callback

        self.param_mins = param_mins

        self.param_maxs = param_maxs

        #### Parameters ####

        self.num_params = len(param_mins)

        if param_data == None:
            self.generate_params_halton()
        else:
            self.param_data = param_data

        self.param_names = param_names

        #### Uncertainty ####
        self.sigma_max = sigma_max

        self._inv_normalized_sigma = inv_obs_mat * np.linalg.det(inv_obs_mat) ** (
                -1.0 / inv_obs_mat.shape[0]
        )

    # TODO: this should go to the sampling class
    def generate_params_halton(self):
        """Generate a Halton table of the parameters"""

        from scipy.stats import qmc

        halton_sampler = qmc.Halton(self.num_params, scramble=False)
        param_table = halton_sampler.random(n=self.num_samples)

        for param_i in range(self.num_params):
            for sim_i in range(self.num_samples):
                mean = 0.5 * (self.param_maxs[param_i] + self.param_mins[param_i])
                std = 0.5 * (self.param_maxs[param_i] - self.param_mins[param_i])
                param_table[sim_i][param_i] = (
                        mean + (param_table[sim_i][param_i] - 0.5) * 2 * std
                )

        self.param_data = np.array(param_table, ndmin=2)

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
            ctrl_names=obj.get("ctrl_names", None),
            inv_obs_weight=obj.get("inv_obs_weight", None),
            sim_data=obj.get("sim_data", None),
            callback=obj.get("callback", None),
            param_data=obj.get("param_data", None),
            param_names=obj.get("param_names", None),
        )

    def run(self):
        """This function runs the callback function"""

        if self.callback is None:
            raise ValueError("No callbacallsck function defined")

        self.callback(self)
