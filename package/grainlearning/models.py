#%%
from typing import Type, List, Callable, Tuple
import numpy as np


class Model:
    """This is a base class which is used to call a user defined model.

        It contains the :class:`.Parameters` and the :class:`.Observations` classes.

        The number of samples, parameters and observations should be set first.

        Initialize a Model like this:

        .. highlight:: python    def check_input(self, obj: dict):

        .. code-block:: python

            class MyModel(Model):
                parameters = Parameters(
                    names=["k", "t"],
                    mins=[100, 0.1],
                    maxs=[300, 10],
                )
                observations = Observations(
                    data=[100, 200, 300], ctrl=[-0.1, -0.2, -0.3], names=["F"], ctrl_name=["x"]
                )
                num_samples = 10

                def __init__(self):
                    self.parameters.generate_halton(self.num_samples)
    obs_weight
                def run(self):
                    # for each parameter calculate the spring force
                    data = []

                    for params in self.parameters.data:
                        F = params[0]*params[1]*self.observations.ctrl
                        data.append(np.array(F,ndmin=2))

                    self.data = np.array(data)

    """

    ##### Parameters #####
    #: Parameter data of shape (num_samples,num_params)
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

    #: Observation data (e.g., axial stress or volumetric strain) of shape (num_obs,num_steps)
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

    #: Observation control (e.g., axail strain or time)
    ctrl_names: List[str]

    #: Number of control data
    num_ctrl: int

    ##### Simulations #####

    #: Simulation data of shape (num_samples,num_obs,num_steps)
    sim_data: np.ndarray

    #: Number of samples (usually specified by user)
    num_samples: int

    #: Callback function
    callback: Callable

    ##### Uncertainty #####

    #: Minimum value of the uncertainty
    sigma_min: float = 1.0e-6

    #: Maximum value of the uncertainty
    sigma_max: float = 1.0e6
    
    #: calculated normalized sigma to weigh the covariance matrix
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

        

    def generate_params_halton(self):
        """Generate a Halton table of the parameters.

        :param num_samples: number of simulations
        """

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
        """The class can also be initialized using a dictionary style.

        :param cls: The Parameters class referenced to itself.
        :param obj: Dictionary containing the input parameters to the object.
        :return: An initialized Parameters object

        Example usage:

        .. highlight:: python
        .. code-block:: python

            parameters = Parameters.from_dict({
                "names": ["E", "Eta", "Psi"],
                "mins": [1e6,0.05,7.],
                "maxs": [4e7,1.0,13.],
            })

        """

        assert "obs_data" in obj.keys(), "Error no obs_data key found in input"
        assert "num_samples" in obj.keys(), "Error no num_samples key found in input"
        assert "param_mins" in obj.keys(), "Error no param_mins key found in input"
        assert "param_maxs" in obj.keys(), "Error no param_maxs key found in input"

        return cls(
            obs_data=obj["obs_data"],
            num_samples=obj["num_samples"],
            param_mins=obj["param_mins"],
            param_maxs=obj["param_maxs"],
            ctrl_data=obj.get("ctrl_data",None),
            obs_names=obj.get("obs_names", None),
            ctrl_names=obj.get("ctrl_names", None),
            inv_obs_weight=obj.get("inv_obs_weight", None),
            sim_data=obj.get("sim_data", None),
            callback=obj.get("callback", None),
            param_data=obj.get("param_data", None),
            param_names=obj.get("param_names", None),
        )

    def run(self):
        """By default this function calls the callback function. It can be overwritten as well"""

        if self.callback is None:
            raise ValueError("No callback function defined")

        self.callback(self)
