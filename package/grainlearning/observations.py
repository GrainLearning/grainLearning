from typing import Type, List
import numpy as np


class Observations:
    """This class containes the reference/observation data.

    Initialize the module like this:

    .. highlight:: python
    .. code-block:: python

        observations = Observations(
            data = [[0.1, 0.5, 0.9],[0.007, 0.011, 0.034]],
            ctrl = [0.01, 0.02, 0.03],
            ctrl_key = "axial_strain",
            data_keys: ["stress","volumetric_strain"]
        )

    :param data: array containing the reference data (e.g., from experiments).
    :param names: List of names corresponding to dataset (measurements).
    :param ctrl: The control or reference data.
    :param ctrl_name: The control name corresponding to the reference data.
    """

    #: Number of steps or sequence size in the dataset.
    num_steps: int = 0
    #: Number of observations in the dataset
    num_obs: int = 0

    #: array containing the data (e.g., axial stress or volumetric strain) of shape (num_obs,num_steps)
    data: np.ndarray

    #: observation keys
    names: List[str]

    #: Control data
    ctrl = np.ndarray

    #: observation control (e.g., axail strain or time)
    ctrl_name: str

    def __init__(
        self, data: np.array, ctrl: np.array, names: List[str], ctrl_name: str
    ):
        """Initialize the Ibservations class."""
        self.data = np.array(data, ndmin=2) # ensure data is of shape (num_obs,num_shape).
        
        self.names = names

        self.ctrl = np.array(ctrl)

        self.ctrl_name = ctrl_name

        self.num_obs, self.num_steps = self.data.shape

    @classmethod
    def from_dict(
        cls: Type["Observations"],
        obj: dict,
    ):
        """The class can also be initialized using a dictionary style.

        :param cls: The Observations class referenced to itself.
        :param obj: Dictionary containing the input to the object.
        :return: An initialized Observations object

        Example usage:

        .. highlight:: python
        .. code-block:: python

            observations = Observations.from_dict({
                "data": [[0.1, 0.5, 0.9],[0.007, 0.011, 0.034]],
                "ctrl": [0.01, 0.02, 0.03],
                "ctrl_key": "axail_strain",
                "data_keys": ["stress","volumetric_strain"]
            })

        """
        return cls(
            data=obj["data"],
            ctrl=obj["ctrl"],
            names=obj["names"],
            ctrl_name=obj["ctrl_name"],
        )
