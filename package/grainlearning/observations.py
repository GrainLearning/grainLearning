import typing as t
import numpy as np
import pandas as pd


class Observations:
    """This class containes the reference/observation data."""

    #: Number of steps or sequence size in the dataset.
    num_steps: int = 0
    #: Number of observations in the dataset
    num_obs: int = 0

    #: array containing the data (e.g., axial stress or volumetric strain)
    data: np.ndarray
    
    #: observation keys
    names: list[str]

    #: Control data
    ctrl = np.ndarray

    #: observation control (e.g., axail strain or time)
    ctrl_name: str

    def __init__(
        self, data: np.array, ctrl: np.array, names: list[str], ctrl_name: str
    ):
        """Initialize the Ibservations class.

        :param data: array containing the reference data (e.g., from experiments).
        :param names: List of names corresponding to dataset (measurements).
        :param ctrl: The control or reference data.
        :param ctrl_name: The control name corresponding to the reference data.
        """
        self.data = np.array(data, ndmin=2)
        self.names = names

        self.ctrl = np.array(ctrl)
        
        self.ctrl_name = ctrl_name

        self.num_obs, self.num_steps = self.data.shape

    @classmethod
    def from_dict(
        cls: t.Type["Observations"],
        obj: dict,
    ):
        """Create an Observations class from a dictionary.

        Example:

        The input will look like this

        .. code-block:: json
        
            {
                "data": [[0.1, 0.5, 0.9],[0.007, 0.011, 0.034]],
                "ctrl": [0.01, 0.02, 0.03],
                "ctrl_key": "axail_strain",
                "data_keys": ["stress","volumetric_strain"]
            }


        :param cls: The Observations class referenced to itself.
        :param obj: Dictionary containing the input parameters to the object.
        :return: An initialized Observations object
        """
        return cls(
            data=obj["data"],
            ctrl=obj["ctrl"],
            names=obj["names"],
            ctrl_name=obj["ctrl_name"],
        )
