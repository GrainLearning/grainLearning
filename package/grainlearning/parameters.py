from typing import List,Type
import numpy as np


# TODO add so that the user can input a dataframe with initial parameters
class Parameters:
    """This class contains the data and methods related to parameters.
    
    Initialize the module like this:

    .. highlight:: python
    .. code-block:: python

        parameters = Parameters(
            names=["E","pois"],
            mins=[1e6,0.05,7.],
            maxs=[4e7,1.0,13.]
        )
        
    :param names: The names of the parameters.
    :param mins: A list of containing the minimum values a parameter may have.
    :param maxs: A list of containing the maximum values a parameter may have.


    """

    #: array containing the parameters.
    data: np.array

    #: The minimum values of the parameters.
    mins: List[float] = []

    #: The maximum values of the parameters.
    maxs: List[float] = []

    #: Names of the parameters.
    names: List[str]

    #: Number of parameters.
    num_params: int

    #: The data of the previous calibration iterations.
    data_records: List = []

    def __init__(self, names: List[str], mins: List[float], maxs: List[float]):
        """Initialize the Parameters class."""
        self.names = names
        self.mins = mins
        self.maxs = maxs
        self.num_params = len(names)

    @classmethod
    def from_dict(cls: Type["Parameters"], obj: dict) ->Type["Parameters"]:
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
        return cls(names=obj["names"], mins=obj["mins"], maxs=obj["maxs"])

    def generate_halton(self, num_samples: int):
        """Generate a Halton table of the parameters.

        :param num_samples: number of simulations
        """

        from scipy.stats import qmc

        halton_sampler = qmc.Halton(self.num_params, scramble=False)
        param_table = halton_sampler.random(n=num_samples)

        for i in range(self.num_params):
            for j in range(num_samples):
                mean = 0.5 * (self.maxs[i] + self.mins[i])
                std = 0.5 * (self.maxs[i] - self.mins[i])
                param_table[j][i] = mean + (param_table[j][i] - 0.5) * 2 * std

        self.data = np.array(param_table, ndmin=2)
        self.data_records.append(self.data)
