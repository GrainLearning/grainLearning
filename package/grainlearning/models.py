
from typing import Type, List
import numpy as np
from .parameters import Parameters
from .observations import Observations


class Model:
    """This is a base class which is used to call a user defined model. 
    
    It contains the :class:`.Parameters` and the :class:`.Observations` classes.
    
    The number of samples, parameters and observations should be set first.
    
    Initialize a Model like this:
    
    .. highlight:: python
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

            def run(self):
                # for each parameter calculate the spring force
                data = []
                
                for params in self.parameters.data:
                    F = params[0]*params[1]*self.observations.ctrl
                    data.append(np.array(F,ndmin=2))
                    
                self.data = np.array(data)

    """

    #: Parameter class containing parameter data.
    parameters: Type["Parameters"]
    
    #: Observation class containing the reference data.
    observations: Type["Observations"]
    
    #: This is the simulated data which should have the shape (num_samples,num_obs,num_steps)
    data: np.array
    
    #: This is a list of simulation data in the previous iterations
    data_records: List[np.array] = []

    #: Number of samples (usually specified by user)
    num_samples: int = 0
    
    def run(self):
        """This function is called to populate the model"""
        pass
    
    # def get_control_data(self)->pd.Series:
    #     """ Gets control data. 
        
    #     This can be used in the simulations.

    #     :return: A Dataframe view containing only the
    #     """
    #     # Note slice(None) gives all content at that level
    #     return self.data.loc[:,self.observations.control]

    # def get_key_data(self):
    #     """ Gets  the key data. 

    #     :return: A Dataframe view containing only the keys
    #     """
    #     return self.data.loc[:,self.observations.keys]
