""" Author: Hongyang Cheng <chyalexcheng@gmail.com>
     A collection of all kins of helper functions (IO, plotting, ...) 
"""

from math import *
import sys, os
import numpy as np

from sklearn.mixture import BayesianGaussianMixture

import subprocess
from typing import Type, List, Callable, Tuple

def startSimulations(platform,software,tableName,fileName):   
 #platform desktop, aws or rcg    # software so far only yade 
 argument= tableName+" "+fileName
 if platform=='desktop':
     # Definition where shell script can be found
     path_to_shell = os.getcwd()+'/platform_shells/desktop' 
     if software=='yade':
         command = 'sh '+path_to_shell+'/yadeDesktop.sh'+" "+argument  
         subprocess.call(command, shell=True)  
     else:
         print(Fore.RED +"Chosen 'software' has not been implemented yet. Check 'startSimulations()' in 'tools.py'")
         sys.exit
         
 elif platform=='aws':  
     path_to_shell = os.getcwd()+'/platform_shells/aws' 
     if software=='yade':
         command = 'sh '+path_to_shell+'/yadeAWS.sh'+" "+argument  
         subprocess.call(command, shell=True)  
     else:
         print(Fore.RED +"Chosen 'software' has not been implemented yet. Check 'startSimulations()' in 'tools.py'")
         sys.exit
  
 elif platform=='rcg':  
     path_to_shell = os.getcwd()+'/platform_shells/rcg' 
     if software=='yade':
         command = 'sh '+path_to_shell+'/yadeRCG.sh'+" "+argument  
         subprocess.call(command, shell=True)  
     else:
         print(Fore.RED +"Chosen 'software' has not been implemented yet. Check 'startSimulations()' in 'tools.py'")
         sys.exit         
 else:
  print('Exit code. Hardware for yade simulations not properly defined')
  quit()


def write_to_table(sim_name, table, names, curr_iter = 0, threads = 8):
    """
    write parameter samples into a text file
    """
    
    # Computation of decimal number for unique key 
    table_file_name = f'{sim_name}_iter{curr_iter}_samples.txt'
    
    fout = open(table_file_name, 'w')
    num, dim = table.shape
    magn = floor(log(num, 10)) + 1
    fout.write(' '.join(['!OMP_NUM_THREADS', 'description', 'key'] + names + ['\n']))
    for j in range(num):
       description = 'Iter'+str(curr_iter)+'-Sample'+str(j).zfill(magn)
       fout.write(' '.join(['%2i' % threads] + [description] + ['%9i' % j] + ['%20.10e' % table[j][i] for i in range(dim)] + ['\n']))
    fout.close()
    return table_file_name


def get_keys_and_data(fileName):
    """
    Get keys and corresponding data sequence from a Yade output file

    :param fileName: string

    :return: keysAndData: dictionary
    """
    data = np.genfromtxt(fileName)
    fopen = open(fileName, 'r')
    keys = (fopen.read().splitlines()[0]).split('\t\t')
    if '#' in keys: keys.remove('#')
    keys_and_data = {}
    for key in keys:
        if '#' in key: key_no_hash = key.split(' ')[-1]
        else: key_no_hash = key
        keys_and_data[key_no_hash] = data[:, keys.index(key)]
    return keys_and_data


def regenerate_params_with_gmm(
        proposal: np.ndarray,
        param_data: np.ndarray,
        num: int,
        max_num_components: int,
        prior_weight: float,
        cov_type: str = "full",
        resample_to_unweighted: Callable = None,
        param_mins: List[float] = None,
        param_maxs: List[float] = None,
        n_init = 1,
        tol = 0.001,
        max_iter = 100,
        seed = None,
    ) -> np.ndarray:
    """
    Resample parameters using a variational Gaussian mixture model

    :param proposal: ndarray of shape model.num_samples
        proposal probability distribution associated to the current parameter data
    
    :param param_data: ndarray of shape (model.num_samples, model.num_params)
        current parameter data

    :param num: int
        number of samples for the resampling

    :param max_num_components: int, default = num/10

    :param prior_weight: float, default = 1./maxNumComponents
        weight_concentration_prior of the BayesianGaussianMixture class
        The dirichlet concentration of each component on the weight distribution (Dirichlet).
        This is commonly called gamma in the literature.
        The higher concentration puts more mass in the center and will lead to more components being active,
        while a lower concentration parameter will lead to more mass at the edge of the mixture weights simplex.
        (https://scikit-learn.org/stable/modules/generated/sklearn.mixture.BayesianGaussianMixture.html)

    :param cov_type: string, default = 'full'
        covariance_type of the BayesianGaussianMixture class
        String describing the type of covariance parameters to use. Must be one of:
        'full' (each component has its own general covariance matrix),
        'tied' (all components share the same general covariance matrix),
        'diag' (each component has its own diagonal covariance matrix),
        'spherical' (each component has its own single variance).
        (https://scikit-learn.org/stable/modules/generated/sklearn.mixture.BayesianGaussianMixture.html)

    :param resample_to_unweighted: Callable
        Function to expand samples from weighted to unweighted

    :param param_mins: list
        lower bound of the parameter values

    :param param_maxs: list
        uper bound of the parameter values

    :param seed: int
        random generation seed, defaults to None

    :return:
        new_param_data: ndarray, parameter samples for the next iteration

        gmm: BayesianGaussianMixture
            A variational Gaussian mixture model trained with current parameter samples and proposal probabilities
    """

    # expand the parameters from a proposal distribution represented via importance sampling
    indices = resample_to_unweighted(proposal)
    expanded_param_data = param_data[indices]

    # normalize parameter samples
    max_params = np.amax(expanded_param_data, axis=0)  # find max along axis

    expanded_param_data = (
            expanded_param_data / max_params
    )  # and do array broadcasting to divide by max

    gmm = BayesianGaussianMixture(
        n_components=max_num_components,
        weight_concentration_prior=prior_weight,
        covariance_type=cov_type,
        n_init = n_init,
        tol = tol,
        max_iter = max_iter,
        random_state = seed,
    )

    gmm.fit(expanded_param_data)
    new_param_data, _ = gmm.sample(num)
    new_param_data *= max_params

    return new_param_data, gmm


def get_pool(mpi=False, threads=1):
    """
    Create a thread pool for paralleling DEM simulations within GrainLearning

    :param mpi: bool, default=False

    :param threads: int, default=1
    """
    if mpi:  # using MPI
        from mpipool import MPIPool
        pool = MPIPool()
        pool.start()
        if not pool.is_master():
            sys.exit(0)
    elif threads > 1:  # using multiprocessing
        from multiprocessing import Pool
        pool = Pool(processes=threads, maxtasksperchild=10)
    else:
        raise RuntimeError("Wrong arguments: either mpi=True or threads>1.")
    return pool


def unweighted_resample(weights, expand_num=10):
    # take int(N*w) copies of each weight, which ensures particles with the same weight are drawn uniformly
    N = len(weights)*expand_num
    num_copies = (np.floor(N*np.asarray(weights))).astype(int)
    indexes = np.zeros(sum(num_copies), 'i')
    k = 0
    for i in range(len(weights)):
        for _ in range(num_copies[i]): # make n copies
            indexes[k] = i
            k += 1
    return indexes

def residual_resample(weights, expand_num=10):
    N = len(weights)*expand_num
    indexes = np.zeros(N, 'i')

    # take int(N*w) copies of each weight, which ensures particles with the
    # same weight are drawn uniformly
    num_copies = (np.floor(N*np.asarray(weights))).astype(int)
    k = 0
    for i in range(len(weights)):
        for _ in range(num_copies[i]): # make n copies
            indexes[k] = i
            k += 1

    # use multinormal resample on the residual to fill up the rest. This
    # maximizes the variance of the samples
    residual = weights - num_copies     # get fractional part
    residual /= sum(residual)           # normalize
    cumulative_sum = np.cumsum(residual)
    cumulative_sum[-1] = 1. # avoid round-off errors: ensures sum is exactly one
    indexes[k:N] = np.searchsorted(cumulative_sum, np.random.random(N-k))

    return indexes



def stratified_resample(weights, expand_num=10):
    """ Performs the stratified resampling algorithm used by particle filters.
    This algorithms aims to make selections relatively uniformly across the
    particles. It divides the cumulative sum of the weights into N equal
    divisions, and then selects one particle randomly from each division. This
    guarantees that each sample is between 0 and 2/N apart.
    Parameters
    ----------
    weights : list-like of float
        list of weights as floats
    Returns
    -------
    indexes : ndarray of ints
        array of indexes into the weights defining the resample. i.e. the
        index of the zeroth resample is indexes[0], etc.
    """

    N = len(weights)
    # make N subdivisions, and chose a random position within each one
    positions = (np.random.random(N) + range(N)) / N

    indexes = np.zeros(N, 'i')
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes


def systematic_resample(weights, expand_num=10):
    """ Performs the systemic resampling algorithm used by particle filters.
    This algorithm separates the sample space into N divisions. A single random
    offset is used to to choose where to sample from for all divisions. This
    guarantees that every sample is exactly 1/N apart.
    Parameters
    ----------
    weights : list-like of float
        list of weights as floats
    Returns
    -------
    indexes : ndarray of ints
        array of indexes into the weights defining the resample. i.e. the
        index of the zeroth resample is indexes[0], etc.
    """
    N = len(weights)

    # make N subdivisions, and choose positions with a consistent random offset
    positions = (np.random.random() + np.arange(N)) / N

    indexes = np.zeros(N, 'i')
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes


def multinomial_resample(weights, expand_num=10):
    """ This is the naive form of roulette sampling where we compute the
    cumulative sum of the weights and then use binary search to select the
    resampled point based on a uniformly distributed random number. Run time
    is O(n log n). You do not want to use this algorithm in practice; for some
    reason it is popular in blogs and online courses so I included it for
    reference.
   Parameters
   ----------
    weights : list-like of float
        list of weights as floats
    Returns
    -------
    indexes : ndarray of ints
        array of indexes into the weights defining the resample. i.e. the
        index of the zeroth resample is indexes[0], etc.
    """
    cumulative_sum = np.cumsum(weights)
    cumulative_sum[-1] = 1.  # avoid round-off errors: ensures sum is exactly one
    return np.searchsorted(cumulative_sum, np.random.random(len(weights)*expand_num))


def voronoi_vols(samples: np.ndarray):
	from scipy.spatial import Voronoi, ConvexHull
	v = Voronoi(samples)
	vol = np.zeros(v.npoints)
	for i, reg_num in enumerate(v.point_region):
		indices = v.regions[reg_num]
		if -1 in indices:
			vol[i] = -1.0
		else:
			vol[i] = ConvexHull(v.vertices[indices]).volume
	return vol
