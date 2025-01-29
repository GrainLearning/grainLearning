"""
This module contains tools for the GrainLearning project.
"""
import sys
import os
import math
import subprocess
from typing import List, Callable
import numpy as np
from sklearn.mixture import BayesianGaussianMixture
from scipy.spatial import Voronoi, ConvexHull


def run_yade_from_shell(table_file_name, model_script, path_to_shell=None, platform='desktop'):
    # platform desktop, aws or rcg    # software so far only yade
    if not path_to_shell:
        path_to_shell = os.getcwd()
    argument = table_file_name + " " + model_script
    if platform == 'desktop':
        # Definition where shell script can be found
        command = 'sh ' + path_to_shell + '/yadeDesktop.sh' + " " + argument
        subprocess.call(command, shell=True)

    elif platform == 'aws':
        command = 'sh ' + path_to_shell + '/yadeAWS.sh' + " " + argument
        subprocess.call(command, shell=True)

    elif platform == 'rcg':
        command = 'sh ' + path_to_shell + '/yadeRCG.sh' + " " + argument
        subprocess.call(command, shell=True)
    else:
        RuntimeError(
            "Chosen 'platform' has not been implemented yet. Check 'run_yade_from_shell()' in 'tools.py'")
        exit()


def write_to_table(sim_name, table, names, curr_iter=0, threads=8):
    """
    write parameter samples into a text file

    :param sim_name: string
    :param table: numpy array
    :param names: list of strings
    :param curr_iter: int
    :param threads: int
    :return: string
    """

    # Computation of decimal number for unique key
    table_file_name = f'{os.getcwd()}/{sim_name}_Iter{curr_iter}_Samples.txt'

    with open(table_file_name, 'w') as f_out:
        num, dim = table.shape
        mag = math.floor(math.log(num, 10)) + 1
        f_out.write(' '.join(['!OMP_NUM_THREADS', 'description', 'key'] + names + ['\n']))
        for j in range(num):
            description = f'{sim_name}_Iter' + str(curr_iter) + '_Sample' + str(j).zfill(mag)
            f_out.write(' '.join(
                [f'{threads:2d}'] + [description] +
                [f'{j:9d}'] + [f'{table[j][i]:20.10e}' for i in range(dim)] + ['\n']))
    return table_file_name


def get_keys_and_data(file_name: str, delimiters=None):
    """
    Get keys and corresponding data sequence from a text file

    :param file_name: string
    :param delimiters: list of strings, default = ['\t', ' ', ',']
    :return: keys_and_data: dictionary
    """
    if delimiters is None:
        delimiters = ['\t', ' ', ',']
    data = np.genfromtxt(file_name, ndmin=2)

    with open(file_name, 'r') as f_open:
        first_line = f_open.read().splitlines()[0]
        for d in delimiters:
            keys = first_line.split(d)
            # remove # in the header line
            if '#' in keys:
                keys.remove('#')
            # remove empty strings from the list
            keys = list(filter(None, keys))
            if len(keys) == data.shape[1]:
                break

    # store data in a dictionary
    keys_and_data = {}
    for key in keys:
        if '#' in key:
            key_no_hash = key.split(' ')[-1]
        else:
            key_no_hash = key
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
    param_min: List[float] = None,
    param_max: List[float] = None,
    n_init=1,
    tol=0.001,
    max_iter=100,
    seed=None,
) -> np.ndarray:
    """
    Resample parameters using a variational Gaussian mixture model

    :param proposal: ndarray of shape system.num_samples
        proposal probability distribution associated to the current parameter data

    :param param_data: ndarray of shape (system.num_samples, system.num_params)
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

    :param param_min: list
        lower bound of the parameter values

    :param param_max: list
        upper bound of the parameter values

    :param seed: int
        random generation random_state, defaults to None

    :param n_init: int, default = 1
        Number of initializations to perform. The best results are kept.

    :param tol: float, default = 0.001
        Convergence threshold. EM iterations will stop when the lower bound average gain is below this threshold.

    :param max_iter: int, default = 100
        Number of EM iterations to perform.

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
        n_init=n_init,
        tol=tol,
        max_iter=max_iter,
        random_state=seed,
    )

    gmm.fit(expanded_param_data)
    new_param_data, _ = gmm.sample(num)
    new_param_data *= max_params

    return new_param_data, gmm


def get_pool(mpi=False, threads=1):
    """
    Create a thread pool for paralleling model evaluations within GrainLearning

    TODO improve the scheduler for running simulation instances in parallel
    1. On Desktop: use multiprocessing
    2. On HPC: use MPI or multiprocessing
    3. On Cloud (e.g. AWS)

    :param mpi: bool, default=False
    :param threads: int, default=1
    """
    # if mpi:  # using MPI
    #     from mpipool import MPIPool
    #     pool = MPIPool()
    #     pool.start()
    #     if not pool.is_master():
    #         sys.exit(0)
    if threads > 1:  # using multiprocessing
        from multiprocessing import Pool
        pool = Pool(processes=threads, maxtasksperchild=10)
    if not mpi and threads == 1:
        raise RuntimeError("Wrong arguments: either mpi=True or threads>1.")
    return pool


def unweighted_resample(weights, expand_num=10):
    """
    Resample from the weighted samples to unweighted samples

    Take int(N*w) copies of each weight, which ensures particles with the same weight are drawn uniformly
    :param weights: ndarray of shape (N,)
    :param expand_num: int, default=10
    :return: ndarray of shape (N*expand_num,)
    """
    n = len(weights) * expand_num
    num_copies = (np.floor(n * np.asarray(weights))).astype(int)
    indexes = np.zeros(sum(num_copies), 'i')
    k = 0
    for i in range(len(weights)):
        for _ in range(num_copies[i]):  # make n copies
            indexes[k] = i
            k += 1
    return indexes


def residual_resample(weights, expand_num=10):
    """
    Resample from the weighted samples to unweighted samples using the residual resampling algorithm

    :param weights: ndarray of shape (N,)
    :param expand_num: int, default=10
    :return: ndarray of shape (N*expand_num,)
    """
    n = len(weights) * expand_num
    indexes = np.zeros(n, 'i')

    # take int(N*w) copies of each weight, which ensures particles with the
    # same weight are drawn uniformly
    num_copies = (np.floor(n * np.asarray(weights))).astype(int)
    k = 0
    for i in range(len(weights)):
        for _ in range(num_copies[i]):  # make n copies
            indexes[k] = i
            k += 1

    # use multinormal resample on the residual to fill up the rest. This
    # maximizes the variance of the samples
    residual = weights - num_copies  # get fractional part
    residual /= sum(residual)  # normalize
    cumulative_sum = np.cumsum(residual)
    cumulative_sum[-1] = 1.  # avoid round-off errors: ensures sum is exactly one
    indexes[k:n] = np.searchsorted(cumulative_sum, np.random.random(n - k))

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

    n = len(weights)
    # make N subdivisions, and chose a random position within each one
    positions = (np.random.random(n) + range(n)) / n

    indexes = np.zeros(n, 'i')
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < n:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes


def systematic_resample(weights, expand_num=10):
    """ Performs the systemic resampling algorithm used by particle filters.
    This algorithm separates the sample space into N divisions. A single random
    offset is used to choose where to sample from for all divisions. This
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
    n = len(weights)

    # make N subdivisions, and choose positions with a consistent random offset
    positions = (np.random.random() + np.arange(n)) / n

    indexes = np.zeros(n, 'i')
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < n:
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
    return np.searchsorted(cumulative_sum, np.random.random(len(weights) * expand_num))


def voronoi_vols(samples: np.ndarray):
    """Compute the volumes of the Voronoi cells associated with a set of points.

    :param samples: ndarray of shape (N, D)
    :return: ndarray of shape (N,)
    """
    v = Voronoi(samples)
    vol = np.zeros(v.npoints)
    for i, reg_num in enumerate(v.point_region):
        indices = v.regions[reg_num]
        if -1 in indices:
            vol[i] = -1.0
        else:
            vol[i] = ConvexHull(v.vertices[indices]).volume
    return vol


def plot_param_stats(fig_name, param_names, means, covs, save_fig=0):
    """
    Plot the posterior means and coefficients of variation of the model parameters over time.
    :param fig_name: string
    :param param_names: parameter names
    :param means: ndarray
    :param covs: ndarray
    :param save_fig: bool defaults to False
    """
    import matplotlib.pylab as plt
    num = len(param_names)
    n_cols = int(np.ceil(num / 2))
    plt.figure('Posterior means of the parameters')
    for i in range(num):
        plt.subplot(2, n_cols, i + 1)
        plt.plot(means[:, i])
        plt.xlabel("'Time' step")
        plt.ylabel(f'Mean of {param_names[i]}')
        plt.grid(True)
    if save_fig:
        plt.savefig(f'{fig_name}_param_means.png')

    plt.figure('Posterior coefficients of variance of the parameters')
    for i in range(num):
        plt.subplot(2, n_cols, i + 1)
        plt.plot(covs[:, i])
        plt.xlabel("'Time' step")
        plt.ylabel(f'Coefficient of variation of {param_names[i]}')
        plt.grid(True)
    if save_fig:
        plt.savefig(f'{fig_name}_param_covs.png')


def plot_posterior(fig_name, param_names, param_data, posterior, save_fig=0):
    """
    Plot the evolution of discrete posterior distribution over the parameters in time.
    :param fig_name: string
    :param param_names: parameter names
    :param param_data: ndarray
    :param posterior: ndarray
    :param save_fig: bool defaults to False
    """
    try:
        import matplotlib.pylab as plt
    except ImportError:
        print(
            'matplotlib is not installed, cannot plot posterior distribution. Please install with grainlearning[plot]')
    num_steps = posterior.shape[0]
    for i, name in enumerate(param_names):
        plt.figure(f'Posterior distribution of {name}')
        for j in range(6):
            plt.subplot(2, 3, j + 1)
            plt.plot(param_data[:, i], posterior[int(num_steps * (j + 1) / 6 - 1), :], 'o')
            plt.title(f"'Time' step {int(num_steps * (j + 1) / 6 - 1):3d} ")
            plt.xlabel(r'$' + name + '$')
            plt.ylabel('Posterior probability mass')
            plt.grid(True)
        if save_fig:
            plt.savefig(f'{fig_name}_posterior_{name}.png')


def plot_param_data(fig_name, param_names, param_data_list, save_fig=0):
    import matplotlib.pylab as plt
    import seaborn as sns
    flare_cmap = sns.color_palette("flare", as_cmap=True)
    num = len(param_names)
    n_cols = int(np.ceil(num / 2))
    num = num - 1
    num_iter = len(param_data_list)
    plt.figure('Resampling the parameter space')
    for j in range(num):
        plt.subplot(2, n_cols, j + 1)
        for i in range(num_iter):
            plt.scatter(param_data_list[i][:, j], param_data_list[i][:, j + 1], edgecolors='white',
                        color=flare_cmap(i / (num_iter - 1)), label=f'Iter No. {i:d}')
            plt.xlabel(r'$' + param_names[j] + '$')
            plt.ylabel(r'$' + param_names[j + 1] + '$')
            plt.legend()
    if save_fig:
        plt.savefig(f'{fig_name}_param_space.png')


def plot_obs_and_sim(fig_name, ctrl_name, obs_names, ctrl_data, obs_data, sim_data, posteriors, save_fig=0):
    """
    Plot the ensemble prediction, observation data, and top three best-fits
    :param fig_name: string
    :param ctrl_name: name of the control variable
    :param obs_names: names of the observables
    :param ctrl_data: ndarray
    :param obs_data: ndarray
    :param sim_data: ndarray
    :param posteriors: ndarray
    :param save_fig: bool defaults to False
    """
    import matplotlib.pylab as plt
    ensemble_mean = np.einsum('ijk, ki->jk', sim_data, posteriors)
    ensemble_std = np.einsum('ijk, ki->jk', (sim_data - ensemble_mean) ** 2, posteriors)
    ensemble_std = np.sqrt(ensemble_std)
    num = len(obs_names)
    ncols = int(np.ceil(num / 2)) if num > 1 else 1
    plt.figure('Model prediction versus observation')
    for i in range(num):
        plt.subplot(2, ncols, i + 1)

        plt.fill_between(
            ctrl_data,
            ensemble_mean[i, :] - 2 * ensemble_std[i, :],
            ensemble_mean[i, :] + 2 * ensemble_std[i, :],
            color='darkred',
            label='ensemble prediction'
        )

        for j in (-posteriors[-1, :]).argsort()[:3]:
            plt.plot(ctrl_data, sim_data[j, i, :], label=f'sim No. {j:d}')

        markevery = int(np.ceil(len(ctrl_data) / 100))

        plt.plot(ctrl_data,
                 obs_data[i, :], 'ok',
                 label='obs.',
                 markevery=markevery
                 )

        plt.xlabel(ctrl_name)
        plt.ylabel(obs_names[i])
        # Show legend in the first sub-plot
        if i == 0:
            plt.legend()
        plt.grid(True)

    if save_fig:
        plt.savefig(f'{fig_name}_obs_and_sim.png')


def plot_pdf(fig_name, param_names, samples, save_fig=0, true_params=None):
    """
    Plot the posterior density function of the parameter distribution
    :param fig_name: string
    :param sim_name: string
    :param param_names: list of strings containing parameter names
    :param samples: list of ndarray of shape (num_samples, self.num_params)
    :param save_fig: bool defaults to False
    :param true_params: list of true parameter values, default=None
    """
    import seaborn as sns
    import pandas as pd
    import matplotlib.pylab as plt

    # create a deep copy of the samples (FIXME: this is a workaround for a bug in iBF constructor)
    new_samples = [np.copy(sample) for sample in samples]
    # pad with NaNs to make all samples have the same length
    max_len = int(max([len(sample) for sample in new_samples]))
    for i in range(len(new_samples)):
        new_samples[i] = np.pad(new_samples[i], ((0, max_len - len(new_samples[i])), (0, 0)), 'constant',
                                constant_values=np.nan)
    # fill the third dimension with iteration number
    stacked_samples = np.vstack([new_samples[i] for i in range(len(new_samples))])
    third_dim_indices = np.repeat(np.arange(len(new_samples)), new_samples[0].shape[0])
    new_samples = np.hstack([stacked_samples, third_dim_indices[:, np.newaxis]])

    # convert the array into a pandas dataframe
    new_names = param_names + ['Iter']
    df = pd.DataFrame(new_samples, columns=new_names)

    # plot the parameter distribution in a scatter plot
    fig = sns.PairGrid(df, diag_sharey=False, hue="Iter", palette='flare')
    fig.map_upper(sns.scatterplot, s=15)
    fig.map_lower(sns.kdeplot, thresh=1e-3, levels=5)
    fig.map_diag(sns.kdeplot, lw=2)
    fig.add_legend(loc='upper right')
    fig.fig.canvas.manager.set_window_title('Estimated posterior probability density function')
    if true_params is not None:
        for i in range(len(param_names)):
            for j in range(len(param_names)):
                # plot the true parameter values as a yellow start
                if i != j:
                    # keep the marker on top of other plots
                    fig.axes[i, j].scatter(true_params[j], true_params[i], color='gold', edgecolors='k', marker='*',
                                           s=200, zorder=10)

    fig.tight_layout()
    if save_fig:
        fig.savefig(f'{fig_name}_scatterplot.png')


def close_plots(save_fig=0):
    import matplotlib.pylab as plt
    if not save_fig:
        plt.show()
    plt.close('all')


def write_dict_to_file(data, file_name):
    """Write a python dictionary data into a text file

    :param data: dictionary, keys are the names of the columns, values are the data
    :param file_name: string, name of the file
    """
    with open(file_name, 'w') as f:
        keys = data.keys()
        f.write('# ' + ' '.join(keys) + '\n')
        # check if data[list(keys)[0]] is a list
        if isinstance(data[list(keys)[0]], list):
            num = len(data[list(keys)[0]])
            for i in range(num):
                f.write(' '.join([str(data[key][i]) for key in keys]) + '\n')
        else:
            f.write(' '.join([str(data[key]) for key in keys]) + '\n')
