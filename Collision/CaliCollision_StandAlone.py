"""
Author: Hongyang Cheng <chyalexcheng@gmail.com>
Bayesian calibration of Hertzian contact parameters using two-particle collision simulation
"""

import numpy as np
import sys

# load GrainLearning modules
sys.path.append('../')
from smc import *
from plotResults import *
import pickle
import matplotlib.pylab as plt

# user-defined parameter: upper limit of the normalized covariance coefficient
sigma = float(input("Give an initial guess of the upper limit of normalized covariance: "))
# target effective sample size
ess = 0.3
obsWeights = [1.0]
# maximum number of iterations
maxNumOfIters = 10
# number of threads
threads = 1

# get observation data file (synthetic data from DEM)
ObsData = np.loadtxt('collisionOrg.dat')
# add Gaussian noise
noise = np.random.normal(0, 0.3 * max(ObsData[:,1]), ObsData.shape[0])

# give ranges of parameter values (E_m, \nu)
paramNames = ['E_m', 'nu']
numParams = len(paramNames)
# use uniform sampling for the first iteration
paramRanges = {'E_m': [7, 11], 'nu': [0.0, 0.5]}
# key for simulation control
obsCtrl = 'u'
# key for output data
simDataKeys = ['f']

# set number of samples per iteration (e.g., num1D * N * logN for quasi-Sequential Monte Carlo)
numSamples1D = 10
numSamples = int(numSamples1D * numParams * log(numParams))
# set the maximum Gaussian components and prior weight
maxNumComponents = int(numSamples / 10)
priorWeight = 1. / maxNumComponents
covType = 'tied'

# write synthetic observation data to file
obsDataFile = open('collisionObs.dat', 'w')
obsDataFile.write('#\t\tu\t\tf\n')
for i in range(ObsData.shape[0]):
    obsDataFile.write('%s\t\t%s\n' % (ObsData[i,0], noise[i] + ObsData[i,1]))
obsDataFile.close()

# instantiate the problem
iterNO = int(input('\nWhich iteration of Bayesian calibration is this?\n'))
# stand-alone mode: use GrainLearning as a post-process tool using pre-run DEM data
smcTest = smc(sigma, ess, obsWeights,
              yadeDataDir='cali_results', threads=threads,
              obsCtrl=obsCtrl, simDataKeys=simDataKeys, simName='2particle', obsFileName='collisionObs.dat',
              seed=None, loadSamples=True, runYadeInGL=False, standAlone=True)

# load or generate the initial parameter samples
smcTest.initParams(paramNames, paramRanges, numSamples, paramsFile='smcTable%i.txt' % iterNO, subDir='iter%i' % iterNO)

# initialize the weights
# include "proposalFile='gmmForCollision_%i.pkl' % iterNO" as a function parameter
# to take into account proposal probabilities, avoiding bias in resampling
smcTest.initialize(maxNumComponents, priorWeight, covType=covType)

# sequential Monte Carlo
ips, covs = smcTest.run(iterNO=iterNO)

# get the parameter samples (ensemble) and posterior probability
posterior = smcTest.getPosterior()
smcSamples = smcTest.getSmcSamples()

# plot means of PDF over the parameters
plotIPs(paramNames, ips.T, covs.T, smcTest.getNumSteps(), posterior, smcSamples[0])

# resample parameters
caliStep = -1
gmm, maxNumComponents = smcTest.resampleParams(caliStep=caliStep, paramRanges=paramRanges, iterNO=iterNO)

# plot initial and resampled parameters
plotAllSamples(smcTest.getSmcSamples(), smcTest.getNames())

# save trained Gaussian mixture model
pickle.dump(gmm, open('gmmForCollision_%i' % (iterNO + 1) + '.pkl', 'wb'))

# plot ground truth and added Gaussian noise
plt.figure('Comparison')
plt.plot(smcTest.getObsData()[:, 0], smcTest.getObsData()[:, 1], color='grey', label='Ground truth + noise')
plt.plot(ObsData[:, 0], ObsData[:, 1], 'ko', markevery=smcTest.numObs/10, label='Ground truth')

# get top three realizations with high probabilities
m = smcTest.getNumSteps()
n = smcTest.numSamples
weights = smcTest.getPosterior() * np.repeat(smcTest.proposal, m).reshape(n, m)
weights /= sum(weights)
for i in (-weights[:, caliStep]).argsort()[:3]:
	plt.plot(smcTest.getObsData()[:, 0], smcTest.yadeData[:, i, 0], label='sim%i' % i)
plt.xlabel('Overlap'); plt.ylabel('Force')
plt.legend(); plt.show()
