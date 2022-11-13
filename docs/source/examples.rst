Examples
========

In particle simulations, plastic deformation at the macro scale
arises from contact sliding in the tangential and rolling/twisting directions
and the irrecoverable change of the microstructure.
Because the parameters relevant to these microscopic phenomena
are not directly measurable in a laboratory, calibration of DEM models
is generally treated as an inverse problem using "inverse methods",
ranging from trial and error to sophisticated statistical inference.
Solving an inverse problem that involves nonlinearity and/or discontinuity
in the *forward* model (DEM) is very challenging.
Furthermore, because of the potentially large computational cost
for running the simulations, the "trials" have to be selected with an optimized strategy to boost the efficiency.


In GrainLearning, the probability distribution of model states and parameters, conditioned on given reference data (termed "posterior distribution") can be approximated by sequential Monte Carlo methods.
To efficiently sample parameter space, a multi-level (re)sampling algorithm is utilized.
For the first iteration of Bayesian filtering, the parameter values are uniformly sampled from quasi-random numbers, which leads to conventional sequential quasi-Monte Carlo (SQMC) filtering.
For the subsequent iterations, new parameter values are drawn from the posterior distribution updated by the previous iteration.
Iterative Bayesian filtering allows us to sample near potential posterior modes in parameter space, with an increasing sample density over the iterations, until the ensemble predictions of the model parameters converge.

Because the posterior distribution of the model parameters is typically unknown, a non-informative proposal density is used to uniformly explore the parameter space in the first iteration (<img src="https://latex.codecogs.com/svg.latex?k=0" title="k=0" />).
The sampling is done with quasi-random numbers (defined an initial guess of the upper and lower bounds of parameters), which leads to the so-called sequential quasi-Monte Carlo (SQMC) filter with an number of samples scales with <img src="https://latex.codecogs.com/svg.latex?d\log{d}" title="d\log{d}" />.
Although a uniform initial sampling is unbiased, it is very inefficient and ensured to degenerate as time approaches infinity.
To avoid circumvent weight degeneracy, the inverse problem is solved again from time *0* to *T* with new samples drawn from a more sensible proposal distribution and reinitialized weights.

The posterior distribution of the parameters resulting from the previous Bayesian filtering <img src="https://latex.codecogs.com/svg.latex?p_{k-1}(\vec{\Theta}|\vec{y}_{1:T})" title="p_{k-1}(\vec{\Theta}|\vec{y}_{1:T})" /> is chosen as the proposal distribution; the parameter samples <img src="https://latex.codecogs.com/svg.latex?\vec{\Theta}^{(i)}" title="\vec{\Theta}^{(i)}" /> are drawn from
<img src="https://latex.codecogs.com/svg.latex?p_{k-1}(\vec{\Theta}|\vec{y}_{1:T})" title="p_{k-1}(\vec{\Theta}|\vec{y}_{1:T})" />.
For each iteration with <img src="https://latex.codecogs.com/svg.latex?k>0" title="k>0" />, the proposal density is constructed by training a nonparametric Gaussian mixture with the samples and importance weights from the previous iteration.
Because the closer to a posterior distribution mode the higher the sample density, resampling from the repeatedly updated proposal density allows to zoom into highly probable parameter subspace in very few iterations.
The iterative (re)sampling scheme brings three major advantages to the Bayesian filtering framework:

 1. The posterior distribution is iteratively estimated with an increasing resolution on the posterior landscape.
 2. The multi-level sampling algorithm keeps allocating model evaluations in parameter subspace where the posterior probabilities are expected to be high, thus significantly improving computational efficiency.
 3. Resampling that takes place between two consecutive iterations can effectively overcome weight degeneracy problem while keeping sample trajectories intact within the time/load history.
