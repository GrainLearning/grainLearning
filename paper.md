---
title: 'GrainLearning: A Bayesian uncertainty quantification toolbox for discrete and continuum numerical models of granular materials'
tags:
  - Bayesian inference
  - Calibration
  - Discrete element method
  - Granular materials
  - Uncertainty Quantification
  - Multi-particle simulation
authors:
  - name: Luisa Orozco
    orcid: 0000-0002-9153-650X
    corresponding: true # corresponding author
    equal-contrib: true
    affiliation: 1
  - name: Aron Jansen
    orcid: 0000-0002-4764-9347
    equal-contrib: true
    affiliation: 1
  - name: Retief Lubbe
    equal-contrib: true
    affiliation: 2
  - name: Hongyang Cheng
    orcid: 0000-0001-7652-8600
    equal-contrib: true
    affiliation: 2
affiliations:
 - name: Netherlands eScience center, The Netherlands
   index: 1
 - name: Soil Micro Mechanics (SMM), Faculty of Engineering Technology, MESA+, University of Twente, The Netherlands
   index: 2
date: 13 January 2024
bibliography: paper.bib
---

# Summary

How to keep dikes safe with rising sea levels? Why are ripples formed in sand? What can we prepare for landing on Mars? At the center of these questions is the understanding of how the grains, as a self-organizing material, collide, flow, or get jammed and compressed. State-of-the-art algorithms allow for simulating millions of grains individually in a computer. However, such computations can take very long and produce complex data difficult to interpret and be upscaled to large-scale applications such as sediment transport and debris flows. GrainLearning is an open-source toolbox with machine learning and statistical inference modules allowing for emulating granular material behavior and learning material uncertainties from real-life observations.

# Statement of need

Understanding the link from particle motions to the macroscopic material response is essential to develop accurate models for processes such as 3D printing with metal powders, pharmaceutical powder compaction, flow and handling of cereals in the alimentary industry, grinding and transport of construction materials. Discrete Element Method (DEM) has been used widely as the fundamental tool to produce the data to understand such link. However, DEM simulations are highly computationally intensive and some of the parameters used in the contact laws cannot be directly measured experimentally.

GrainLearning [@Cheng2023] arises as a tool for Bayesian calibration of such computational models, which means the model parameters are estimated with a certain level of uncertainty, constrained on (noisy) real-world observations. Effectively, this makes the simulations digital twins of real-world processes with uncertainties propagated on model outputs, which then can be used for optimization or decision-making.

GrainLearning started in the geotechnical engineering community and was primarily used for granular materials in quasi-static, laboratory conditions [@Cheng2018a; @Cheng2019]. These include triaxial [@Hartmann2022; @LI2024105957] and oedometric [@Cheng2019] compressions of soil samples.
In the particle technology community, attempts with GrainLearning have been made to identify contact parameters for polymer and pharmaceutical powders against angle-of-repose [@essay91991], shear cell [@Thornton2023], and sintering experiments [@ALVAREZ2022117000]. Satisfactory results have been obtained in simulation cases where the grains were in dynamic regimes or treated under multi-physical processes.

# Functionality

- **Calibration**: By means of Sequential Monte Carlo filtering GrainLearning can infer and update model parameters. By learning the underlying distribution using a variational Gaussian model, highly probable zones are identified and sampled iteratively until a tolerance for the overall uncertainty is reached. This process requires the input of: a time series reference data, the ranges of the parameters to infer and a tolerance. The software iteratively minimizes the discrepancy between the model solution and the reference data.
- **Surrogate modeling**: Besides using direct simulation results (e.g. DEM) GrainLearning offers the capability of building surrogates (e.g. recurrent neural networks) as an alternative to computationally expensive DEM simulations, effectively reducing the cost by several orders of magnitude.

# Acknowledgements

The last author would like to thank the Netherlands eScience Center for the funding provided under grant number NLESC.OEC.2021.032.

# References