---
title: 'PyCS3: A Python toolbox for time-delay measurements in lensed quasars'
tags:
  - Python
  - astronomy
authors:
  - name: Martin Millon
    orcid: 0000-0001-7051-497X
    affiliation: 1
  - name: Malte Tewes
    affiliation: 2
  - name: Vivien Bonvin
    affiliation: 1
  - name: Bastian Lengen
    affiliation: 1
  - name: Frederic Courbin
    affiliation: 1
affiliations:
  - name: Institute of Physics, Laboratory of Astrophysique, Ecole Polytechnique Fédérale de Lausanne (EPFL), Switzerland
    index: 1
  - name: Argelander-Institut für Astronomie, Bonn, Germany
    index: 2 
date: 24 June 2020
bibliography: paper.bib
---

# Summary
Time-delay cosmography is a competitive technique for measuring the current expansion rate of the Universe, that is, the Hubble Constant [see, e.g., @Riess2019 for a review and @Wong:2019 for recent results]. It relies on the strong gravitational lensing effect that happens when a massive foreground galaxy deviates the light from a background object, producing 2 or 4 mirage images of the same background source. In this configuration, the optical path length is slightly different in each multiple image and thus, the travel time of the photons along those paths is also slightly different. If the background source is varying, the same variations are visible in all multiple images with different delays. Lensed quasars or lensed supernovae are ideal targets to measure such (relative) time delays, because they are variable on short timescale, and are sufficiently bright to be observed at cosmological distance. These measured delays can be used to infer the so-called time-delay distance, $D_{\Delta t}$, which is directly inversely proportional to the Hubble Constant. The method relies on three main ingredients : 

 - a precise and accurate determination of the time delays
 - a model of the mass distribution of the lensing galaxy 
 - an estimate of the mass of all the galaxies along the line of sight that also deviates the light rays, and thus perturbs the time delays. 
 
 Obtaining the time delays of lensed quasars typically requires a decade of continuous observation to produce long light curves which contain several variations of the quasars that can unambiguously matched between the multiple images. An example of the light curves of the multiple images of the lensed quasar RXJ1131-1231 is presented in \autoref{fig:lcs}. The aim of the PyCS3 software is to measure time delays between such curves.

![Light curves of the lensed quasar RXJ1131-1231 presented in @Millon1:2020 (left panel). The same quasar variations can be seen in image D 92 days after in image A, whereas images A, B, and C arrive approximately at the same time. The right panel shows an Hubble Space Telescope image of RXJ1131-1231 [@Suyu2017].\label{fig:lcs}](RXJ1131.png)
 
 
# Statement of need
 The "simple" problem of measuring time delays between irregularly sampled light curves has received attention for almost three decades [e.g., @Press1992]. A complicating factor is the microlensing of the multiple images which happens when stars in the lens galaxy are passing in front of the quasar images, also acting as gravitational lenses and affecting the shape of the light curves. Microlensing effects perturb the light curve of each quasar image individually. Ignoring these perturbations would often result in significant biases on the measured time delays, that directly propagate to the Hubble Constant.
 
 ``PyCS3`` is a python package developed by the [COSMOGRAIL](www.cosmograil.org) collaboration and designed to address this problem. It allows us to measure time delays in lensed quasars in the presence of microlensing by providing a flexible and data-driven model of the "extrinsic" variations with splines to account for microlensing and recover an accurate estimate of the time delays. A realistic estimation of the uncertainties is also extremely important for cosmography. The approach followed by ``PyCS3`` is based on best-fit point estimation and a framework to faithfully simulate the input data and assess the uncertainties following a Monte Carlo approach. While a bayesian inference of the delay, given a model for quasar variability and microlensing, would be very attractive, it is hampered by the difficulty to accurately model microlensing and the high number of nuisance parameters in the problem.
 
  The previous version of the package (``PyCS``) was first presented in @Tewes1:2013 and successfully applied to real data in @Tewes2:2013; @Bonvin:2017; @Bonvin:2018 and @Bonvin:2019. The method was also tested on simulated light curves of the Time-Delay Challenge [@Liao:2016; @Bonvin:2016] and was empirically shown to provide excellent results, even if the light curves are strongly affected by microlensing.
   
We have now developed an automated pipeline based on ``PyCS3`` to measure time delays in a large sample of lensed quasars [@Millon1:2020; @Millon2:2020]. Such improvements toward automation of the procedure is necessary with the hundreds of new lensed quasars expected to be discovered in the near future. 

# Functionality
 The basic functionality of ``PyCS3`` is built around a LightCurve class to manipulate photometric monitoring data. It has methods to import, shift, fit and export light curves. These are located in the `pycs3.gen` subpackage.

``PyCS3`` contains two time-delay estimators, namely the free-knot splines and the regression difference, that are in the `pycs3.spl` and `pycs3.regdiff` subpackages. These two estimators are fundamentally different and allows us to check the robustness of the measured time delays. The subpackage `pycs3.sim` is used to generate simulated light curves in order to estimate the uncertainties of the time-delay measurements. ``PyCS3`` ensures that the simulated curves have the same constraining power than the original data which is crucial for a correct estimation of the uncertainties. These simulated curves can then be shifted with either the free-knot splines or the regression difference estimator. 

The ``script`` folder contains a pipeline to automate the measurement of the time delays. The functions that are used by this pipeline are located in the `pycs3.pipe` subpackage. It automatically explores several set of estimator parameters, generates simulated light curves, returns the best fit value and the associated uncertainties before selecting and combining the different sets of estimator parameters. It also makes use of the subpackage `pycs3.tdcomb` to display and combine the final time-delay estimates.  The details of the method can be found in @Millon1:2020. 

Finally, the ``tdlmc_test`` folder contains an ensemble of python scripts to apply ``PyCS3`` on the simulated data generated for the Time-Delay Challenge [@Liao:2016]. It provides a benchmark test framework to assess the precision and accuracy of the ``PyCS3``'s curve-shifting algorithms. 

# Acknowledgement

We acknowledge the support of the Swiss National Science Foundation (SNSF) and the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme (COSMICLENS: grant agreement No 787886).

# References
