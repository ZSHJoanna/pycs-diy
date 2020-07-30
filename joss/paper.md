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
affiliations:
  - name: Institute of Physics, Laboratory of Astrophysique, Ecole Polytechnique Fédérale de Lausanne (EPFL), Switzerland
    index: 1
  - name: Argelander-Institut für Astronomie, Bonn, Germany
    index: 2 
date: 24 June 2020
bibliography: paper.bib
---

# Summary
Time-delay cosmography is now a competitive technique for measuring the current expansion rate of the Universe, that is, the Hubble Constant [@Wong:2019]. It relies on the strong gravitational lensing effect that happens when a massive foreground galaxy deviates the light from a background object, producing 2 or 4 mirage images of the same background source. In this configuration, the length of the optical path taken by the photon is slightly different in each multiple image and thus, the travel time of the photon is also slightly different. If the background source is varying, the same variations are visible in all multiple images but delayed in time. Lensed quasars or lensed supernovae are ideal targets because they are variable on short time-scale and they are sufficiently bright to be observed at cosmological distance. These lens systems can be used to infer the so-called time-delay distance, $D_{\Delta t}$, which is directly inversely proportional to the Hubble Constant. The method relies on three main ingredients : 

 - a precise and accurate determination of the time delays
 - a model of the mass distribution of the lensing galaxy 
 - an estimate of the mass of all the galaxies along the line of sight that also deviates the light rays, and thus perturb the time delays. 
 
 An example of the light curves of the multiple images of a lensed quasar is presented in \autoref{fig:lcs}.

![Light curves of the lensed quasar RXJ1131-1231 presented in @Millon1:2020 (left panel). The same quasar variations can be seen in image D 92 days after in image A, whereas images A, B, and C arrive approximately at the same time. The right panel shows an Hubble Space Telescope image of RXJ1131-1231 [@Suyu2017].\label{fig:lcs}](RXJ1131.png)
 
 
# Statement of need
 
``PyCS3`` is a python package developed by the [COSMOGRAIL](www.cosmograil.org) collaboration and designed to address the first of the points listed above. ``PyCS3`` allows us to measure time delays in lensed quasars, even in the presence of microlensing, which happens when stars of the lens galaxy are passing in front of the quasar images, also acting as gravitational lenses. This introduces some 'extrinsic' variation that are visible in only one image on top of the quasar 'intrinsic' variation visible in all multiple images. ``PyCS3`` provides a flexible modelling of the extrinsic variation with splines to account for microlensing and recover an unbiased estimate of the time delays. Ignoring this effect might cause large error in the determination of the time delay, that directly propagate to the Hubble Constant. 

The previous version of the package was first presented in @Tewes1:2013 and successfully applied to real data in @Tewes2:2013; @Bonvin:2017; @Bonvin:2018 and @Bonvin:2019. The method was also tested on simulated light curves of the Time-Delay Challenge [@Liao:2016; @Bonvin:2016] and demonstrated that time delays can be measured accurately with a systematic bias of less than 1%, even if the light curves are affected by microlensing.
   
We also recently presented an automated pipeline based ``PyCS3`` to measure time delays in a large sample of lensed quasars [@Millon1:2020; @Millon2:2020]. Such improvements toward automation of the procedure is necessary with the hundreds of new lensed quasars expected to be discovered in the near future. 

# Functionality
The basic functionality of ``PyCS3`` includes a LightCurve class to manipulate photometric monitoring data. It includes methods to import, shift, fit and export light curves. These are located in the `pycs3.gen` subpackage.

``PyCS3`` also contains two time delay estimators, namely the free knot-spline and the regression difference, that are in the `pycs3.spl` and `pycs3.regdiff` subpackages. These two estimators are fundamentally different and allows us to check the robustness of the measured time delays. The subpackage `pycs3.sim` can be used to generate simulated light curves in order to estimate the uncertainties on the time-delay measurements. These simulated curves can then be optimised with either the free knot-spline and the regression difference estimator. 

The ``script`` folder contains a pipeline to automate the measurement of the time-delays. The functions that are used by this pipeline are located in the `pycs3.pipe` subpackage. It automatically explores several set of estimator parameters, generates simulated light curves with the same constraining power than the original data, returns the best fit value and the associated uncertainties before selecting and combining the sets of estimator parameters. It also make use of the subpackage `pycs3.mltd` to display and combine the final time-delay estimates.  The details of the method can be found in @Millon1:2020. 

Finally, the ``tdlmc_test`` folder contains an ensemble of python scripts to apply ``PyCS3`` on the simulated data generated for the Time-Delay Challenge [@Liao:2016]. It provides a benchmark test framework to assess the precision and accuracy of the ``PyCS3``'s curve-shifting algorithms. 

# Acknowledgement

We acknowledge the support of the Swiss National Science Foundation (SNSF) and the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme (COSMICLENS: grant agreement No 787886).

# References