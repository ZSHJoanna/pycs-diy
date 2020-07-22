---
title: 'PyCS3: A Python toolbox for time-delays measurement in lensed quasars'
tags:
  - Python
  - astronomy
authors:
  - name: Martin Millon
    orcid: 0000-0001-7051-497X
    affiliation: 1
  - name: Vivien Bonvin
    affiliation: 1
  - name: Malte Tewes
    affiliation: 1
  - name: Bastian Lengen
    affiliation: 1
affiliations:
  - name: Institute of Physics, Laboratory of Astrophysique, Ecole Polytechnique Fédérale de Lausanne (EPFL)
    index: 1
date: 24 June 2020
bibliography: paper.bib
---

# Summary
Time-delay cosmography is now a competitive technique for measuring the current expansion rate of the Universe, that is, the Hubble Constant [@Wong:2019]. It relies on the strong gravitationnal lensing effect that happens when a massive foreground galaxy deviates the light from a background object, producing multiple mirage images of the same background source. In this configuration, the length of the optical path taken by the photon is slightly different in each multiple image. Thus, the travel time of the photon is also different and the same variations are visible in all multiple images but delayed in time. Lensed quasars or lensed supernovae are ideal targets because they are variable on short time-scale, so the time-delay can be measured and they are sufficiently bright to be observed at cosmological distance. These lens systems can be used to infer the so-called time-delay distance , $D_{\Delta t}$, which is directly inversely proportional to the Hubble Constant. Time-delay cosmography relies on three main ingredients : 

 - a precise and accurate determination of the time delays
 - a model of the mass distribution of the lensing galaxy 
 - an estimate of the mass of all the galaxies along the line of sight that also deviates the light rays, and thus perturb the time delays. 

``PyCS3`` is a python package designed to adress the first of this three points. ``PyCS3`` allows us to measure time delays in lensed quasars in the presence of microlensing, which happens when stars of the lens galaxy are passing in front of the quasar images, also acting as gravitationnal lenses. This introduces some 'extrinsic' variation that are visible in only one image on top of the quasar 'intrinsic' variation visible in all multiple images. ``PyCS3`` provides a flexible modelling of the extrinsic variation with spline to account for microlensing and recover an unbiased estimate of the time delay. 

# Functionnality

``PyCS3`` relies on two time delay estimators, namely the free knot-spline and the regression difference. 
Here talk about automated pipeline, microlensing features and test pipeline by BAstian


# Acknowledgement

# References