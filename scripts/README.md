#PyCS3 pipeline

This folder contains all the script to process the light curves and measure the time delays. The complete description of the pipeline can be found in [Millon et al. (2020)](https://arxiv.org/abs/2002.05736).

You should first define a working directory that must contain your light curves in the sub-folder `data`. 
Your data should have the naming convention `lensname_dataset.rdb` and must contains one column for the days of the observation and two columns per lens image for the measured magnitudes and its associated uncertainties. A dataset typically corresponds to a monitoring campaign conducted by a single telescope.  

##1. Create data set 

Run the command : 

`python3 1_create_dataset.py lensname dataset --dir='path_to_working_directory'`

to setup the config file. It will create a new folder for your lens and dataset under the subfolder `Simulation/lensname_dataset` from your working directory. If you are not specifying a working directory, it will simply use the current folder `'./'`. The config file is created in the `config` directory under the name `config_lensname_dataset.py` and can be modified. 

##2. Fit spline 
The first step consists in fitting spline to your original data. You can change the mean knotstep and the microlensing model from the config file.

##3. Generate mock light curves
####3a. Fit the parameter of the generative noise model 

####3b. Draw mock lights from the generative noise model 

####3c. Optimise the mock light curves 

####3d. Check statistics 

##4. Marginalise over the estimator parameters

####4a. Get the time delay Estimates 

####4b. Margininalise the spline Estimates 

####4c. Marginalise the regression difference Estimates

####4d. Marginalise spline and regression difference Estimates 

##5. Combine Estimates from different data set. 
