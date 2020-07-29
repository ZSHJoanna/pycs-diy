# PyCS3 multiple pipline

This folder contains all the script to process multiple light curves. The complete description of the pipeline can be found in (ADD LINK TO THE PAPER).

You should first define a working directory that must contain your light curves in the sub-folder `data`. 
Your data should have the naming convention `name_double_pairX_ECAM.rdb` with X the number of the light curve and must contain one column for the days of the observation and two columns per lens image for the measured magnitudes and its associated uncertainties. 
You also need to have the true time delay for your light curves in the sub-folder `data/truth` with the naming convention `truth_name.txt` where the second column are the true time delay like in the following example :
        test_double_pair1     -70.89
        test_double_pair2     104.22
        test_double_pair3     56.59

A list of light curves from the TDC1 (ADD LINK) that can be used with this sub-package can be downloaded at (ADD THE LINK)

## 0. Set up
 
Run the command : 

    python3 multiple_0.py name double numberofcurves --dir='path_to_working_directory'

This will set up the directories for this multiple pipeline and generate the multiple config file in the `./config/multiple` directory under the name `config_multiple_name.py` and can be modified before running the next script. Check the README from the main pipeline for more informations on the different parameters.

## 1. Create data set 

Run the command : 

    python3 multiple_1.py name double numberofcurves --dir='path_to_working_directory'

This will run the script `1_create_dataset.py` from the PyCS3 main pipeline for all the curves. The specific config file for each of the curve is created in `config` directory under the name `config_name_double_pairX_EXAM.py`. The single config files are then updated with the parameters from the multiple config file previously mentionned.
The initial guess will be randomly taken around the true time delay from the `./data/truth/truth_name.txt`.
The initial guesses are saved in `./Simulation/multiple/name_double/post_gaussian_guess.txt`. 


## 2. Fit spline

Run the command : 

    python3 multiple_2.py name double numberofcurves --dir='path_to_working_directory'

This will run the script `2_fit_spline.py` from the PyCS3 main pipeline for each curve.

## 3. Launch spline
Modify the `./cluster/multiple_launcher.sh` with the parameters from your cluster. Then make sure the line 18 execute the `start_3all.slurm` and update the number of curves you will run on the line 28. Then run the command while in the `./cluster` folder :

    ./multiple_launcher.sh

This will run the script `3a_generate_tweakml.py`, `3b_draw_copy_mocks.py`, `3c_optimise_copy_mocks.py` and `3d_check_statistics.py` from the PyCS3 main pipeline for each curves.

## 4a. Get the time delay for the spline

Run the command : 

    python3 multiple_4a.py name double numberofcurves --dir='path_to_working_directory'

This will run the script `4a_plot_results.py` from the PyCS3 main pipeline for each curve.

## 3bis. Launch regdiff

Run the command : 

    python3 multiple_update_config.py name double numberofcurves --dir='path_to_working_directory'

Update every config file in order to use the regdiff instead of the spline.
Then modify the `./cluster/multiple_launcher.sh` with the parameters from your cluster. Then make sure the line 18 execute the `start_3c.slurm` and update the number of curves you will run on the line 28. Then run the command while in the `./cluster` folder :

    ./multiple_launcher.sh

This will run the script `3c_optimise_copy_mocks.py` from the PyCS3 main pipeline for each curves, but this time with regdiff.

## 4a. Get the time delay for the regdiff

Run the command : 

    python3 multiple_4a.py name double numberofcurves --dir='path_to_working_directory'

This will run the script `4a_plot_results.py` from the PyCS3 main pipeline for each curve, but this time with regdiff.

## 4b. Margininalise the spline Estimates 

Run the command : 

    python3 multiple_4b.py name double numberofcurves --dir='path_to_working_directory'

This will run the script `4b_marginalise_spline.py` from the PyCS3 main pipeline for each curve for three different `sigmathresh = 0, 0.5, 1000`. You can find informations on the marginalisation in the README of the main pipeline.

## 4b. Margininalise the regdiff Estimates 

Run the command : 

    python3 multiple_4c.py name double numberofcurves --dir='path_to_working_directory'

This will run the script `4c_marginalise_regdiff.py` from the PyCS3 main pipeline for each curve for three different `sigmathresh = 0, 0.5, 1000`. You can find informations on the marginalisation in the README of the main pipeline.

## 4d. Marginalise spline and regression difference Estimates 

Run the command : 

    python3 multiple_4d.py name double numberofcurves --dir='path_to_working_directory'

This will run the script `4d_marginalise_rall.py` from the PyCS3 main pipeline for each curve with both the `sigmathresh = 0.5`. 

## 6. Compute TDC1 metrics
You need to update the `config_multiple_name.py` with the simulation that failed in the `failed_sim` list. You can then choose if you want to display the gold and silver sample and add the pair you want in the silver/golden sample.
You can also decide what plot you want to have displayed. Then, run the command :

    python3 multiple_6.py name double numberofcurves --dir='path_to_working_directory'

This will compute the statistics for each sample, and for each estimate. Summary files are created in the `./Simulation/multiple/name` folder. A sub-folder `figure` is also created and contains all the figure produced.
