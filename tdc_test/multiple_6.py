# coding=utf-8

import argparse as ap
import os
import pickle
import re
import sys

import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats


def compute_SNR(number: int, name: str, name_type: str, work_dir: str):
    """
    return the SNR for a pair of light curve.

    :type	number: int
    :param	number: index of the curve in the data folder
    :type	name: string
    :param	name: name of the set of light curve
    :type	name_type: string
    :param	name_type: name of the data (double or quad)
    :type	work_dir: string
    :param	work_dir: directory you're currently working in
    """
    data_directory = work_dir + 'data/'
    # load light curves in magnitude
    patt = re.compile("[^\t]+")
    [mag_A, magerr_A, mag_B, magerr_B] = [np.array([]), np.array([]), np.array([]), np.array([])]
    with open(data_directory + name + "_" + name_type + "_pair%i_ECAM.rdb" % number, 'r') as f:
        Lines = f.readlines()
        i = 0
        for line in Lines:
            i += 1
            if i <= 2: continue
            tmp = patt.findall(line)
            mag_A = np.append(mag_A, float(tmp[1]))
            magerr_A = np.append(magerr_A, float(tmp[2]))
            mag_B = np.append(mag_B, float(tmp[3]))
            magerr_B = np.append(magerr_B, float(tmp[4][:-1]))

    # update light curves in flux
    ZP = 22.5
    F_A = 10 ** (-1 / 2.5 * (mag_A - ZP))
    F_B = 10 ** (-1 / 2.5 * (mag_B - ZP))
    DF_A = F_A - 10 ** (-1 / 2.5 * (mag_A - ZP + magerr_A))
    DF_B = F_B - 10 ** (-1 / 2.5 * (mag_B - ZP + magerr_B))
    SNR = np.min([np.median(F_A / DF_A), np.median(F_B / DF_B)])
    return (SNR)


def compute_ML(truth: float, number: int, name: str, name_type: str, work_dir: str):
    """
    compute the intensity of the microlensing for a pair light curve.

    :type	truth: float
    :param	truth: true time delay between the 2 curves
    :type	number: int
    :param	number: index of the curve in the data folder
    :type	name: string
    :param	name: name of the set of light curve
    :type	name_type: string
    :param	name_type: name of the data (double or quad)
    :type	work_dir: string
    :param	work_dir: directory you're currently working in
    """

    data_directory = work_dir + 'data/'
    # load light curves in magnitude
    patt = re.compile("[^\t]+")
    [mag_A, mag_B, time_A, time_B] = [np.array([]), np.array([]), np.array([]), np.array([])]
    with open(data_directory + name + "_" + name_type + "_pair%i_ECAM.rdb" % number, 'r') as f:
        Lines = f.readlines()
        i = 0
        for line in Lines:
            i += 1
            if i <= 2: continue
            tmp = patt.findall(line)
            time_A = np.append(time_A, float(tmp[0]))
            mag_A = np.append(mag_A, float(tmp[1]))
            mag_B = np.append(mag_B, float(tmp[3]))
            f.close()
    time_B = time_A - truth
    mag_B = np.interp(time_A, time_B, mag_B)
    mag_B = mag_B - np.min(mag_B)
    mag_A = mag_A - np.min(mag_A)
    ML = mag_A - mag_B
    return (np.max(ML) - np.min(ML))


class Data:
    """
    Class that will contain the result of the simulation for a specific sample and a specific estimate
    The class has a name, a list of the pair of curves in the sample, the true time delay for each pair of curve, the median and errors for each pair of curves.
    """

    def __init__(self, name, lens, truth, median, error_up, error_down):
        """
        Create the sample from a name and 5 list

        :type	self: Data
        :param	self: The Data you're creating
        :type	name: str
        :param	name: The name you're giving to your Data
        :type	lens: 1D numpy array
        :param	lens: The list of the indices of the curves in the sample
        :type	truth: 1D numpy array
        :param	truth: The true time delay for each pair of curves
        :type	median: 1D numpy array
        :param	median: The median from the pickle of a pair of curve, in a specific marginalisation
        :type	error_up: 1D numpy array
        :param	error_up: The error_up from the pickle of a pair of curve, in a specific marginalisation
        :type	error_down: 1D numpy array
        :param	error_down: The error_down from the pickle of a pair of curve, in a specific marginalisation
        """
        self.name = name
        self.lens = np.array(lens)
        self.truth = np.array(truth)
        self.median = np.array(median)
        self.error_up = np.array(error_up)
        self.error_down = np.array(error_down)

    def __str__(self):
        """
        Print the attributes from the class
        """
        return ("name :       " + str(self.name) +
                "\nlens :       " + str(self.lens) +
                "\ntruth :      " + str(self.truth) +
                "\nmedian :     " + str(self.median) +
                "\nerror_up :   " + str(self.error_up) +
                "\nerror_down : " + str(self.error_down))

    def error(self):
        """
        return the error between the true time delay and the estimated time delay
        """
        return self.truth - self.median

    def error_relative(self):
        """
        return the error between the true time delay and the estimated time delay
        """
        return self.error() / (0.5 * self.error_up + 0.5 * self.error_down)

    def precision(self):
        """
        return the precision of the estimate
        """
        return (0.5 * self.error_up + 0.5 * self.error_down) / np.abs(self.median)

    def accuracy(self):
        """
        return the accuracy of the estimate
        """
        return self.error() / self.truth

    def chi2(self):
        """
        return the chi2 of the estimate
        """
        return self.error() ** 2 / (0.5 * self.error_up + 0.5 * self.error_down) ** 2

    def outliers(self):
        """
        return the list of the outliers with chi2 > 10
        """
        return self.lens[self.chi2() > 10]

    def SNR(self, name: str, name_type: str, work_dir: str):
        """
        return the SNR for each of the pair of curves.

        :type	self: Data
        :param	self: The Data for which you're computing the SNR
        :type	name: string
        :param	name: name of the set of light curve
        :type	name_type: string
        :param	name_type: name of the data (double or quad)
        :type	work_dir: string
        :param	work_dir: directory you're currently working in
        """
        SNR = []
        for i in self.lens:
            SNR.append(compute_SNR(i, name, name_type, work_dir))
        return SNR

    def ML(self, name: str, name_type: str, work_dir: str):
        """
        return the intensity of the microlensing for each of the pair of curves.

        :type	self: Data
        :param	self: The Data for which you're computing the microlensing
        :type	name: string
        :param	name: name of the set of light curve
        :type	name_type: string
        :param	name_type: name of the data (double or quad)
        :type	work_dir: string
        :param	work_dir: directory you're currently working in
        """
        ML = []
        j = 0
        for i in self.lens:
            ML.append(compute_ML(self.truth[j], i, name, name_type, work_dir))
            j += 1
        return ML


def summed_stats(data: Data, number_pair: int, print_txt: bool = True):
    """
    return the metrics [f, precision, accuracy, chi2, X] from the TDC1 for a specific Data.
    :type	data: Data
    :param	data: Data for which you want to compute the metrics from the tdc1
    :type	number_pair: int
    :param	number_pair: the total number of pair of curves
    :type	print_txt: bool
    :param	print_txt: if you want to display the metrics while computing it
    """
    f = len(data.lens) / number_pair
    chi2 = np.sum(data.chi2()) / len(data.lens)
    error_chi2 = np.std(data.chi2()) / np.sqrt(len(data.lens))
    precision = np.sum(data.precision()) / len(data.lens)
    error_precision = np.std(data.precision()) / np.sqrt(len(data.lens))
    accuracy = np.sum(data.accuracy()) / len(data.lens)
    error_accuracy = np.std(data.accuracy()) / np.sqrt(len(data.lens))
    X = np.count_nonzero(data.chi2() < 10) / len(data.lens)
    print(np.count_nonzero(data.chi2() < 10))
    print(len(data.lens))
    if print_txt == True:
        print("name \t\t", data.name)
        print("f \t\t", f)
        print("precision \t", precision)
        print("accuracy \t", accuracy)
        print("chi2 \t\t", chi2)
        print("X \t\t", X, "\n")
    return [f, precision, accuracy, chi2, X], [error_precision, error_accuracy, error_chi2]


def create_Data(data_name: str, path: str, full_truth: np.array, sample: np.array, name: str, name_type: str,
                work_dir: str = './'):
    """
    return a new Data for a specific marginalisation (pkl) and sample.

    :type	data_name: string
    :param	data_name: name you're giving to the Data
    :type	path: string
    :param	path: path to the pickle you're using
    :type	full_truth: 1D numpy array
    :param	full_truth: list of the true time delay, for all the pair of curves, not only thoses in the sample
    :type	sample: 1D numpy array
    :param	sample: list of the indices for the pair of light curves you want in the Data
    :type	name: string
    :param	name: name of the set of light curve
    :type	name_type: string
    :param	name_type: name of the data (double or quad)
    :type	work_dir: string
    :param	work_dir: directory you're currently working in
    """
    Simulation_directory = os.path.join(work_dir, "Simulation")
    lens_name = []
    median = []
    error_up = []
    error_down = []
    for i in sample:
        lens_name.append(name + '_' + name_type + '_' + 'pair%i' % i)
        tmp = pickle.load(open(os.path.join(Simulation_directory, lens_name[-1] + "_ECAM" + path), 'rb'))
        median.append(tmp.medians[0])
        error_up.append(tmp.errors_up[0])
        error_down.append(tmp.errors_down[0])
    return Data(data_name, sample, full_truth[sample - 1], median, error_up, error_down)


'''
def sort_SNR(sample: np.array, name: str, name_type: str, work_dir: str='./'):
	SNR = np.array([])
	for i in sample:
		SNR = np.append(SNR, compute_SNR(i, name, name_type, work_dir))
	median_SNR = np.median(SNR)
	low_SNR=sample[SNR<=median_SNR]
	high_SNR=sample[SNR>median_SNR]
	return [low_SNR, high_SNR]
'''


def load_Datalist(sample_name: str, full_truth: np.array, samplelist: np.array, name: str, name_type: str,
                  number_pair: int, work_dir: str = './'):
    """
    return a list of Data for each of the estimate you're using

    :type	sample_name: string
    :param	sample_name: name of the sample (FS, SS or GS)
    :type	full_truth: 1D numpy array
    :param	full_truth: list of the true time delay, for all the pair of curves, not only thoses in the sample
    :type	samplelist: 1D numpy array of sample (1D numpy array)
    :param	samplelist: list of the sample
    :type	name: string
    :param	name: name of the set of light curve
    :type	name_type: string
    :param	name_type: name of the data (double or quad)
    :type	number_pair: int
    :param	number_pair: the total number of pair of curves
    :type	work_dir: string
    :param	work_dir: directory you're currently working in
    """
    # path of the different marginalisation
    Simulation_directory = os.path.join(work_dir, "Simulation")
    Simulation_multiple_directory = os.path.join(Simulation_directory, "multiple", name + "_double")
    path_spl_sigma_0 = "/marginalisation_spline/marginalisation_spline_sigma_0.00_combined.pkl"
    path_spl_sigma_05 = "/marginalisation_spline/marginalisation_spline_sigma_0.50_combined.pkl"
    path_spl_sigma_1000 = "/marginalisation_spline/marginalisation_spline_sigma_1000.00_combined.pkl"
    path_regdiff_set1 = "/marginalisation_regdiff/marginalisation_regdiff_Set 1_PS_combined.pkl"
    path_regdiff_set2 = "/marginalisation_regdiff/marginalisation_regdiff_Set 2_PS_combined.pkl"
    path_regdiff_set3 = "/marginalisation_regdiff/marginalisation_regdiff_Set 3_PS_combined.pkl"
    path_regdiff_set4 = "/marginalisation_regdiff/marginalisation_regdiff_Set 4_PS_combined.pkl"
    path_regdiff_sigma_0 = "/marginalisation_regdiff/marginalisation_regdiff_sigma_0.00_combined.pkl"
    path_regdiff_sigma_05 = "/marginalisation_regdiff/marginalisation_regdiff_sigma_0.50_combined.pkl"
    path_regdiff_sigma_1000 = "/marginalisation_regdiff/marginalisation_regdiff_sigma_1000.00_combined.pkl"
    path_both_baseline = "/marginalisation_final/marginalisation_final_sigma_0.00_combined.pkl"
    # create the data
    spl_sigma_0 = create_Data("spline marg ($\\tau_{thresh}$ = 0)", path_spl_sigma_0, full_truth, samplelist[0], name,
                              name_type, work_dir=work_dir)
    spl_sigma_05 = create_Data("spline marg ($\\tau_{thresh}$ = 0.5)", path_spl_sigma_05, full_truth, samplelist[1],
                               name, name_type, work_dir=work_dir)
    spl_sigma_1000 = create_Data("spline marg ($\\tau_{thresh}$ = 1000)", path_spl_sigma_1000, full_truth,
                                 samplelist[2], name, name_type, work_dir=work_dir)
    regdiff_set1 = create_Data("regdiff (set 1)", path_regdiff_set1, full_truth, samplelist[3], name, name_type,
                               work_dir=work_dir)
    regdiff_set2 = create_Data("regdiff (set 2)", path_regdiff_set2, full_truth, samplelist[4], name, name_type,
                               work_dir=work_dir)
    regdiff_set3 = create_Data("regdiff (set 3)", path_regdiff_set3, full_truth, samplelist[5], name, name_type,
                               work_dir=work_dir)
    regdiff_set4 = create_Data("regdiff (set 4)", path_regdiff_set4, full_truth, samplelist[6], name, name_type,
                               work_dir=work_dir)
    regdiff_sigma_0 = create_Data("regdiff marg ($\\tau_{thresh}$ = 0)", path_regdiff_sigma_0, full_truth,
                                  samplelist[7], name, name_type, work_dir=work_dir)
    regdiff_sigma_05 = create_Data("regdiff marg ($\\tau_{thresh}$ = 0.5)", path_regdiff_sigma_05, full_truth,
                                   samplelist[8], name, name_type, work_dir=work_dir)
    regdiff_sigma_1000 = create_Data("regdiff marg ($\\tau_{thresh}$ = 1000)", path_regdiff_sigma_1000, full_truth,
                                     samplelist[9], name, name_type, work_dir=work_dir)
    both_baseline = create_Data("marg spline-regdiff (baseline)", path_both_baseline, full_truth, samplelist[10], name,
                                name_type, work_dir=work_dir)

    '''
    # split the sample in 2 : high SNR and low SNR
    [low_SNR, high_SNR] = sort_SNR(sample, name, name_type, work_dir)
    high_SNR_spl_sigma_05 = create_Data("high SNR spline marg (sigma = 0.5)", path_spl_sigma_05, full_truth, high_SNR, name, name_type)
    high_SNR_regdiff_sigma_05 = create_Data("high SNR regdiff marg (sigma = 0.5)", path_regdiff_sigma_05, full_truth, high_SNR, name, name_type)
    high_SNR_both_baseline = create_Data("high SNR marg spline-regdiff (baseline)", path_both_baseline, full_truth, high_SNR, name, name_type)
    low_SNR_spl_sigma_05 = create_Data("low SNR spline marg (sigma = 0.5)", path_spl_sigma_05, full_truth, low_SNR, name, name_type)
    low_SNR_regdiff_sigma_05 = create_Data("low SNR regdiff marg (sigma = 0.5)", path_regdiff_sigma_05, full_truth, low_SNR, name, name_type)
    low_SNR_both_baseline = create_Data("low SNR marg spline-regdiff (baseline)", path_both_baseline, full_truth, low_SNR, name, name_type)
    '''

    # write the statistics of the data in a .csv file
    Data_list = [spl_sigma_0, spl_sigma_05, spl_sigma_1000, regdiff_set1, regdiff_set2,
                 regdiff_set3, regdiff_set4, regdiff_sigma_0, regdiff_sigma_05, regdiff_sigma_1000,
                 both_baseline]  # , high_SNR_spl_sigma_05, high_SNR_regdiff_sigma_05, high_SNR_both_baseline,
    # low_SNR_spl_sigma_05, low_SNR_regdiff_sigma_05, low_SNR_both_baseline]
    tmp = 'test, f, precision [%], err, accuracy[%],err, chi2,err, X'
    for data in Data_list:
        stats, error = summed_stats(data, number_pair)
        tmp += "\n" + data.name + "," + str(stats[0]) + "," + str(stats[1] * 100) + "," + str(
            error[0] * 100) + "," + str(stats[2] * 100) + "," + str(error[1] * 100) + "," + str(stats[3]) + "," + str(
            error[2]) + "," + str(stats[4])
    with open(os.path.join(Simulation_multiple_directory,'%s_statistics.csv' % sample_name), 'w') as f:
        f.write(tmp)
        f.close()
    return Data_list


def create_silver_sample(full_truth: np.array, sample: np.array, data: Data, chi2: bool):
    """
    return the silver sample for a specific sample and Data.
    The cuts are : true time delay < 100 days and precision < 40%

    :type	full_truth: 1D numpy array
    :param	full_truth: list of the true time delay, for all the pair of curves, not only thoses in the sample
    :type	sample: 1D numpy array of sample
    :param	sample: full sample you're using to compute the silver sample
    :type	data: Data
    :param	data: Data you're using to compute the precision and chi2
    :type	chi2: bool
    :param	chi2: if you want to have the cut at chi2<3 in top of the 2 other cuts
    """
    truth = full_truth[sample - 1]
    if chi2 == True:
        to_reject = np.append(np.append(sample[np.abs(truth) >= 100], sample[data.chi2() >= 3]),
                              sample[data.precision() >= 0.40])
    else:
        to_reject = np.append(sample[np.abs(truth) >= 100], sample[data.precision() >= 0.40])
    return np.setdiff1d(sample, to_reject)


def create_multiple_silver_sample(full_truth: np.array, sample: np.array, datalist: np.array):
    """
    return a list of silver sample corresponding to a Datalist
    For the splines, the cuts are : true time delay < 100 days and precision < 40%
    For the regdiff, the cuts are : true time delay < 100 days, precision < 40% and chi2<3

    :type	full_truth: 1D numpy array
    :param	full_truth: list of the true time delay, for all the pair of curves, not only thoses in the sample
    :type	sample: 1D numpy array of sample
    :param	sample: full sample you're using to compute the silver sample
    :type	datalist: 1D array of Data
    :param	datalist: list of data you're using to compute the precision and chi2
    """
    SSlist = []
    i = 0
    for data in datalist:
        if i in [0, 1, 2]:
            SSlist.append(create_silver_sample(full_truth, sample, data, False))
        else:
            SSlist.append(create_silver_sample(full_truth, sample, data, True))
        i += 1

    SSlist[0], SSlist[2] = SSlist[1], SSlist[1]  # set the same cut for all the marg_spline
    SSlist[7], SSlist[9] = SSlist[8], SSlist[8]  # set the same cut for all the marg regdiff
    SSlist[10] = np.intersect1d(SSlist[1], SSlist[8])
    return SSlist


def plot_delay(sample_name: str, data: Data, figure_directory: str):
    """
    plot the time delay estimate and the true time delay for each curves in a Data

    :type	sample_name: string
    :param	sample_name: name of the sample (FS, SS, GS)
    :type	data: Data
    :param	data: Data you're ploting the estimate from
    :type	figure_directory: string
    :param	figure_directory: path to the directory where the plot will be saved
    """
    fig, ax = plt.subplots()
    ax.errorbar(data.lens, data.median, 0.5 * (data.error_up + data.error_down), marker='.', linestyle='none',
                label='Simulated time delay')
    ax.plot(data.lens, data.truth, '.r', label='True time delay')
    ax.set_xlabel('Simulation [ ]')
    ax.set_ylabel('Time delay [d]')
    ax.legend()
    fig.savefig(figure_directory + "%s_%s_delay.png" % (sample_name, data.name))
    plt.close('all')


def plot_error_rel(sample_name: str, data: Data, figure_directory: str):
    """
    plot the relative errorfor each curves in a Data

    :type	sample_name: string
    :param	sample_name: name of the sample (FS, SS, GS)
    :type	data: Data
    :param	data: Data you're ploting the estimate from
    :type	figure_directory: string
    :param	figure_directory: path to the directory where the plot will be saved
    """
    fig, ax = plt.subplots()
    error_rel = data.error() / (0.5 * data.error_up + 0.5 * data.error_down)
    ax.hist(error_rel, bins=50, label='Simulations')
    ax.plot([-1, -1], ax.get_ylim(), 'k--', linewidth=2, label='1-$\sigma$ threshold')
    ax.autoscale(False)
    ax.plot([1, 1], ax.get_ylim(), 'k--', linewidth=2)
    ax.set_xlabel('Relative Error [$\sigma$]')
    ax.set_ylabel('Count [ ]')
    ax.legend()
    fig.savefig(figure_directory + "%s_%s_Error_rel_hist.png" % (sample_name, data.name))
    plt.close('all')


def plot_precision(sample_name: str, data: Data, figure_directory: str):
    """
    plot the histogram precision for each curves in a Data

    :type	sample_name: string
    :param	sample_name: name of the sample (FS, SS, GS)
    :type	data: Data
    :param	data: Data you're ploting the estimate from
    :type	figure_directory: string
    :param	figure_directory: path to the directory where the plot will be saved
    """
    fig, ax = plt.subplots()
    precision = data.precision() * 100
    mu_prec, std_prec = np.median(precision), np.std(precision)
    ax.hist(precision, bins=50, label='Simulations')
    p16 = np.percentile(precision, 16)
    p84 = np.percentile(precision, 84)
    ax.plot([mu_prec, mu_prec], ax.get_ylim(), 'r', linewidth=2, label='median = %.2f' % mu_prec)
    ax.autoscale(False)
    ax.plot([p16, p16], ax.get_ylim(), '--k', linewidth=2, label='16th percentile = %.2f' % p16)
    ax.plot([p84, p84], ax.get_ylim(), '-.k', linewidth=2, label='84th percentile = %.2f' % p84)
    ax.set_xlabel('Precision [%]')
    ax.set_ylabel('Count [ ]')
    ax.legend()
    fig.savefig(figure_directory + "%s_%s_precision.png" % (sample_name, data.name))
    plt.close('all')


def plot_accuracy(sample_name: str, data: Data, figure_directory: str):
    """
    plot the histogram of the accuracy for each curves in a Data

    :type	sample_name: string
    :param	sample_name: name of the sample (FS, SS, GS)
    :type	data: Data
    :param	data: Data you're ploting the estimate from
    :type	figure_directory: string
    :param	figure_directory: path to the directory where the plot will be saved
    """
    fig, ax = plt.subplots()
    accuracy = data.accuracy() * 100
    ax.hist(accuracy, bins=50, label='Simulations')
    mu_acc, std_acc = np.median(accuracy), np.std(accuracy)
    p16 = np.percentile(accuracy, 16)
    p84 = np.percentile(accuracy, 84)
    ax.plot([mu_acc, mu_acc], ax.get_ylim(), 'r', linewidth=2, label='median = %.2f' % mu_acc)
    ax.autoscale(False)
    ax.plot([p16, p16], ax.get_ylim(), '--k', linewidth=2, label='16th percentile = %.2f' % p16)
    ax.plot([p84, p84], ax.get_ylim(), '-.k', linewidth=2, label='84th percentile = %.2f' % p84)
    ax.set_xlabel('Accuracy [%]')
    ax.set_ylabel('Count [ ]')
    ax.legend()
    fig.savefig(figure_directory + "%s_%s_accuracy.png" % (sample_name, data.name))
    plt.close('all')


def plot_chi2(sample_name: str, data: Data, figure_directory: str):
    """
    plot the histogram of the chi2 for each curves in a Data

    :type	sample_name: string
    :param	sample_name: name of the sample (FS, SS, GS)
    :type	data: Data
    :param	data: Data you're ploting the estimate from
    :type	figure_directory: string
    :param	figure_directory: path to the directory where the plot will be saved
    """
    fig, ax = plt.subplots()
    chi2 = data.chi2()
    ax.hist(chi2, bins=50, label='Simulations')
    mu_chi2, std_chi2 = np.median(chi2), np.std(chi2)
    p16 = np.percentile(chi2, 16)
    p84 = np.percentile(chi2, 84)
    ax.plot([mu_chi2, mu_chi2], ax.get_ylim(), 'r', linewidth=2, label='median = %.2f' % mu_chi2)
    ax.autoscale(False)
    ax.plot([p16, p16], ax.get_ylim(), '--k', linewidth=2, label='16th percentile = %.2f' % p16)
    ax.plot([p84, p84], ax.get_ylim(), '-.k', linewidth=2, label='84th percentile = %.2f' % p84)
    ax.set_xlabel('$\chi^2$ [ ]')
    ax.set_ylabel('Count [ ]')
    ax.legend()
    fig.savefig(figure_directory + "%s_%s_chi2.png" % (sample_name, data.name))
    plt.close('all')


def plot_SNR(sample_name: str, data: Data, figure_directory: str, name: str, name_type: str, work_dir: str):
    """
    plot the SNR vs the precision, accuracy and chi2 for each curves in a Data.
    Also compute the Pearson's coefficient to estimate the correlation between the SNR and the metrics.

    :type	sample_name: string
    :param	sample_name: name of the sample (FS, SS, GS)
    :type	data: Data
    :param	data: Data you're ploting the estimate from
    :type	figure_directory: string
    :param	figure_directory: path to the directory where the plot will be saved
    :type	name: string
    :param	name: name of the set of light curve
    :type	name_type: string
    :param	name_type: name of the data (double or quad)
    :type	work_dir: string
    :param	work_dir: directory you're currently working in
    """
    SNR = data.SNR(name, name_type, work_dir)
    precision = data.precision()
    accuracy = data.accuracy()
    chi2 = data.chi2()

    fig1, ax1 = plt.subplots()
    ax1.plot(SNR, precision * 100, marker='.', linewidth=0,
             label="Pearson's coef = %f, p-value = %f" % scipy.stats.pearsonr(SNR, precision))
    ax1.legend()
    ax1.set_ylabel('Precision [%]')
    ax1.set_xlabel('SNR [ ]')
    fig1.savefig(figure_directory + "%s_%s_SNR_vs_precision.png" % (sample_name, data.name))

    fig2, ax2 = plt.subplots()
    ax2.plot(SNR, accuracy * 100, marker='.', linewidth=0,
             label="Pearson's coef = %f, p-value = %f" % scipy.stats.pearsonr(SNR, accuracy))
    ax2.legend()
    ax2.set_ylabel('Accuracy [%]')
    ax2.set_xlabel('SNR [ ]')
    fig2.savefig(figure_directory + "%s_%s_SNR_vs_accuracy.png" % (sample_name, data.name))

    fig3, ax3 = plt.subplots()
    ax3.plot(SNR, chi2, marker='.', linewidth=0,
             label="Pearson's coef = %f, p-value = %f" % scipy.stats.pearsonr(SNR, chi2))
    ax3.legend()
    ax3.set_ylabel('$\chi^2$ [ ]')
    ax3.set_xlabel('SNR [ ]')
    fig3.savefig(figure_directory + "%s_%s_SNR_vs_chi2.png" % (sample_name, data.name))

    plt.close("all")


def plot_ML(sample_name: str, data: Data, figure_directory: str, name: str, name_type: str, work_dir: str):
    """
    plot the microlensing vs the precision, accuracy and chi2 for each curves in a Data.
    Also compute the Pearson's coefficient to estimate the correlation between the microlensing and the metrics.

    :type	sample_name: string
    :param	sample_name: name of the sample (FS, SS, GS)
    :type	data: Data
    :param	data: Data you're ploting the estimate from
    :type	figure_directory: string
    :param	figure_directory: path to the directory where the plot will be saved
    :type	name: string
    :param	name: name of the set of light curve
    :type	name_type: string
    :param	name_type: name of the data (double or quad)
    :type	work_dir: string
    :param	work_dir: directory you're currently working in
    """
    ML = data.ML(name, name_type, work_dir)
    precision = data.precision()
    accuracy = data.accuracy()
    chi2 = data.chi2()

    fig1, ax1 = plt.subplots()
    ax1.plot(ML, precision * 100, marker='.', linewidth=0,
             label="Pearson's coef = %f, p-value = %f" % scipy.stats.pearsonr(ML, precision))
    ax1.legend()
    ax1.set_ylabel('Precision [%]')
    ax1.set_xlabel('ML [mag]')
    fig1.savefig(figure_directory + "%s_%s_ML_vs_precision.png" % (sample_name, data.name))

    fig2, ax2 = plt.subplots()
    ax2.plot(ML, accuracy * 100, marker='.', linewidth=0,
             label="Pearson's coef = %f, p-value = %f" % scipy.stats.pearsonr(ML, accuracy))
    ax2.legend()
    ax2.set_ylabel('Accuracy [%]')
    ax2.set_xlabel('ML [mag]')
    fig2.savefig(figure_directory + "%s_%s_ML_vs_accuracy.png" % (sample_name, data.name))

    fig3, ax3 = plt.subplots()
    ax3.plot(ML, chi2, marker='.', linewidth=0,
             label="Pearson's coef = %f, p-value = %f" % scipy.stats.pearsonr(ML, chi2))
    ax3.legend()
    ax3.set_ylabel('$\chi^2$ [ ]')
    ax3.set_xlabel('ML [mag]')
    fig3.savefig(figure_directory + "%s_%s_ML_vs_chi2.png" % (sample_name, data.name))

    plt.close('all')


def getColor(c, N, idx):
    """
    Get the color from a color map

    :type	c: cmap
    :param	c: color map
    :type	N: integer
    :param	N: total number of color you want
    :type	idx: int
    :param	idx: index of the color you want in the range(0, N-1)

    """
    cmap = mpl.cm.get_cmap(c)
    norm = mpl.colors.Normalize(vmin=0.0, vmax=N - 1)
    return cmap(norm(idx))


def plot_tdc1(Datalist: np.array, sample_name: str, figure_directory: str, compare_tdc1: bool, number_pair: int,
              ploterror_bar=True, showregdiffset=False):
    """
    plot metrics for each sample and data in a summary plot inspired by the TDC1.
    There is also the possiblity to compare those results to the submissions from the TDC1.

    :type	Datalist: 1D array of Data
    :param	Datalist: Datalist for each of the sample ([FS_datalist, SS_datalist, GS_datalist])
    :type	sample_name: string
    :param	sample_name: Name of the sample (FS, SS or GS)
    :type	figure_directory: string
    :param	figure_directory: path to the directory where the plot will be saved
    :type	compare_tdc1: bool
    :param	compare_tdc1: if you want to display the submissions from the TDC1 on the plot
    :type	number_pair: int
    :param	number_pair: the total number of pair of curves
    :type   ploterror_bar: bool
    :param  ploterror_bar: if you want to have error bars on the metrics
    """
    # results from the tdc1
    if (compare_tdc1 == True):
        total_f = [0.22, 0.18, 0.02, 0.34, 0.34, 0.30, 0.30, 0.30, 0.28]
        total_chi2 = [0.59, 0.78, 0.51, 1.165, 0.458, 0.099, 0.813, 0.494, 1.28]
        total_chi2_error = np.zeros(len(total_chi2))
        total_P = [0.097, 0.06, 0.155, 0.036, 0.059, 0.247, 0.068, 0.042, 0.051]
        total_P_error = np.zeros(len(total_P))
        total_A = [0.000, -0.003, 0.037, 0.002, -0.020, -0.030, -0.004, -0.001, 0.007]
        total_A_error = np.zeros(len(total_A))
        total_X = [0.66, 0.96, 0.95, 0.98, 1.0, 1.0, 1.0, 1.0, 0.95]
        author = ['Rumbaugh', 'Hojjati', 'Kumar', 'Jackson', 'Shafieloo', 'pyCS-D3CS', 'pyCS-SDI', 'pyCS-SPL', 'JPL']
        markers = ['.', '.', '.', '.', '.', '.', '.', '.', '.']
    else:
        total_f, total_chi2, total_P, total_A, total_X, total_P_error, total_A_error, total_chi2_error, author, color, markers = [], [], [], [], [], [], [], [], [], [], []
    total_f, total_chi2, total_P, total_A, total_X, total_A_error, total_P_error, total_chi2_error = np.array(
        total_f), np.array(total_chi2), np.array(total_P), np.array(total_A), np.array(total_X), np.array(
        total_A_error), np.array(total_P_error), np.array(total_chi2_error)

    # load our results
    count = -1
    for Data in Datalist:
        count += 1
        if count in [3, 4, 5, 6] and not showregdiffset : continue
        [f, precision, accuracy, chi2, X], [error_precision, error_accuracy, error_chi2] = summed_stats(Data,
                                                                                                        number_pair,
                                                                                                        True)
        total_f = np.append(total_f, f)
        total_P = np.append(total_P, precision)
        total_P_error = np.append(total_P_error, error_precision)
        total_A = np.append(total_A, accuracy)
        total_A_error = np.append(total_A_error, error_accuracy)
        total_chi2 = np.append(total_chi2, chi2)
        total_chi2_error = np.append(total_chi2_error, error_chi2)
        total_X = np.append(total_X, X)
        author.append(str(sample_name + " " + Data.name))
    f, chi2, P, A, X = total_f, total_chi2, total_P, total_A, total_X
    markers = markers + ['v', 'v', 'v', '^', '^', '^', 'x']
    colors = ["royalblue", "crimson", "seagreen", 'brown', 'black', 'violet', 'paleturquoise', 'palevioletred', 'olive',
              'indianred', 'cyan', 'darkgoldenrod', 'chocolate', 'indigo', 'steelblue', 'gold']

    # Do the plot
    N = f.size
    c = "hsv"
    bx1 = plt.subplot(4, 4, 1)
    rect = patches.Rectangle((0.3, 0), 1, 0.03, facecolor='Gainsboro', zorder=0)
    rect.set_alpha(0.3)
    bx1.add_patch(rect)
    rect = patches.Rectangle((0.5, 0), 1, 0.03, facecolor='Grey', zorder=0)
    rect.set_alpha(0.3)
    bx1.add_patch(rect)
    for i in range(f.size):
        bx1.scatter(f[i], P[i], color=colors[i], marker=markers[i], s=16)
        if ploterror_bar:bx1.errorbar(f[i], P[i], yerr=total_P_error[i], ecolor=colors[i], elinewidth=0.5, zorder=1 )
    plt.setp(bx1.get_xticklabels(), visible=False)
    bx1.set_ylabel('$P$')

    bx2 = plt.subplot(4, 4, 5, sharex=bx1)
    rect = patches.Rectangle((0.3, -0.03), 1, 0.06, facecolor='Gainsboro', zorder=0)
    rect.set_alpha(0.3)
    bx2.add_patch(rect)
    rect = patches.Rectangle((0.5, -0.03), 1, 0.06, facecolor='Grey', zorder=0)
    rect.set_alpha(0.3)
    bx2.add_patch(rect)
    for i in range(f.size):
        bx2.scatter(f[i], A[i], color=colors[i], marker=markers[i], s=16)
        if ploterror_bar: bx2.errorbar(f[i], A[i], yerr=total_A_error[i], ecolor=colors[i], elinewidth=0.5, zorder=1 )
    plt.setp(bx2.get_xticklabels(), visible=False)
    bx2.set_ylabel('A')

    bx3 = plt.subplot(4, 4, 6, sharey=bx2)
    rect = patches.Rectangle((0, -0.03), 0.03, 0.06, facecolor='Grey', zorder=0)
    rect.set_alpha(0.3)
    bx3.add_patch(rect)
    for i in range(P.size):
        bx3.scatter(P[i], A[i], color=colors[i], marker=markers[i], s=16)
        if ploterror_bar: bx3.errorbar(P[i], A[i], xerr=total_P_error[i], yerr=total_A_error[i], ecolor=colors[i],
                                       elinewidth=0.5, zorder=1 )
    plt.setp(bx3.get_xticklabels(), visible=False)
    plt.setp(bx3.get_yticklabels(), visible=False)

    bx4 = plt.subplot(4, 4, 9, sharex=bx1)
    rect = patches.Rectangle((0.3, 0), 1, 1.5, facecolor='Gainsboro', zorder=0)
    rect.set_alpha(0.3)
    bx4.add_patch(rect)
    rect = patches.Rectangle((0.5, 0), 1, 1.5, facecolor='Grey', zorder=0)
    rect.set_alpha(0.3)
    bx4.add_patch(rect)
    for i in range(f.size):
        bx4.scatter(f[i], chi2[i], color=colors[i], marker=markers[i], s=16)
        if ploterror_bar: bx4.errorbar(f[i], chi2[i], yerr=total_chi2_error[i], ecolor=colors[i],
                                       elinewidth=0.5, zorder=1 )
    plt.setp(bx4.get_xticklabels(), visible=False)
    bx4.set_ylabel('$\chi^2$')

    bx5 = plt.subplot(4, 4, 10, sharex=bx3, sharey=bx4)
    rect = patches.Rectangle((0, 0), 0.03, 1.5, facecolor='Grey', zorder=0)
    rect.set_alpha(0.3)
    bx5.add_patch(rect)
    for i in range(P.size):
        bx5.scatter(P[i], chi2[i], color=colors[i], marker=markers[i], s=16)
        if ploterror_bar: bx5.errorbar(P[i], chi2[i], xerr=total_P_error[i], yerr=total_chi2_error[i], ecolor=colors[i],
                                       elinewidth=0.5, zorder=1 )
    plt.setp(bx5.get_xticklabels(), visible=False)
    plt.setp(bx5.get_yticklabels(), visible=False)

    bx6 = plt.subplot(4, 4, 11, sharey=bx4)
    rect = patches.Rectangle((-0.03, 0), 0.06, 1.5, facecolor='Grey', zorder=0)
    rect.set_alpha(0.3)
    bx6.add_patch(rect)
    for i in range(A.size):
        bx6.scatter(A[i], chi2[i], color=colors[i], marker=markers[i], s=16)
        if ploterror_bar: bx6.errorbar(A[i], chi2[i], xerr=total_A_error[i], yerr=total_chi2_error[i], ecolor=colors[i],
                                       elinewidth=0.5, zorder=1 )
    plt.setp(bx6.get_xticklabels(), visible=False)
    plt.setp(bx6.get_yticklabels(), visible=False)

    bx7 = plt.subplot(4, 4, 13, sharex=bx1)
    for i in range(f.size):
        bx7.scatter(f[i], X[i], color=colors[i], marker=markers[i], s=16)
    bx7.set_xlabel('f')
    bx7.set_ylabel('X')

    bx8 = plt.subplot(4, 4, 14, sharex=bx3, sharey=bx7)
    for i in range(P.size):
        bx8.scatter(P[i], X[i], color=colors[i], marker=markers[i], s=16)
        if ploterror_bar: bx8.errorbar(P[i], X[i], xerr=total_P_error[i], ecolor=colors[i],
                                       elinewidth=0.5, zorder=1 )
    plt.setp(bx8.get_yticklabels(), visible=False)
    bx8.set_xlabel('P')

    bx9 = plt.subplot(4, 4, 15, sharex=bx6, sharey=bx7)
    for i in range(A.size):
        bx9.scatter(A[i], X[i], color=colors[i], marker=markers[i], s=16)
        if ploterror_bar: bx9.errorbar(A[i], X[i], xerr=total_A_error[i], ecolor=colors[i],
                                       elinewidth=0.5, zorder=1 )
    plt.setp(bx9.get_yticklabels(), visible=False)
    bx9.set_xlabel('A')

    bx10 = plt.subplot(4, 4, 16, sharey=bx7)
    for i in range(chi2.size):
        bx10.scatter(chi2[i], X[i], color=colors[i], marker=markers[i], s=16)
        if ploterror_bar: bx9.errorbar(chi2[i], X[i], xerr=total_chi2_error[i], ecolor=colors[i],
                                       elinewidth=0.5, zorder=1 )
    plt.setp(bx10.get_yticklabels(), visible=False)
    bx10.set_xlabel('$\chi^2$')

    # set ticks and limit
    bx1.set_xlim([-0.02, 1.02])
    bx1.set_xticks([0.2, 0.4, 0.6, 0.8])
    bx1.set_ylim([0, 0.25])
    bx1.set_yticks([0, 0.10, 0.20])

    bx2.set_ylim([-0.05, 0.05])
    bx2.set_yticks([-0.03, -0.01, 0.01, 0.03])

    bx3.set_xlim([0, 0.25])
    bx3.set_xticks([0, 0.10, 0.20])

    bx4.set_ylim([0, 2])
    bx4.set_yticks([0.5, 1, 1.5])

    bx6.set_xlim([-0.05, 0.05])
    bx6.set_xticks([-0.03, 0, 0.03])

    bx7.set_ylim([0.90, 1.005])
    bx7.set_yticks([0.92, 0.94, 0.96, 0.98])

    bx10.set_xlim([0, 2])
    bx10.set_xticks([0.5, 1, 1.5])

    if sample_name == "FS":
        plt.suptitle('Metrics for the Full Sample')
    elif sample_name == "SS":
        plt.suptitle('Metrics for the Silver Sample')
    elif sample_name == "GS":
        plt.suptitle('Metrics for the Golden Sample')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.legend(author, ncol=2, fontsize='xx-small', loc=1, bbox_to_anchor=[1.51, 4])
    plt.savefig(figure_directory + '%s_Final_Plot.png' % sample_name, dpi=400)

    plt.close("all")


def main(name, name_type, number_pair=1, work_dir='./'):
    # load config file
    sys.path.append(os.path.join(work_dir, "config/multiple"))
    config_multiple_name = __import__("config_multiple_%s" % name)
    print(config_multiple_name.display_gold)

    config_directory = os.path.join(work_dir, "config")
    multiple_config_directory = os.path.join(config_directory, "multiple")
    data_directory = os.path.join(".", "data")
    truth_directory = os.path.join(data_directory, "truth")
    Simulation_directory = os.path.join(work_dir, "Simulation")
    dataname = 'ECAM'
    Simulation_multiple_directory = os.path.join(Simulation_directory, "multiple", name + "_double")
    figure_directory = os.path.join(Simulation_multiple_directory, "figure")
    figure_FS_directory = os.path.join(figure_directory, "FS")

    if not os.path.exists(os.path.join(Simulation_directory, "multiple")):
        print("I will create the multiple simulation directory for you ! ")
        os.mkdir(os.path.join(Simulation_directory, "multiple"))
    if not os.path.exists(Simulation_multiple_directory):
        os.mkdir(Simulation_multiple_directory)
    if not os.path.exists(figure_directory):
        os.mkdir(figure_directory)
    if not os.path.exists(figure_FS_directory):
        os.mkdir(figure_FS_directory)
    if (config_multiple_name.display_silver == True):
        figure_SS_directory = figure_directory + "SS/"
        if not os.path.exists(figure_SS_directory):
            os.mkdir(figure_SS_directory)
    if (config_multiple_name.display_gold == True):
        figure_GS_directory = figure_directory + "GS/"
        if not os.path.exists(figure_GS_directory):
            os.mkdir(figure_GS_directory)

    ### Get the true time-delays and update it with the sign from the config file
    truth = []
    with open(os.path.join(truth_directory, 'truth_' + name + '.txt'), 'r') as f:
        Lines = f.readlines()
        i = 0
        for line in Lines:
            i += 1
            truth.append(float(line.partition(' ')[2]))
            if (i == number_pair): break
    if (config_multiple_name.sign == -1):
        truth = np.array([-x for x in truth])
    elif (config_multiple_name.sign == 1):
        truth = np.array(truth)
    else:
        print('ERROR : Make sure the sign of the config_multiple file is +1 or -1')
        sys.exit()

    ### load the time-delay estimate for each pair sample/sigma_threshold/noise
    # Order of the data in datalist :
    # 0-2: spl sigma =0, 0.5, 1000      / 3-6: regdiff set 1-4
    # 7-9: regdiff sigma = 0, 0.5, 1000 / 10 : spl-regdiff
    sample = range(1, number_pair + 1)
    # Full sample
    FS = np.setdiff1d(sample, config_multiple_name.failed_sim)
    FSlist = [FS, FS, FS, FS, FS, FS, FS, FS, FS, FS, FS]
    print('####'.center(80, '#'))
    print(" Loading the data for the full sample. ".center(80, '#'))
    print('####'.center(80, '#'))
    FS_Datalist = load_Datalist('FS', truth, FSlist, name, name_type, number_pair, work_dir=work_dir)
    list_Datalist = [FS_Datalist]
    if (config_multiple_name.display_delay == True):
        for data in FS_Datalist: plot_delay("FS", data, figure_FS_directory)
    if (config_multiple_name.display_relative_err == True):
        for data in FS_Datalist: plot_error_rel("FS", data, figure_FS_directory)
    if (config_multiple_name.display_precision == True):
        for data in FS_Datalist: plot_precision("FS", data, figure_FS_directory)
    if (config_multiple_name.display_accuracy == True):
        for data in FS_Datalist: plot_accuracy("FS", data, figure_FS_directory)
    if (config_multiple_name.display_chi2 == True):
        for data in FS_Datalist: plot_chi2("FS", data, figure_FS_directory)
    if (config_multiple_name.display_SNR == True):
        for data in FS_Datalist: plot_SNR("FS", data, figure_FS_directory, name, name_type, work_dir)
    if (config_multiple_name.display_ML == True):
        for data in FS_Datalist: plot_ML("FS", data, figure_FS_directory, name, name_type, work_dir)

    # Silver sample
    if (config_multiple_name.display_silver == True):
        SSlist = create_multiple_silver_sample(truth, FS, FS_Datalist)
        for SS in SSlist:
            SS = np.setdiff1d(SS, config_multiple_name.remove_silver_sample)
        print('####'.center(80, '#'))
        print(" Loading the data for the silver sample. ".center(80, '#'))
        print('####'.center(80, '#'))
        SS_Datalist = load_Datalist('SS', truth, SSlist, name, name_type, number_pair, work_dir=work_dir)
        list_Datalist.append(SS_Datalist)
        if (config_multiple_name.display_delay == True):
            for data in SS_Datalist: plot_delay("SS", data, figure_SS_directory)
        if (config_multiple_name.display_relative_err == True):
            for data in SS_Datalist: plot_error_rel("SS", data, figure_SS_directory)
        if (config_multiple_name.display_precision == True):
            for data in SS_Datalist: plot_precision("SS", data, figure_SS_directory)
        if (config_multiple_name.display_accuracy == True):
            for data in SS_Datalist: plot_accuracy("SS", data, figure_SS_directory)
        if (config_multiple_name.display_chi2 == True):
            for data in SS_Datalist: plot_chi2("SS", data, figure_SS_directory)
        if (config_multiple_name.display_SNR == True):
            for data in SS_Datalist: plot_SNR("SS", data, figure_SS_directory, name, name_type, work_dir)
        if (config_multiple_name.display_ML == True):
            for data in SS_Datalist: plot_ML("SS", data, figure_SS_directory, name, name_type, work_dir)

    # Golden sample
    if (config_multiple_name.display_gold == True):
        GSlist = []
        for SS in SSlist:
            GSlist.append(np.setdiff1d(SS, config_multiple_name.remove_golden_sample))
        print('####'.center(80, '#'))
        print(" Loading the data for the golden sample. ".center(80, '#'))
        print('####'.center(80, '#'))
        GS_Datalist = load_Datalist('GS', truth, GSlist, name, name_type, number_pair, work_dir=work_dir)
        list_Datalist.append(SS_Datalist)
        if (config_multiple_name.display_delay == True):
            for data in GS_Datalist: plot_delay("GS", data, figure_GS_directory)
        if (config_multiple_name.display_relative_err == True):
            for data in GS_Datalist: plot_error_rel("GS", data, figure_GS_directory)
        if (config_multiple_name.display_precision == True):
            for data in GS_Datalist: plot_precision("GS", data, figure_GS_directory)
        if (config_multiple_name.display_accuracy == True):
            for data in GS_Datalist: plot_accuracy("GS", data, figure_GS_directory)
        if (config_multiple_name.display_chi2 == True):
            for data in GS_Datalist: plot_chi2("GS", data, figure_GS_directory)
        if (config_multiple_name.display_SNR == True):
            for data in GS_Datalist: plot_SNR("GS", data, figure_GS_directory, name, name_type, work_dir)
        if (config_multiple_name.display_ML == True):
            for data in GS_Datalist: plot_ML("GS", data, figure_GS_directory, name, name_type, work_dir)

    if (config_multiple_name.display_tdc1 == True):
        plot_tdc1(FS_Datalist, "FS", figure_directory, config_multiple_name.compare_tdc1, number_pair)
        if (config_multiple_name.display_silver == True):
            plot_tdc1(SS_Datalist, "SS", figure_directory, config_multiple_name.compare_tdc1, number_pair)
        if (config_multiple_name.display_gold == True):
            plot_tdc1(GS_Datalist, "GS", figure_directory, config_multiple_name.compare_tdc1, number_pair)


if __name__ == '__main__':
    parser = ap.ArgumentParser(prog="python {}".format(os.path.basename(__file__)),
                               description="Reformat the txt file from the Time Delay Challenge to an usable rdb file.",
                               formatter_class=ap.RawTextHelpFormatter)
    help_name = "name of the sample. Make sure the directory has the same name."
    help_name_type = "Type of the data ie double or quad"
    help_number_pair = "number of pair in the rung folder. Make sure the folder have the format name_pair0"
    help_work_dir = "name of the work directory"
    parser.add_argument(dest='name', type=str,
                        metavar='name', action='store',
                        help=help_name)
    parser.add_argument(dest='name_type', type=str,
                        metavar='name_type', action='store',
                        help=help_name_type)
    parser.add_argument(dest='number_pair', type=int,
                        metavar='number_pair', action='store',
                        help=help_number_pair)
    parser.add_argument('--dir', dest='work_dir', type=str,
                        metavar='', action='store', default='./',
                        help=help_work_dir)
    args = parser.parse_args()
    main(args.name, args.name_type, args.number_pair, work_dir=args.work_dir)
