"This module contains various function that facilitate the formatting of the inputs when using the automated pipeline"

import json
import os
import sys
from inspect import getsource
from textwrap import dedent

import numpy as np


def proquest(askquestions):
    """
    Asks the user if he wants to proceed. If not, exits python.

    :param askquestions: boolean, askquestions is a switch, True or False, that allows to skip the questions.

    """
    if askquestions: # pragma: no cover
        answer = input("Tell me, do you want to go on ? (yes/no) ")
        if answer[:3] != "yes":
            sys.exit("Ok, bye.")
        print("")  # to skip one line after the question.


def getdelays(lcs):
    """
    Function to obtain the time delay pairs given a list of light curves.
    The function return a list of delay between the pai of images and a list containing the name of the image pairs.
    :param lcs: list of LightCurves

    :return: tuple containing (array of delay pairs, array of delay name)
    """

    delay_pair = []
    delay_name = []

    for i, lci in enumerate(lcs):
        for j, lcj in enumerate(lcs):
            if i >= j:
                continue
            else:
                delay_name.append(lci.object + lcj.object)
                delay_pair.append(lcj.timeshift - lci.timeshift)

    delay_pair = np.asarray(delay_pair)
    delay_name = np.asarray(delay_name)

    return delay_pair, delay_name


def write_func_append(fn, stream, **kwargs):
    """
    Function to write a python function in a file, replacing the variable by their value
    :param fn: python function to be written in a file
    :param stream: Python File Object
    :param kwargs: dictionnary, containing the keyword argument to be written in the file

    """
    fn_as_string = getsource(fn)
    for var in kwargs:
        fn_as_string = fn_as_string.replace(var, kwargs[var])
        fn_as_string = dedent(fn_as_string)

    stream.write('\n' + fn_as_string)


def generate_regdiffparamskw(pointdensity, covkernel, pow, errscale):
    """
    Generate the naming keywords, given the parameter of regdiff. See pycs3.regdiff documentation for details.

    :param pointdensity: integer, point density parameter to be passed to regdiff
    :param covkernel: string, covariance kernel to be passed to regdiff, Choose between 'RBF', 'Matern' and "RatQuad"
    :param pow: float, exponent of the Mattern function
    :param errscale: float, additionnal scaling of the photometric error bars.

    :return: list of string containing the keywords
    """
    out_kw = []
    for c in covkernel:
        for pts in pointdensity:
            for p in pow:
                for e in errscale:
                    out_kw.append("_pd%.1f_ck%s_pow%.1f_errsc%i_" % (
                        pts, c, p, e))
    return out_kw


def read_preselected_regdiffparamskw(file):
    """
    Read a text file readable with json and extract the keyword for regdiff.

    :param file: string, path to the file containing the preselected parameters

    :return: list of string containing the keywords
    """
    out_kw = []
    with open(file, 'r') as f:
        dic = json.load(f)
        for d in dic:
            if d['covkernel'] == 'RBF':  # no pow parameter, RBF correspond to Mattern with pow --> inf
                out_kw.append("_pd%.1f_ck%s_errsc%i_" % (
                d['pointdensity'], d['covkernel'], d['errscale']))
            else:
                out_kw.append("_pd%.1f_ck%s_pow%.1f_errsc%i_" % (
                d['pointdensity'], d['covkernel'], d['pow'], d['errscale']))
    return out_kw


def get_keyword_regdiff(pointdensity, covkernel, pow, errscale):
    """
    Generate the dictionnary to be given to the regdiff optimiser. See pycs3.regdiff documentation for details.

    :param pointdensity: integer, point density parameter to be passed to regdiff
    :param covkernel: string, covariance kernel to be passed to regdiff, Choose between 'RBF', 'Matern' and "RatQuad"
    :param pow: float, exponent of the Matern function
    :param errscale: float, additionnal scaling of the photometric error bars.

    :return: list of dictionnary containing the argument to be given to the regdiff optimiser
    """

    kw_list = []
    for c in covkernel:
        for pts in pointdensity:
            for p in pow:
                for e in errscale:
                        kw_list.append(
                            {'covkernel': c, 'pointdensity': pts, 'pow': p, 'errscale': e})
    return kw_list


def get_keyword_regdiff_from_file(file):
    """
    Read a text file readable with json and extract the keyword for regdiff.

    :param file: string, path to the file containing the preselected parameters

    :return: : list of dictionnary containing the argument to be given to the regdiff optimiser
    """
    with open(file, 'r') as f:
        kw_list = json.load(f)
    return kw_list


def get_keyword_spline(kn):
    """
    Generate the dictionnary to be given to the spline optimiser.

    :param kn: integer, knotstep of the spline optimiser

    :return: dictionnary containing the argument to be given to the spline optimiser
    """
    return {'kn': kn}


def convert_delays2timeshifts(timedelay):
    """
    Convert the time-delays you can measure by eye into time-shifts for the individual curve
    :param timedelay: list of time delays (only AB,AC,AD in the case of a quad)

    :return: list of timeshifts
    """
    timeshift = np.zeros(len(timedelay) + 1)
    timeshift[1:] = timedelay
    return timeshift


def mkdir_recursive(path): # pragma: no cover
    """
    Create directory recursively if they subfolders do not exist
    :param path: folder to be created

    """
    sub_path = os.path.dirname(path)
    if not os.path.exists(sub_path):
        mkdir_recursive(sub_path)
    if not os.path.exists(path):
        os.mkdir(path)
