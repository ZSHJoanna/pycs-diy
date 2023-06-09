"""
Machinery to combine various probability distributions that are not necessarily Gaussian, nor analytical.

To combine various distributions, they need to be "linearized" on the same basis. For each delay, I define a interval range (see binlists below). This range will be used to draw numerical distributions matching the theoretical one, if you give me e.g. mean and std of an analytical Gaussian distrib. Make sure to have ranges large enough and binnings thin enough to properly reproduce the distributions.

"""
import copy
import logging
import os
import pickle as pkl

import numpy as np
import pycs3.gen.util

logger = logging.getLogger(__name__)


class CScontainer:
    """
    Curve-shifting container: basically just a bunch of information related to the PyCS structure of storing results.

    Could be a dictionnary as well, e.g.

    {"data": "C2", "knots": 25, "ml": "splml-150", "name": "C2_ks25_ml150", "drawopt": "spl1", "runopt": "spl1t5", "ncopy": 200, "nmocks": 1000, "truetsr": 3, "color": "royalblue"}

    This is simply a nicer interface between a PyCS simulation output and a Group.

    You can also give directly the result file without the extension '_errorbars.pkl' or 'delays.pkl'.
    """

    def __init__(self, name, knots=None, ml=None, data=None, drawopt=None, runopt=None, ncopy=None, nmocks=None, truetsr=None, colour='royalblue', result_file_delays=None,
                 result_file_errorbars=None):
        #mandatory
        self.name = name
        #optional
        self.data = data
        self.knots = knots
        self.ml = ml
        self.drawopt = drawopt
        self.runopt = runopt
        self.ncopy = ncopy
        self.nmocks = nmocks
        self.truetsr = truetsr
        self.colour = colour
        self.result_file_delays = result_file_delays
        self.result_file_errorbars = result_file_errorbars


class Group:
    """
    A class representing all the time-delay estimates from a single PyCS curve-shifting technique (1 choice of estimator, 1 set of estimator parameters, 1 data set) - i.e. all the delays between the various pairs of images.

    Groups should contain:

      * a list of labels, i.e. ["AB", "AC", "BC"] to which all other lists should correspond
      * a list of median delays (50th percentiles)
      * a list of errors_up and errors_down on the delay, corresponding to the 16th and 84th percentiles
      * an ugly but unique name

    Extra arguments computed inside the class:

      * a list of "linearized" distributions
      * a fancy name for display purposes
      * a list of name of the object if they are not standard such as ['A','B',...]

    """

    def __init__(self, labels, medians, errors_up, errors_down, name, binslist=None, lins=None, nicename=None,
                 ran_errors=None, sys_errors=None, objects=None, int_errors=None):

        # start with some assertion tests
        assert (len(labels) == len(medians) == len(errors_up) == len(errors_down))

        # mandatory params
        self.labels = labels
        self.objects = objects
        self.medians = medians
        self.errors_up = errors_up
        self.errors_down = errors_down
        self.name = name

        # optional params
        self.binslist = binslist
        self.nicename = nicename
        self.lins = lins
        self.ran_errors = ran_errors
        self.sys_errors = sys_errors
        self.int_errors = int_errors

    def linearize(self, testmode=True, verbose=True):
        """
        I create a Gaussian distribution with mean = my median and error = (errors_up + errors_down)/2 that I spread on my binslist

        .. warning:: Of course, if your mean and error are not expected to follow a Gaussian distribution, calling this function makes no sense.

        :param verbose: bool. Verbosity
        :param testmode: bool. Defines the number of samples drawn from the Normal distribution. More samples give more precise results but take longer to create and process. Setting testmode to True create 50k samples, otherwise 500k.
        """
        assert (self.binslist is not None)

        if testmode:
            nsamples = 50000
        else:
            nsamples = 500000

        lins = []
        if verbose:
            logger.info("*" * 10)
            logger.info(self.name)
        for ind, label in enumerate(self.labels):
            scale = (self.errors_up[ind] + self.errors_down[ind]) / 2.0
            xs = np.random.normal(loc=self.medians[ind], scale=scale, size=nsamples)

            digitized = np.digitize(xs, self.binslist[ind])
            bin_means = [float(len(xs[digitized == i])) for i in range(1, len(self.binslist[ind]))]
            bin_means = np.nan_to_num(bin_means)
            bin_means = bin_means / np.sum(bin_means)

            # To check we are not too far off
            ci = confinterval(0.5 * (self.binslist[ind][1:] + self.binslist[ind][:-1]), weights=bin_means,
                              testmode=testmode)
            if verbose:
                logger.info("=" * 45)
                logger.info("Delay %s:" % label + "%.2f" % self.medians[ind] + "+/-" + "%.2f" %scale + " --> " + "%.2f" % ci[0] +
                      "+%.2f-%.2f" %(ci[2], ci[1]))

            lins.append(bin_means)

        self.lins = lins

    def niceprint(self):
        """
        Nicely print in the terminal the median and 1sigma values of the current group.
        """
        if not self.nicename:
            name = self.name
        else:
            name = self.nicename

        logger.info("=" * 5 + name + "=" * 5)
        toprint = ""
        for ind, l in enumerate(self.labels):
            toprint += "%s: " % l + "%.2f +%.2f-%.2f\n" % (
                self.medians[ind], self.errors_up[ind], self.errors_down[ind])

        logger.info(toprint)


def getcombweightslist(groups):
    """
    Give me 2+ groups. If they have the same binning (return an error otherwise), I return their bin-by-bin linear combination

    :param groups: list of Group. must have a list of linearized distributions

    :return: weights of the combined distribution
    """

    # assert that each group has linearized distributions
    for group in groups:
        assert group.lins is not None, "Group %s has no linearized distributions - you must compute them firts!" % group.name

    # assert that the binnings are the same and that labels are in the same order
    for group in groups[1:]:
        assert group.binslist == groups[0].binslist, "Group %s has different bins than group %s" % (
            group.name, groups[0].name)
        assert group.labels == groups[0].labels, "Group %s has different labels (or labels order) than group %s" % (
            group.name, groups[0].name)

    # we assume the groups are ordered according to their labels (assert test would have failed otherwise)
    name = "+".join([g.name for g in groups])
    weightslist = {"weights": [], "binslist": groups[0].binslist, "labels": groups[0].labels, "name": name}

    for ind, label in enumerate(groups[0].labels):
        indweights = [g.lins[ind] for g in groups]

        """
        This is trickily written so let me comment
        indweights contains a list of various linearized corresponding normalised distributions binned the same way (i.e. AB pdf for spl20 and spl30). The values of distrib are added bin by bin, then reweighted so that the sum of the result is still normalised to 1 (so they can be combined later with other results from various techniques, instruments, etc...) Since we work with numpy arrays, this can be done in a single line.
        """
        weights = sum(indweights) / float(len(indweights))
        weightslist["weights"].append(weights)
    return weightslist


def getresults(csc, useintrinsic=False):
    """
    Give me a CScontainer, I read the corresponding result folder and create a group from it.

    .. warning:: This works **only** if you already collected the runresults in delaycontainer, using e.g. the :py:func:`pycs3.sim.plot.hists` and :py:func:`pycs3.sim.plot.measvstrue` functions.

    I also save the random and systematic component of the error, in case you want to display them later on.

    :param csc: CScontainer object
    :param useintrinsic: Boolean. If True, add the intrinsic measured error to the total error from sims, in quadrature.
    :return: a Group object


    .. warning:: Although the code says I'm using the mean measured delay, I might be using median instead! This depends on how :py:func:`pycs3.sim.plot.hists` was called! Be careful with this...


    """

    # get the correct results path
    if csc.result_file_delays is None and csc.result_file_errorbars is None:
        raise RuntimeError("Your CScontainer do not contains the '_errorbar.pkl' or '_delays.pkl'. Run pycs3.sim.plot.hists and pycs3.sim.plot.measvstrue first !")
    else:
        cp = csc.result_file_delays
        mp = csc.result_file_errorbars

    # collect the results
    result = (pycs3.gen.util.readpickle(cp), pycs3.gen.util.readpickle(mp))

    # read them
    labels = [d["label"] for d in result[0].data]
    means = []
    errors = []
    ranerrors = []
    syserrors = []
    interrors = []

    for label in labels:
        means.append([d["mean"] for d in result[0].data if d["label"] == label][0])
        if not useintrinsic:
            errors.append([d["tot"] for d in result[1].data if d["label"] == label][0])
        else:
            errors.append(np.sqrt([d["tot"] for d in result[1].data if d["label"] == label][0] ** 2 +
                                  [d["std"] for d in result[0].data if d["label"] == label][0] ** 2))
            interrors.append([d["std"] for d in result[0].data if d["label"] == label][0])
        ranerrors.append([d["ran"] for d in result[1].data if d["label"] == label][0])
        syserrors.append([d["sys"] for d in result[1].data if d["label"] == label][0])

    # create a Group out of them
    group = Group(labels=labels, medians=means, errors_up=errors, errors_down=errors, ran_errors=ranerrors,
                 sys_errors=syserrors, int_errors=interrors, name=csc.name)
    if csc.colour is not None :
        group.plotcolor = csc.colour

    return group

def asgetresults(weightslist, testmode=True):
    """
    Transform any given weightlist into a Group.

    We keep getcombweightlist and asgetresults separated, so we can import any binned result as a Group

    :param weightslist: a dictionnary containing a list of labels, a list of bins and a list of weights, whose order matches. Typically the output of getcombweightlist.
    :param testmode:
    :return: a Group object
    """
    medians = []
    errors_up = []
    errors_down = []
    lins = []
    for b, w in zip(weightslist["binslist"], weightslist["weights"]):
        ci = confinterval(0.5 * (b[1:] + b[:-1]), w, testmode=testmode)
        medians.append(ci[0])
        errors_up.append(ci[2])
        errors_down.append(ci[1])
        lins.append(w)

    return Group(labels=weightslist["labels"], medians=medians, errors_up=errors_up, errors_down=errors_down,
                 binslist=weightslist["binslist"], lins=lins, name=weightslist["name"])


def confinterval(xs, weights=None, conflevel=0.68, cumulative=False, testmode=True):
    """
    Hand-made quantile/percentile computation tool

    :param xs: list, distribution
    :param weights: list, weights of the distribution
    :param conflevel: float, quantile (between 0 and 1) of the conf interval
    :param cumulative: bool. If set to True, return the "bound" (max) value of xs corresponding to the conflevel. For example, if conflevel=0.68, xs=nnu, return the value such that nnu<=val at 68% confidence.
    :param testmode: bool. Defines the finesse of the binning of the histogram used when computing the percentiles. A finer binning gives more precise results but takes longer to process. Setting it to True sets 500 bins, otherwise 10'000.

    :return: return 3 floats: the median value, minimal value and maximal value matching your confidence level
    """

    if testmode:
        nbins = 500
    else:
        nbins = 10000

    hist, bin_edges = np.histogram(xs, weights=weights, bins=nbins)

    if not cumulative:
        frac = 0.
        for ind, val in enumerate(hist):
            frac += val
            if frac / sum(hist) >= 0.5:
                meanval = (bin_edges[ind] + bin_edges[ind - 1]) / 2.0
                break

        lowerhalf = 0.
        upperhalf = 0.
        for ind, val in enumerate(bin_edges[:-1]):
            if val < meanval:
                lowerhalf += hist[ind]
            else:
                upperhalf += hist[ind]

        limfrac = (1.0 - conflevel) / 2.0
        lowerfrac, upperfrac = 0., 0.
        for ind, val in enumerate(hist):
            lowerfrac += val
            if lowerfrac / sum(hist) > limfrac:
                bottomlimit = (bin_edges[ind] + bin_edges[ind - 1]) / 2.0
                break

        for ind, val in enumerate(hist):
            upperfrac += val
            if upperfrac / sum(hist) > 1.0 - limfrac:
                toplimit = (bin_edges[ind] + bin_edges[ind - 1]) / 2.0
                break

        return meanval, (meanval - bottomlimit), -1.0 * (meanval - toplimit)

    else:
        frac = 0.
        for ind, val in enumerate(hist):
            frac += val
            if frac / sum(hist) >= conflevel:
                return (bin_edges[ind] + bin_edges[ind - 1]) / 2.0


def get_bestprec(groups, refimg=None, verbose=False):
    """
    Return the most precise Group among a list of Groups.

    The most precise Group is computed in a straightforward, naive way: simply sum the relative precision over all the pair of delays that contain refimg in their label (or all, if no refimg given), then take the estimate that has the lowest sum.

    To avoid giving too much weight to the larger delays, we compute the precision relative to the "mean" delay from all estimates.

    :param groups: list of Groups
    :param refimg: reference image (e.g. "A"), if you want to discard all the delays that have not "A" in their label
    :param verbose: bool. Verbosity
    :return: index of the most precise Group in the list
    """

    # compute average means:
    avgmeans = []
    for l in groups[0].labels:
        medians = [group.medians[group.labels.index(l)] for group in groups]
        avgmeans.append(np.mean(medians))

    precs = []
    for group in groups:
        thisgroupprecs = []

        if refimg is None:
            mylabels = group.labels
        else:
            mylabels = [l for l in group.labels if refimg in l]

        for l in mylabels:
            mean = avgmeans[groups[0].labels.index(l)]
            err = (group.errors_up[group.labels.index(l)] + group.errors_down[group.labels.index(l)]) / 2.0

            thisgroupprecs.append(err / np.abs(mean))
            if verbose:
                logger.info("=" * 5)
                logger.info("%s: %.2f +- %.2f" % (l, np.abs(mean), err))
                logger.info("P = %.2f" % (err / np.abs(mean)))

        precs.append(sum(thisgroupprecs))

    if verbose:
        for ind, prec in enumerate(precs):
            logger.info("Total prec of %i: %f" % (ind, prec))

    return precs.index(min(precs))


def compute_sigmas(refgroup, groups, verbose=False, niceprint=True):
    """
    Compute the tension (in sigma units) between a reference group and the rest of the series

    :param refgroup: Reference delays to which you want to compute the tension with. Dict must be the output of getresults function.
    :param groups: List of Groups on which the tension is computed.
    :param niceprint: Bool. If True, print the tension in the terminal in a nice way.
    :param verbose: Bool. Verbosity
    :return: a list (one elt per dict in dictslist, same order) of list (one val per label, same order than each dict) of sigmas.
    """

    sigmaslist = []
    for group in groups:
        sigmas = []
        for l in group.labels:

            # the reference estimate
            refmedian = refgroup.medians[refgroup.labels.index(l)]
            referr_up = refgroup.errors_up[refgroup.labels.index(l)]
            referr_down = refgroup.errors_down[refgroup.labels.index(l)]

            # the estimate compared to the ref
            median = group.medians[group.labels.index(l)]
            err_up = group.errors_up[group.labels.index(l)]
            err_down = group.errors_down[group.labels.index(l)]
            if median < refmedian:
                sigma = np.abs(median - refmedian) / np.sqrt(err_up ** 2 + referr_down ** 2)
            else:
                sigma = np.abs(median - refmedian) / np.sqrt(err_down ** 2 + referr_up ** 2)

            sigmas.append(sigma)
            if verbose:
                logger.info("=" * 15)
                logger.info("%s: " % l, end=' ')
                # reference estimate
                logger.info("ref: %.2f + %.2f - %.2f" % (refmedian, referr_up, referr_down))

                #  comparision estimate
                logger.info("ref: %.2f + %.2f - %.2f" % (median, err_up, err_down))
                logger.info("tension: ", sigma)

        sigmaslist.append(sigmas)

    if niceprint:
        logger.info("=" * 5 + "Tension comparison" + "=" * 5)
        logger.info("Reference is %s" % refgroup.name)
        for name, sigmas, labels in zip([g.name for g in groups], sigmaslist, [g.labels for g in groups]):
            logger.info("---%s---" % name)
            for sigma, label in zip(sigmas, labels):
                logger.info("  %s: %.2f" % (label, sigma))

    return sigmaslist


def convolve_estimates(group, errors, testmode=True):
    """
    Convolve the group estimates by the errors distribution. This can be used if you want to add another source of
    errors (e.g., microlensing time delay, Tie & Kochanek 2018).

    Warning : this will linearize the distributions before the convolution so it will just add in quadrature the sigma of the
    two Gaussian distribution and displace the mean...
    @todo : Make this work for non-Gaussian distribution

    :param group: Group that contain the estimates you want to convolve
    :param errors: Group that contains the error distribution.
    :param testmode: Bool, control the finesse of the binning

    :return: a new group, resulting from the convolution of the group error with errors.
    """

    # we make sure the group is already linearized
    assert group.lins is not None, "Your group has not been linearized. You must do it first"

    # errors are given as an analytical Gaussian distrib, I directly linearize on binslist
    if errors.lins is None:
        # I need to put the zero of the error distribution at the center of the padding
        # --> I simply shift the binslist I use so that it is centered on 0
        newbinslist = []
        for b in group.binslist:
            bmean = np.mean(b)
            newbins = b - bmean
            newbinslist.append(newbins)

        errors.binslist = newbinslist
        errors.linearize(testmode=testmode)

    # errors is linearized, but not necessarily on the same binning
    else:

        newbinslist = []
        newlinslist = []

        for oldlins, oldbins, elabel in zip(errors.lins, errors.binslist, errors.labels):
            mybins = group.binslist[group.labels.index(elabel)]
            newbins = mybins - np.mean(mybins)  # centering the padding
            centeroldbins = [oldbins[i] + (oldbins[i+1] - oldbins[i]) / 2. for i in range(len(oldbins)-1)]
            newvals = np.interp(x=newbins, xp=centeroldbins, fp=oldlins, left=0, right=0)
            newvals = newvals / sum(newvals)

            newbinslist.append(newbins)
            newlinslist.append(newvals[:-1])

        errors.binslist = newbinslist
        errors.lins = newlinslist

    # and do the convolution
    convols = []
    for gl, label in zip(group.lins, group.labels):
        el = errors.lins[errors.labels.index(label)]

        convols.append(np.convolve(gl, el, mode="same"))

    convmedians = []
    converrors_up = []
    converrors_down = []
    for c, b in zip(convols, group.binslist):
        ci = confinterval(xs=0.5 * (b[1:] + b[:-1]), weights=c, testmode=testmode)
        convmedians.append(ci[0])
        converrors_down.append(ci[1])
        converrors_up.append(ci[2])

    convolved = Group(labels=group.labels, medians=convmedians, errors_up=converrors_up, errors_down=converrors_down,
                      name=group.name + "-conv-" + errors.name, binslist=group.binslist, lins=convols)

    return convolved


def combine_estimates(groups, sigmathresh, verbose=True, testmode=True):
    """
    Top level function doing everything in one line (or close to...)

    Give me a list of groups, I pick the most precise on by myself, compute the tension with all the others. If the tension exceed a certain threshold, I combine (i.e. sum) the best estimate with the most precise estimate that it is in tension with. I iteratively compute the tension and combine with the remaining estimates if necessary.

    :param groups: list of Groups
    :param sigmathresh: float, tension threshold, if the tension exceed this value, the groups in tension are combined
    :param verbose:  boolean, verbosity
    :param testmode: boolean, for fast computation, if you want more stable results use False
    :return: the final combined Group
    """

    # I will iterately remove groups from this list, so I duplicate it to avoid removing from the original one.
    mygroups = [g for g in groups]

    # I first compute the tension between the best estimate and the others
    refgroup = groups[get_bestprec(groups)]
    sigmaslist = compute_sigmas(refgroup, groups, niceprint=verbose)

    # If the tension exceed a threshold in a least one of the delays, I identify the corresponding estimates.
    threshgroups = []
    for ind, sigmas in enumerate(sigmaslist):
        for sigma in sigmas:
            if sigma > sigmathresh:
                threshgroups.append(groups[ind])
                break

    i = 1  # to keep track of the iterations
    # Iterative combination of estimates in tension
    while len(threshgroups) > 0:

        # when iteratively computing the tension, the combined estimates must weight more and more since it results from the combination of an increasing number of estimates. Yet, getcombweighslist return normalized pdf. To bypass this, I add bestest to the combination list one more time at each iteration. Sounds stupid but if it works it ain't stupid.
        tocombine = []
        for ind in np.arange(i):
            tocombine.append(refgroup)

        tocombine.append(threshgroups[get_bestprec(threshgroups)])

        combined = asgetresults(getcombweightslist(tocombine), testmode=testmode)
        # small haxx to get rid of repetitions in the name
        combined.name = '+'.join(list(set(combined.name.split('+')))[::-1])

        # remove the already combined group from mygroups
        mygroups = [g for g in mygroups if g.name not in (refgroup.name, threshgroups[get_bestprec(threshgroups)].name)]

        # recompute the tension
        sigmaslist = compute_sigmas(combined, mygroups, niceprint=verbose)
        threshgroups = []
        for ind, sigmas in enumerate(sigmaslist):
            for sigma in sigmas:
                if sigma > sigmathresh:
                    threshgroups.append(mygroups[ind])
                    break

        # Loop again if threshgroups > 0, otherwise exit the loop and return the combined estimate
        refgroup = combined  # this is the new ref we want to compare the remaining estimates to.

        i += 1

    return refgroup


def mult_estimates(groups, testmode=True):
    """
    Multiply various groups - assert that they can be multiplied (same binning, same label order)

    :param groups: list of Group that you want to multiply
    :param testmode: bool, control the finesse of the multiplication and binning

    :return: Group, multiplication of the entry groups
    """

    # assert that each group has linearized distributions
    for group in groups:
        assert group.lins is not None, "Group %s has no linearized distributions - you must compute them firts!" % group.name

    # assert that the binnings are the same and that labels are in the same order
    # todo: instead of rejecting if label orders are not the same, simply do the multiplication label per label, as done in other functions above.
    for group in groups[1:]:
        assert np.allclose(group.binslist, groups[0].binslist), "Group %s has different bins than group %s" % (
            group.name, groups[0].name)
        assert group.labels == groups[0].labels, "Group %s has different labels (or labels order) than group %s" % (
            group.name, groups[0].name)

    # we assume the groups are ordered according to their labels (assert test would have failed otherwise)
    name = "-x-".join([g.name for g in groups])
    weightslist = {"weights": [], "binslist": groups[0].binslist, "labels": groups[0].labels, "name": name}

    for ind, label in enumerate(groups[0].labels):
        indweights = [g.lins[ind] for g in groups]

        multweights = np.ones(len(indweights[0]))
        for iw in indweights:
            multweights *= np.array(iw)

        # renormalize multweights to 1
        multweights = multweights / np.sum(multweights)

        weightslist["weights"].append(multweights)

    return asgetresults(weightslist, testmode=testmode)


def group_estimate(path_list, name_list = None, colors=None, sigma_thresh=0, new_name_marg = 'Combined',
                   object_name=None, testmode=True):
    """
    Top level function to combine groups stored in pickle files.
    Give a list of path to the pickle files containing the group to be combined. I am a more flexible version of
    combine_estimate(), since you can rename the group and I automatically re-adjust the bins. This function is used
    at the very last step of the automated pipelines when the spline and regdiff are combined.

    :param path_list: list of path to the pickles containing
    :param name_list: list of string, if you want to rename the groups
    :param delay_labels:
    :param colors: list of string, if you want to change the color of the groups
    :param sigma_thresh: sigmathresh: float, tension threshold, if the tension exceed this value, the groups in tension are combined (0 is a true marginalization)
    :param new_name_marg: string, name of the combined Group
    :param testmode: boolean, for fast computation, if you want more stable results use False
    :param object_name: list of string, if you want to rename the object
    :return: tuple containing (list of Group, combined Group)
    """
    if testmode:
        nbins = 500
    else:
        nbins = 5000

    group_list = []
    medians_list = []
    errors_up_list = []
    errors_down_list = []
    if len(path_list) != len(name_list):
        raise RuntimeError("Path list and name_list should have he same lenght")
    for p, path in enumerate(path_list):
        if not os.path.isfile(path):
            logger.warning("I cannot find %s. I will skip this one. Be careful !" % path)
            continue

        group = pkl.load(open(path, 'rb'))
        if name_list is not None :
            group.name = name_list[p]
        group_list.append(group)
        medians_list.append(group.medians)
        errors_up_list.append(group.errors_up)
        errors_down_list.append(group.errors_down)

    # build the bin list :
    medians_list = np.asarray(medians_list)
    errors_down_list = np.asarray(errors_down_list)
    errors_up_list = np.asarray(errors_up_list)
    binslist = []
    for i in range(len(medians_list[0, :])):
        bins = np.linspace(min(medians_list[:, i]) - 10 * min(errors_down_list[:, i]),
                           max(medians_list[:, i]) + 10 * max(errors_up_list[:, i]), nbins)
        binslist.append(bins)

    color_id = 0
    for g, group in enumerate(group_list):
        if colors is not None :
            group.plotcolor = colors[color_id]
        group.binslist = binslist
        group.linearize(testmode=testmode)
        group.objects = object_name
        color_id += 1
        if color_id >= len(colors):
            logger.warning("I don't have enough colors in my list, I'll restart from the beginning.")
            color_id = 0  # reset the color form the beginning

    combined = copy.deepcopy(pycs3.tdcomb.comb.combine_estimates(group_list, sigmathresh=sigma_thresh, testmode=testmode))
    combined.linearize(testmode=testmode)
    combined.name = 'Combined ($\\tau_{thresh} = %2.1f$)' % sigma_thresh
    combined.plotcolor = 'black'
    logger.info("Final combination for marginalisation %s"%new_name_marg)
    combined.niceprint()

    return group_list, combined

