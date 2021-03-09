"""
Functions to run curve shifting techniques on lightcurves produced by sim.multidraw.
We define the class runresults that "holds" results obtained by curve shifting techniques (saved into pickles).
"""
import copy as pythoncopy
import logging
import os
import time
from glob import glob

import numpy as np
import pycs3.gen.lc
import pycs3.gen.lc_func
import pycs3.gen.util
import pycs3.sim.draw

logger = logging.getLogger(__name__)


def optfct_aux(argument):
    """
    Auxiliary function for multithreading
    """
    arg, id, kwargs, optfct = argument
    try:
        out = optfct(arg, **kwargs)
    except Exception as e:
        # logger.warning("I have a problem with the curve number %i. Details : %s" % (i, e))
        dic = {'success': False, 'failed_id': [id], 'error_list': [e]}
        out = None
    else:
        dic = {'success': True, 'failed_id': [], 'error_list': []}

    return out, dic

def applyopt(optfct, lcslist, **kwargs):
    """
    Applies optfct (an optimizing function that takes a list of lightcurves as single argument)
    to all the elements (list of lightcurves)
    of lcset (a list of lists of lightcurves).

    Optimizes the lightcurves themselves, in place, and returns a list of the outputs of the optimizers, corresponding to the lcslist.
    For instance, if the optfct output is a spline, it also contains the final r2s, that I will later save into the pkls !

    :param optfct: function to apply to the LightCurves
    :param lcslist: list of LightCurves
    :param kwargs: dictionnary of kwargs to be transmitted to the optfct

    :return: optfct_outs: a list of the optimised LightCurves and a dictionnary containing the failed attempt to shift the curves.
    """

    # ncpuava = multiprocess.cpu_count()

    # if ncpu > 1 :
    #     #todo : there is a very nasty bug here : the microlensing is not optimised in multiprocessing. I can't find why this is happening so do not use multiprocessing until this is solve. The bug is quiet it will just provide a very bad fit to the data !!! This must have to do with object that are not deepcopied...
    #     logger.warning('Multi-processing is not supported on this verison ! You can still use a higher level of parallelisation. I will run on a single core for the moment.')
    #     ncpu = 1

    # if ncpu == 1:
    # curves are optimised one after the other :
    # This now works with multi-threading. Consider using ncpu=1 if you use higher level of parallelisation.
    # If one thread is failing, this should not impact the others, but the optfct_outs won't have the same length as the lcslist.
    # Call clean_simlist if you want to remove the failed attempts from the lcslist.

    logger.info("Starting the curve shifting on a single CPU, no multiprocessing...")
    start = time.time()
    kwargs_vec = [kwargs for k in lcslist]
    optfct_outs = []
    sucess_dic = {'success': True, 'failed_id': [], 'error_list': []}
    for i, lcs in enumerate(lcslist):
        try:
            optout = optfct(lcs, **kwargs_vec[i])
        except Exception as e: # pragma: no cover
            logger.warning("I have a problem with the curve number %i. Details : %s" %(i,e))
            sucess_dic['failed_id'].append(i)
            sucess_dic['success'] = False
            sucess_dic['error_list'].append(e)
        else:
            optfct_outs.append(optout)


    # else: # pragma: no cover # Multithreading version. Cannot be tested on CI server
    #
    #     if ncpu == None:
    #         ncpu = ncpuava
    #     if ncpu == -1:
    #         ncpu = ncpuava - 1
    #
    #     logger.info("Starting the curve shifting on %i/%i CPUs." % (ncpu, ncpuava))
    #     start = time.time()
    #
    #     job_args = [(copy.deepcopy(k), id, copy.deepcopy(kwargs), copy.deepcopy(optfct)) for id, k in enumerate(lcslist)]
    #     sucess_dic = {'success': True, 'failed_id': [], 'error_list': []}
    #
    #     pool = multiprocess.Pool(ncpu)
    #     results = pool.map(optfct_aux, job_args)  # order should be conserved with map_async
    #     pool.close()
    #     pool.join()
    #
    #     optfct_outs = []
    #     for output in results:
    #         if output[0] is not None:
    #             optfct_outs.append(output[0])
    #         if output[1]['success'] is False:
    #             sucess_dic['success'] = False
    #             sucess_dic['failed_id'] = sucess_dic['failed_id'] + output[1]['failed_id']
    #             sucess_dic['error_list'] = sucess_dic['error_list'] + output[1]['error_list']


    if len(optfct_outs) == 0:
        logger.warning(" It seems that your optfct does not return anything ! ")
    else:
        logger.info("Shifted %i/%i simulations, time : %s" % (len(optfct_outs),
                                                                      len(lcslist),
                                                                      pycs3.gen.util.strtd(time.time() - start)))
        if len(lcslist) - len(optfct_outs) > 0 :
            logger.warning("I failed for %i curves." % (len(lcslist) - len(optfct_outs)))

    return optfct_outs, sucess_dic


class RunResults:
    """
    Summarizes the huge list of list of lightcurves as a numpy array of timeshifts and some further info,
    to serve as input for plots, and actual time delay determinations.
    This replaces the old "boot.pkl" files ...
    The TRUE shifts are also saved (if available)

    All this is not related to a particular optimization technique.
    Please also provide the success_dic to remove the curves where the optimiser failed.

    Note the mask functionality.
    """

    def __init__(self, lcslist, qs=None, name="None", plotcolour="#008800", success_dic=None):
        """
        lcslist may or may not have "truetimeshifts". If not, I will put 0.0 as true shifts.

        qs should be a numpy array (as long as lcslist) that contains some chi2, r2, or d2 stuff to quantify how good the fit was.

        All the lcs in lcslist should be in the same order (I will check this, even if slow).
        I will not sort these lcs (as you might want them in an unsorted order).

        """
        if qs is not None:
            self.qs = qs
            if qs.shape[0] != len(lcslist): # pragma: no cover
                raise RuntimeError("These qs don't have the right length !")
        else:
            # We put zeros...
            self.qs = np.zeros(len(lcslist))

        if len(lcslist) == 0:# pragma: no cover
            raise RuntimeError("Should this happen ?")

        self.tsarray = np.vstack([[l.timeshift for l in lcs] for lcs in lcslist])
        # First index selects the simulation, second index selects the timeshifts of the curves in each lcs.
        # We build a similar array for the true shifts (value = 0.0 if the curve was not drawn)
        self.truetsarray = np.vstack([[getattr(l, "truetimeshift", 0.0) for l in lcs] for lcs in lcslist])

        # We check the ordering of the lcs in lcslist
        objectstringsasset = set(["/".join([l.object for l in lcs]) for lcs in lcslist])
        if len(objectstringsasset) != 1:# pragma: no cover
            raise RuntimeError("Ouch, your lcs in lcslist are not identical/ordered !")

        self.labels = [l.object for l in lcslist[0]]

        self.name = name
        self.autoname = name

        self.plottrue = False  # By default we plot the measured delays.
        self.plotgauss = False
        self.plotcolour = plotcolour
        self.success_dic = success_dic

        self.check()

    def __len__(self):
        """
        The number of runs
        """
        return self.tsarray.shape[0]

    def nimages(self):
        """
        The number of images (4 for a quad, 2 for a double) ...
        """
        return self.tsarray.shape[1]

    def __str__(self):
        return "Runresults '%s' (%i)" % (getattr(self, "name", "untitled"), len(self))

    def copy(self):
        """
        Return a copy of itself

        """
        return pythoncopy.deepcopy(self)

    def check(self):
        """
        Check that the RunResults object is correctly built.

        """

        if self.qs.shape[0] != len(self):# pragma: no cover
            raise RuntimeError("qs length error")
        if self.tsarray.shape != self.truetsarray.shape:# pragma: no cover
            raise RuntimeError("tsarray shape error")

    def applymask(self, mask):
        """
        Removes some of the runresults according to your mask.
        """

        self.tsarray = self.tsarray[mask]
        self.truetsarray = self.truetsarray[mask]
        self.qs = self.qs[mask]
        self.check()

    def gettruets(self):
        """
        Returns some summary stats about the true delays.
        Used to find histogram ranges for plots, etc.
        """
        ret = {"center": np.median(self.truetsarray, axis=0), "max": np.max(self.truetsarray, axis=0),
               "min": np.min(self.truetsarray, axis=0)}
        spans = ret["max"] - ret["min"]
        ret["type"] = "distribution"
        if np.all(spans < 0.00001):  # Then all true delays are identical
            ret["type"] = "same"
            if np.all(np.absolute(ret["center"]) < 0.00001):  # Then we have no true delays (i.e., they are all 0).
                ret["type"] = "none"

        return ret

    def get_delays_from_ts(self):
        """
        Return the time delays, from the timeshifts. I do not account for the true timeshift.

        :return: dictionary containing the median, max, and min delays + delay labels
        """
        n = len(self.labels)
        couples = [(self.tsarray[:, i], self.tsarray[:, j]) for i in range(n) for j in range(n) if i < j]
        label_couple = [self.labels[i] + self.labels[j] for i in range(n) for j in range(n) if i < j]
        ret = {"center": [np.median(lcs2 - lcs1) for (lcs1, lcs2) in couples]}
        ret["max"] = [np.max(lcs2 - lcs1) for (lcs1, lcs2) in couples]
        ret["min"] = [np.min(lcs2 - lcs1) for (lcs1, lcs2) in couples]
        ret["delay_label"] = label_couple
        ret["type"] = "delay distribution"
        return ret

    def getts(self):
        """
        A bit similar to gettruets, we return the median of the measured ts...
        Used for plots etc, not for calculations.
        """
        ret = {"center": np.median(self.tsarray, axis=0), "max": np.max(self.tsarray, axis=0),
               "min": np.min(self.tsarray, axis=0), "type": "distribution"}
        return ret

    def print_shifts(self):
        """
        Print the shift from a list of RunResults object

        """
        labeltxt = "%s (%s, %i) " % (
            getattr(self, 'name', 'NoName'), "Truth" if self.plottrue else "Measured", self.tsarray.shape[0])
        logger.info('Plotting "%s"' % labeltxt)
        logger.info("     Labels : %s" % (", ".join(self.labels)))
        logger.info("     Median shifts : %s" % (
            ", ".join(["%.2f" % (np.median(self.tsarray[:, i])) for i in range(len(self.labels))])))
        logger.info(
            "     Std shifts : %s" % (
                ", ".join(["%.2f" % (np.std(self.tsarray[:, i])) for i in range(len(self.labels))])))


def joinresults(rrlist):
    """
    Give me a list of runresults objects, I join those into a single one an return the latter.
    """

    if len(rrlist) == 0:# pragma: no cover
        raise RuntimeError("Your rrlist is empty !")

    joined = rrlist[0].copy()  # Just to get an object, with labels from the first rr.
    # Perform lots of test if it is ok to join these results ...
    for rr in rrlist:
        if rr.labels != joined.labels:
            raise RuntimeError("Don't ask me to join runresults of different objects !")

    joined.name = "+".join(list(set([getattr(rr, 'name', 'NoName') for rr in rrlist])))
    joined.autoname = "%s" % joined.name
    joined.tsarray = np.vstack([rr.tsarray for rr in rrlist])
    joined.truetsarray = np.vstack([rr.truetsarray for rr in rrlist])
    joined.qs = np.concatenate([rr.qs for rr in rrlist])

    joined.check()
    return joined


def collect(directory="./test", plotcolour="#008800", name=None):
    """
    Collects the runresult objects from a directory (typically from a multirun run),
    and returns the joined runresults.

    """
    if not os.path.isdir(directory):# pragma: no cover
        raise RuntimeError("I cannot find the directory %s" % directory)
    pklfiles = sorted(glob(os.path.join(directory, "*_runresults.pkl")))
    if len(pklfiles) == 0:# pragma: no cover
        raise RuntimeError("I couldn't find pkl files in directory %s" % directory)
    logger.info("Reading %i runresult pickles..." % (len(pklfiles)))
    rrlist = [pycs3.gen.util.readpickle(pklfile, verbose=False) for pklfile in pklfiles]
    jrr = pycs3.sim.run.joinresults(rrlist)
    jrr.plotcolour = plotcolour
    if name is not None:
        jrr.name = name
    logger.info("OK, I have collected %i runs from %s" % (len(jrr), jrr.name))
    return jrr


def multirun(simset, lcs, optfct, kwargs_optim, optset="multirun", tsrand=10.0, shuffle=True,
             keepopt=False, trace=False, verbose=True, destpath="./", use_test_seed=False):
    """
    Top level wrapper to get delay "histograms" : I will apply the optfct to optimize the shifts
    between curves that you got from :py:func:`pycs3.sim.draw.multidraw`, and save the results in
    form of runresult pickles.

    .. note: Remove my ".workingon" file and I will finish the current pkl and skip the remaining ones !
        This is useful to stop we cleanly.

    It is perfectly ok to launch several instances of myself on the same simset, to go faster.
    I will process every pkl of the simset only once, and prevent other instances from processing the same files.

    You can use me for a lot of different tasks. (note from VB : not to make coffee apparently)

    :param simset: The name of the simulations to run on. Those are in a directory called ``sims_name``.

    :param lcs: Lightcurves that define the initial shifts and microlensings you want to use.
        I will take the lightcurves from the simset, and put these shifts and ML on them.

    :param optset: A new name for the optimisation.
    :param kwargs_optim: dictionary. Containing the keyword argument for the optimisation function

    :param optfct: The optimizing function that takes lcs as single argument, fully optimizes the curves,
        and returns a spline, or a d2 value.
    :type optfct: function

    :param tsrand: I will randomly shift the simulated curves before running the optfct
        This randomizes the initial conditions.
        (uniform distrib from -tsrand to tsrand)

    :param shuffle: if True, I will shuffle the curves before running optc on them, and then sort them immediatly afterwards.

    :param keepopt: a bit similar to Trace, but simpler : we write the optimized lightcurves as well as the output of the optimizers into one pickle file per input pickle file.
        {"optfctoutlist":optfct_outs, "optlcslist":simlcslist}
    :param trace: boolean. To keep trace of the operation applied to tle LightCurves
    :param verbose: boolean. Verbosity.
    :param destpath: string. Path to write the optimised curves and results.
    :param use_test_seed: boolean. Used for testing purposes. If you want to impose the random seed.
    :return: dictionary containing information about which curves optimisation failed.
    """

    # We look for the sims directory
    simdir = os.path.join(destpath, "sims_%s" % simset)
    if not os.path.isdir(simdir):
        raise RuntimeError("Sorry, I cannot find the directory %s" % simset)

    simpkls = sorted(glob(os.path.join(simdir, "*.pkl")))
    if verbose:
        logger.info("I have found %i simulation pickles in %s." % (len(simpkls), simdir))

    # We prepare the destination directory
    destdir = os.path.join(destpath, "sims_%s_opt_%s" % (simset, optset))
    if verbose:
        logger.info("I'll write my results into the directory %s." % destdir)

    if not os.path.isdir(destdir):
        os.mkdir(destdir)
    else:
        if verbose:
            logger.info("(The latter already exists.)")

    # The initial conditions that I will set to the sims
    if verbose:
        logger.info("Initial conditions : ")
        for l in lcs:
            logger.info(l)

    success_dic = {'success': True, 'failed_id': [], 'error_list': []}
    for simpkl in simpkls:

        # First we test if this simpkl is already processed (or if another multirun is working on it).
        simpklfilebase = os.path.splitext(os.path.basename(simpkl))[0]

        workingonfilepath = os.path.join(destdir, simpklfilebase + ".workingon")
        resultsfilepath = os.path.join(destdir, simpklfilebase + "_runresults.pkl")
        optfilepath = os.path.join(destdir, simpklfilebase + "_opt.pkl")

        if os.path.exists(workingonfilepath) or os.path.exists(resultsfilepath):
            continue

        # Ok, we start, hence we want to avoid other instances to work on the same pkl ...
        os.system("date > %s" % workingonfilepath)

        logger.info("--- Casino running on simset %s, optset %s ---" % (simset, optset))
        simlcslist = pycs3.gen.util.readpickle(simpkl)
        logger.info("Working for %s, %i simulations." % (resultsfilepath, len(simlcslist)))

        # We set the initial conditions for the curves to analyse, based on the lcs argument as reference.

        for simlcs in simlcslist:
            pycs3.sim.draw.transfershifts(simlcs, lcs)

        # Now we add uniform noise to the initial time shifts
        if tsrand != 0.0:
            if use_test_seed:
                np.random.seed(1)
            for simlcs in simlcslist:
                for simlc in simlcs:
                    simlc.shifttime(float(np.random.uniform(low=-tsrand, high=tsrand, size=1)))
        else:
            if verbose:
                logger.info("I do NOT randomize initial contidions for the time shifts !")

        # And to the actual shifting, that will take most of the time
        optfctouts = [None] * len(simlcslist)  # reset variabl eof the loop to avoid weird error.
        clean_simlcslist = []
        sucess_dic = {'success': True, 'failed_id': [], 'error_list': []}
        if shuffle:
            for simlcs in simlcslist:
                pycs3.gen.lc_func.shuffle(simlcs)
        optfctouts, success_dic = applyopt(optfct, simlcslist, **kwargs_optim)
        if shuffle:  # We sort them, as they will be passed the constructor of runresuts.
            for simlcs in simlcslist:
                pycs3.gen.lc_func.objsort(simlcs, verbose=False)

        # If the optfct was a spline optmization, this optfctouts is a list of splines.
        # Else it might be something different, we deal with this now.
        # todo : rewrite that properly with instance comparison Rslc for regdiff and Spline for the spl

        if isinstance(optfctouts[0],
                      pycs3.gen.spl.Spline):  # then it's a spline, and we will collect these lastr2nostab values.
            tracesplinelists = [[optfctout] for optfctout in optfctouts]  # just for the trace
            qs = np.array([s.lastr2nostab for s in optfctouts])
            if np.all(qs < 1.0):
                logger.warning("### qs values are very small, did you fit that spline ? ###")

        elif isinstance(optfctouts[0], tuple):
            qs = np.array([s[1] for s in optfctouts])  # then it's regdiff which returns a tuple, I'll take the second element which corresponds to minwtv
            tracesplinelists = [[]] * len(simlcslist)  # just for the trace

        else:
            logger.error("Object : ", type(optfctouts[0]), "is unknown.")
            raise RuntimeError("Invalid instance, please optimise your curves with regdiff or spline.")

        # Trace after shifting
        if trace:
            logger.info("Saving trace of optimized curves ...")
            tracedir = "trace_sims_%s_opt_%s" % (simset, optset)
            for (simlcs, tracesplinelist) in zip(simlcslist, tracesplinelists):
                pycs3.gen.util.trace(lclist=simlcs, splist=tracesplinelist, tracedir=tracedir)

        clean_simlcslist = clean_simlist(simlcslist, success_dic)
        if keepopt:
            # A bit similar to trace, we save the optimized lcs in a pickle file.
            outopt = {"optfctoutlist": optfctouts, "optlcslist": clean_simlcslist}
            pycs3.gen.util.writepickle(outopt, optfilepath)

        # Saving the results
        rr = RunResults(clean_simlcslist, qs=qs, name="sims_%s_opt_%s" % (simset, optset), success_dic=success_dic)
        pycs3.gen.util.writepickle(rr, resultsfilepath)

        # We remove the lock for this pkl file.
        # If the files does not exist we stop !
        if not os.path.exists(workingonfilepath):
            logger.warning("WORKINGON FILE : %s REMOVED !"%workingonfilepath)
            # raise RuntimeError('Workingon file has been removed during the optimisation.')
        else:
            logger.info("REMOVING : %s !" %workingonfilepath)
            os.remove(workingonfilepath)

    return success_dic


def clean_simlist(simlcslist, success_dic):
    """
    Remove the failed optimisation according to the provided success dictionary

    :param simlcslist: list of LightCurves
    :param success_dic: dictionary returned by py:meth:`pycs3.sim.run.applyopt`

    :return: list of LightCurves where the failed optimisation has been removed
    """
    for i in reversed(success_dic['failed_id']):
        logger.info("remove simlcs ", i)
        del simlcslist[i]

    return simlcslist
