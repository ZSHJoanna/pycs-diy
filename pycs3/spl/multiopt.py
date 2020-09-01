"""
Functions to optimize time shifts and microlensing between lcs, using spline fits.


By default the functions don't touch the ML and sourcespline *knots*, it's up to you to enable BOK iterations.
And check the results by eye ...

Typical order of operations : 

Put smooth microlensing on the curves
Fit a source to the curve without microlensing
opt_magshift (does not use the source)
opt_ml (does not change the source)
Fit a new source, to all the curves
Put a less smooth microlensing on them
opt_ml_source


"""
import numpy as np
import pycs3.gen.util
import scipy.optimize as spopt
from pycs3.gen.polyml import polyfit
from pycs3.gen.spl_func import r2, mltv, merge
import logging
logger = logging.getLogger(__name__)


def opt_magshift(lcs, sourcespline=None, verbose=False, trace=False, tracedir='trace'):
    """
    If you don't give any sourcespline, this is a dirty rough magshift optimization,
    using the median mag level (without microlensing), once for all.
    We don't touch the magshift of the first curve.

    If you do give a sourcespline, I'll optimize the magshift of each curve one by one to match the spline.
    New : I even touch the magshift of the first curve.

    We use the l1-norm of the residues, not a usual r2, to avoid local minima. Important !
    This is done with the nosquare=True option to the call of r2 !

    :param lcs: list of LightCurves
    :type lcs: list
    :param sourcespline: Spline to be matched
    :type sourcespline: Spline
    :param verbose: verbosity
    :type verbose: bool
    :param trace: to keep a trace of the operation
    :type trace: bool
    :param tracedir: directory to save the trace
    :type tracedir: str

    """
    if sourcespline is None:

        reflevel = np.median(lcs[0].getmags(noml=True))
        for l in lcs[1:]:
            level = np.median(l.getmags(noml=True))
            l.shiftmag(reflevel - level)
            if trace:
                pycs3.gen.util.trace(lcs, tracedir=tracedir)
        if verbose:
            logger.info("Magshift optimization done.")

    else:

        # for l in lcs[1:]: # We don't touch the first one.
        for l in lcs:

            if verbose:
                logger.info("Magshift optimization on %s ..." % l)
            inip = l.magshift

            def setp(p):
                l.magshift = p[0]

            def errorfct(p):
                setp(p)
                return r2([l], sourcespline, nosquare=True)

            minout = spopt.fmin(errorfct, inip, full_output=True, xtol=0.001, disp=verbose)
            popt = minout[0]
            if verbose:
                logger.info("Optimal magshift: %.4f" % popt)
            setp(popt)
            if verbose:
                logger.info("Magshift optimization of %s done." % l)


def opt_source(lcs, sourcespline, dpmethod="extadj", bokit=0, bokmethod="BF", verbose=True, trace=False, tracedir = 'trace'):
    """
    Optimise just the source spline, without touching the MicroLensing of the lcs.
    At each call, I update the sourcespline with the merged lcs.
    The internal knots of the sourcespline stay where they are, only the external ones are adjusted.

    :param lcs: list of LightCurves
    :type lcs: list
    :param sourcespline: Spline to be matched
    :type sourcespline: Spline
    :param dpmethod: optimisation method
    :type dpmethod: str
    :param bokit: number of iteration to build the spline
    :type bokit: int
    :param bokmethod: method to optimise the knots position
    :type bokmethod: str
    :param verbose: Verbosity
    :type verbose: bool
    :param trace: to keep a trace of the operation
    :type trace: bool
    :param tracedir: directory to save the trace
    :type tracedir: str
    :return: float, final r2 of the fit.
    """

    inir2 = sourcespline.r2(nostab=True)
    if verbose:
        logger.info("Starting source optimization ...")
        logger.info("Initial r2 (before dp update) : %f" % inir2)

    dp = merge(lcs, olddp=sourcespline.datapoints)
    sourcespline.updatedp(dp, dpmethod=dpmethod)

    for n in range(bokit):
        sourcespline.buildbounds(verbose=verbose)
        finalr2 = sourcespline.bok(bokmethod=bokmethod, verbose=verbose)

    if bokit == 0:  # otherwise this is already done by bok.
        sourcespline.optc()
        finalr2 = sourcespline.r2(nostab=True)  # Important, to set sourcesplie.lastr2nostab

    if trace:
        pycs3.gen.util.trace(lcs, [sourcespline], tracedir=tracedir)
    if verbose:
        logger.info("Final r2 : %f" % finalr2)
    return finalr2


def opt_fluxshift(lcs, sourcespline, verbose=True):
    """
    Optimizes the flux shift and the magshift of the lcs (not the first one)
    to get the best fit to the "sourcespline". Does not touch the microlensing, nor the spline.
    So this is a building block to be used iteratively with the other optimizers.
    Especially of the sourcespline, as we fit here even on regions not well constrained by the spline !

    The spline should typically well fit to the first curve.

    :param lcs: list of LightCurves
    :type lcs: list
    :param sourcespline: Spline to fit the flux to.
    :type sourcespline: Spline
    :param verbose: Verbosity
    :type verbose: bool
    """

    for l in lcs[1:]:  # We don't touch the first one.

        if verbose:
            logger.info("Fluxshift optimization on %s ..." % l)

        minfs = l.getminfluxshift()
        inip = (0, 0)

        def setp(p):
            fs = p[0] * 1000.0
            ms = p[1] * 0.1

            if fs < minfs:
                l.setfluxshift(minfs, consmag=False)
            else:
                l.setfluxshift(fs, consmag=False)

            l.magshift = ms

        def errorfct(p):
            setp(p)
            return r2([l], sourcespline)

        minout = spopt.fmin_powell(errorfct, inip, full_output=1, xtol=0.001, disp=verbose)

        popt = minout[0]
        setp(popt)
        if verbose:
            logger.info("Done with %s ..." % l)


def opt_ml(lcs, sourcespline, bokit=0, bokmethod="BF", splflat=False, verbose=True, trace=False, tracedir='trace'):
    """
    Optimizes the microlensing of the lcs (one curve after the other) so that they fit to the spline.
    I work with both polynomial and spline microlensing.
    For spline microlensing, I can do BOK iterations to move the knots.

    .. note:: Does not touch the sourcespline  at all !

    But this it what makes the problem linear (except for the BOK iterations) for both splines and polynomial ML, and
    thus fast !

    Parameters for spline ML :

    :param lcs: list of LightCurve
    :type lcs: list
    :param sourcespline: source Spline object
    :type sourcespline: Spline
    :param bokit: number of iteration to build the BOK.
    :type bokit: int
    :param bokmethod: Choose among :

        - MCBF : Monte Carlo brute force with ntestpos trial positions for each knot
        - BF : brute force, deterministic. Call me twice
        - fminind : fminbound on one knot after the other.
        - fmin : global fminbound

    :type bokmethod: str
    :param splflat: if you want to optimise only the border coefficient after a first optimisation
    :type splflat: bool
    :param verbose: verbosity
    :type verbose: bool
    :param trace: trace all the operation applied to the LightCurve
    :type trace: bool
    :param tracedir: directory to save the trace
    :type tracedir: str

    Parameters for poly ML :
    None ! We just to a linear weighted least squares on each season !
    So for poly ML, all the params above are not used at all.
    We do not return anything. Returning a r2 would make no sense, as we do not touch the sourcepline !

    """

    if trace:
        pycs3.gen.util.trace(lcs, [sourcespline], tracedir=tracedir)

    if verbose:
        logger.info("Starting ML optimization ...")

    for l in lcs:
        if (l.ml is not None) and (l.ml.mltype == "spline"):
            # So this is spline microlensing

            if verbose:
                logger.info("Working on the spline ML of %s" % l)
            l.ml.settargetmags(l, sourcespline)

            for n in range(bokit):
                l.ml.spline.buildbounds(verbose=verbose)
                l.ml.spline.bok(bokmethod=bokmethod, verbose=verbose)

            if splflat:
                l.ml.spline.optc()
                l.ml.spline.optcflat(verbose=False)
            else:
                l.ml.spline.optc()
            if trace:
                pycs3.gen.util.trace(lcs, [sourcespline],tracedir=tracedir)

        if (l.ml is not None) and (l.ml.mltype == "poly"):

            if verbose:
                logger.info("Working on the poly ML of %s" % l)

            # We go through the curve season by season :
            for m in l.ml.mllist:
                nparams = m.nfree

                mlseasjds = l.jds[m.season.indices]
                mlseasjds -= np.mean(mlseasjds)  # Convention for polyml, jds are "centered".
                nomlmags = l.getmags(noml=True)[m.season.indices]
                magerrs = l.magerrs[m.season.indices]

                absjds = l.getjds()[m.season.indices]
                targetmags = sourcespline.eval(absjds)

                polyparams = polyfit(mlseasjds, targetmags - nomlmags, magerrs, nparams)

                m.setparams(polyparams)
    if verbose:
        logger.info("Done !")


def redistribflux(lc1, lc2, sourcespline, verbose=True, maxfrac=0.2):
    """
    Redistributes flux between lc1 and lc2 (assuming these curves suffer form flux sharing), so
    to minimize the r2 with respect to the sourcespline.
    I do not touch the sourcespline, but I do modify your curves in an irreversible way !

    :param lc1: a lightcurve
    :type lc1: LightCurve
    :param lc2: another lightcurve
    :type lc2: LightCurve
    :param sourcespline: the spline that the curves should try to fit to
    :type sourcespline: Spline
    :param verbose: verbosity
    :type verbose: bool
    :param maxfrac: fraction of the maxium amplitude to set the optimisation bound
    :type maxfrac: float
    """
    if not np.all(lc1.jds == lc2.jds): # pragma: no cover
        raise RuntimeError("I do only work on curves with identical jds !")

    if verbose:
        logger.info("Starting redistrib_flux, r2 = %10.2f" % (r2([lc1, lc2], sourcespline)))

    # The initial curves :
    lc1fluxes = lc1.getrawfluxes()
    lc2fluxes = lc2.getrawfluxes()

    maxamp = min(np.min(lc1fluxes), np.min(lc2fluxes))  # maximum amplitute of correction

    def setp(flux, ind):  # flux is an absolute shift in flux for point i
        lc1.mags[ind] = -2.5 * np.log10(lc1fluxes[ind] + flux)
        lc2.mags[ind] = -2.5 * np.log10(lc2fluxes[ind] - flux)

    def errorfct(flux, ind):
        setp(flux, ind)
        return r2([lc1, lc2], sourcespline)

    # We can do this one point at a time ...
    for i in range(len(lc1)):
        out = spopt.optimize.fminbound(errorfct, -maxfrac * maxamp, maxfrac * maxamp, args=(i,), xtol=0.1, disp=True,
                                       full_output=False)
        setp(out, i)

    if verbose:
        logger.info("Done with redistrib,     r2 = %10.2f" % (r2([lc1, lc2], sourcespline)))

def opt_ts_indi(lcs, sourcespline, method="fmin", crit="r2", optml=False, mlsplflat=False, brutestep=1.0, bruter=5,
                verbose=True):
    """
    We shift the curves one by one so that they match to the spline, using fmin or brute force.
    A bit special : I do not touch the spline at all ! Hence I return no r2.
    Idea is to have a fast ts optimization building block.

    The spline should be a shape common to the joined lcs.
    No need to work on copies, as we do not change the ML or spline *iteratively*, but
    only the ML -> nothing can go wrong.

    :param lcs: list of LightCurve
    :type lcs: list
    :param sourcespline: source Spline object
    :type sourcespline: Spline
    :param method: Choose between "fmin" for the scipy.optimize.fmin() algorithm and "brute" for brute force search.
    :type method: str
    :param crit: fitting metric
    :type crit: str
    :param optml: if you want to also optimise the microlensing
    :type optml: bool
    :param mlsplflat: if you want to optimise only the border coefficient after a first optimisation
    :type mlsplflat: bool
    :param brutestep: step size, in days, Used if ``method`` is "brute"
    :type brutestep: float
    :param bruter: radius in number of steps
    :type bruter: int
    :param verbose: verbosity
    :type verbose: bool

    """

    for l in lcs:

        def errorfct(timeshift):
            # Set the shift :
            l.timeshift = timeshift
            # Optimize the ML for this shift :
            if optml:
                opt_ml([l], sourcespline, bokit=0, splflat=mlsplflat, verbose=False)
            # Calculate r2 without touching the spline :
            if crit == "r2":
                error = r2([l], sourcespline)
            elif crit == "tv":
                error = mltv([l], sourcespline)[0]
                logger.warning("Warning, using TV !")
            if verbose:
                logger.info("%s %10.3f %10.3f" % (l.object, l.timeshift, error))
            return error

        initimeshift = l.timeshift

        if method == "fmin":
            out = spopt.fmin(errorfct, initimeshift, xtol=0.1, ftol=0.1, maxiter=None, maxfun=100, full_output=True,
                             disp=verbose)
            opttimeshift = float(out[0][0])
        elif method == "brute":

            testvals = np.linspace(-1.0 * bruter * brutestep, 1.0 * bruter * brutestep, int(2 * bruter + 1)) + initimeshift
            r2vals = np.array(list(map(errorfct, testvals)))
            minindex = np.argmin(r2vals)
            opttimeshift = float(testvals[minindex])

        l.timeshift = opttimeshift
