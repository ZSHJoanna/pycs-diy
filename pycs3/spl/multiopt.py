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


def opt_magshift(lcs, sourcespline=None, verbose=False, trace=False):
    """
    If you don't give any sourcespline, this is a dirty rough magshift optimization,
    using the median mag level (without microlensing), once for all.
    We don't touch the magshift of the first curve.

    If you do give a sourcespline, I'll optimize the magshift of each curve one by one to match the spline.
    New : I even touch the magshift of the first curve.

    We use the l1-norm of the residues, not a usual r2, to avoid local minima. Important !
    This is done with the nosquare=True option to the call of r2 !
    """

    if sourcespline is None:

        reflevel = np.median(lcs[0].getmags(noml=True))
        for l in lcs[1:]:
            level = np.median(l.getmags(noml=True))
            l.shiftmag(reflevel - level)
            if trace:
                pycs3.gen.util.trace(lcs)
        if verbose:
            print("Magshift optimization done.")

    else:

        # for l in lcs[1:]: # We don't touch the first one.
        for l in lcs:

            if verbose:
                print("Magshift optimization on %s ..." % l)
            inip = l.magshift

            def setp(p):
                l.magshift = p

            def errorfct(p):
                setp(p)
                return r2([l], sourcespline, nosquare=True)

            minout = spopt.fmin(errorfct, inip, full_output=1, xtol=0.001, disp=verbose)
            popt = minout[0]
            if verbose:
                print("Optimal magshift: %.4f" % popt)
            setp(popt)
            if verbose:
                print("Magshift optimization of %s done." % l)


def opt_source(lcs, sourcespline, dpmethod="extadj", bokit=0, bokmethod="BF", verbose=True, trace=False):
    """
    Just the source spline, without touching the ML of the lcs.
    At each call, I update the sourcespline with the merged lcs.
    The internal knots of the sourcespline stay where they are, only the external ones are adjusted.
    """

    inir2 = sourcespline.r2(nostab=True)
    if verbose:
        print("Starting source optimization ...")
        print("Initial r2 (before dp update) : %f" % inir2)

    dp = merge(lcs, olddp=sourcespline.datapoints)
    sourcespline.updatedp(dp, dpmethod=dpmethod)

    for n in range(bokit):
        sourcespline.buildbounds(verbose=verbose)
        finalr2 = sourcespline.bok(bokmethod=bokmethod, verbose=verbose)

    if bokit == 0:  # otherwise this is already done by bok.
        sourcespline.optc()
        finalr2 = sourcespline.r2(nostab=True)  # Important, to set sourcesplie.lastr2nostab

    if trace:
        pycs3.gen.util.trace(lcs, [sourcespline])
    if verbose:
        print("Final r2 : %f" % finalr2)
    return finalr2


def opt_fluxshift(lcs, sourcespline, verbose=True):
    """
    Optimizes the flux shift and the magshift of the lcs (not the first one)
    to get the best fit to the "sourcespline". Does not touch the microlensing, nor the spline.
    So this is a building block to be used iteratively with the other optimizers.
    Especially of the sourcespline, as we fit here even on regions not well constrained by the spline !

    The spline should typically well fit to the first curve.
    """

    for l in lcs[1:]:  # We don't touch the first one.

        if verbose:
            print("Fluxshift optimization on %s ..." % l)

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
            print("Done with %s ..." % l)


def opt_ml(lcs, sourcespline, bokit=0, bokmethod="BF", splflat=False, verbose=True, trace=False):
    """
    Optimizes the microlensing of the lcs (one curve after the other) so that they fit to the spline.
    I work with both polynomial and spline microlensing.
    For spline microlensing, I can do BOK iterations to move the knots.

    .. note:: Does not touch the sourcespline  at all !

    But this it what makes the problem linear (except for the BOK iterations) for both splines and polynomial ML, and
    thus fast !

    Parameters for spline ML :
    :param lcs : list of LightCurve
    :param sourcespline: source Spline object
    :param bokit: integer, number of iteration to build the BOK.
    :param bokmethod : string
        - MCBF : Monte Carlo brute force with ntestpos trial positions for each knot
        - BF : brute force, deterministic. Call me twice
        - fminind : fminbound on one knot after the other.
        - fmin :global fminbound
    :param splflat: boolean, if you want to optimise only the border coefficient after a first optimisation
    :param verbose: boolean, verbosity
    :param trace: boolean, trace all the operation applied to the LightCurve

    Parameters for poly ML :

    None ! We just to a linear weighted least squares on each season !
    So for poly ML, all the params above are not used at all.


    We do not return anything. Returning a r2 would make no sense, as we do not touch the sourcepline !

    """

    if trace:
        pycs3.gen.util.trace(lcs, [sourcespline])

    if verbose:
        print("Starting ML optimization ...")

    for l in lcs:
        if (l.ml is not None) and (l.ml.mltype == "spline"):
            # So this is spline microlensing

            if verbose:
                print("Working on the spline ML of %s" % l)
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
                pycs3.gen.util.trace(lcs, [sourcespline])

        if (l.ml is not None) and (l.ml.mltype == "poly"):

            if verbose:
                print("Working on the poly ML of %s" % l)

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
        print("Done !")


def redistribflux(lc1, lc2, sourcespline, verbose=True, maxfrac=0.2):
    """
    Redistributes flux between lc1 and lc2 (assuming these curves suffer form flux sharing), so
    to minimize the r2 with respect to the sourcespline.
    I do not touch the sourcespline, but I do modify your curves in an irreversible way !

    :param lc1: a lightcurve
    :param lc2: another lightcurve
    :param sourcespline: the spline that the curves should try to fit to
    :param verbose: boolean, verbosity
    :param maxfrac: float, fraction of the maxium amplitude to set the optimisation bound

    """
    if not np.all(lc1.jds == lc2.jds):
        raise RuntimeError("I do only work on curves with identical jds !")

    if verbose:
        print("Starting redistrib_flux, r2 = %10.2f" % (r2([lc1, lc2], sourcespline)))

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
        print("Done with redistrib,     r2 = %10.2f" % (r2([lc1, lc2], sourcespline)))


def opt_ts_powell(lcs, sourcespline, optml=True, movefirst=False, verbose=True, trace=False):
    """

    If you use this again, implement mlsplflat for optml, might be important !

    Optimize the timeshifts of all four lightcurves AND the spline coeffs (but not the knots) so to get the best possible fit.

    If optml is True, I will optimize the ml for every trial delays (and then optimize the source again).
    For this I will always start from the initial settings, I don't cumulate the ML - source optimizations.
    This is required as ML - source optimizations are degenerate.
    If optml is false, this would not be needed, as the spline fitting is unique anyway.

    Note that even if I don't modify the sourcespline knots, I do modify the sourcespline datapoints as the time shifts move !

    Improvement ideas :
    see if better to not optimize the first curves time shift ?
    Don't try to optimize the timeshifts at the same time ?
    Do one after the other, with respect to the first curve ?
    Special function that takes a spline fit to the first curve as argument, to do this ?

    -> Yes, see opt_ts_indi below !

    """
    # To start each time shift guess from the same conditions, we will internally work on copies.
    origlcs = [l.copy() for l in lcs]
    origsourcespline = sourcespline.copy()

    def errorfct(p):  # p are the absolute time shifts of the n or n-1 curves
        """
        Warning : I work only on internal copies, and do not set anything on lcs or sourcespline
        If you change me, also change the apply() below !
        """

        # We make internal copies :
        mylcs = [l.copy() for l in origlcs]
        mysourcespline = origsourcespline.copy()

        # We set p of these internal copies :
        # (We do an absolute update of timeshift)
        if p.shape == ():
            p = np.array([p])
        if movefirst:  # this should be changed to gen.lc.gettimeshifts ...
            for (l, shift) in zip(mylcs, p):
                l.timeshift = shift
        else:
            for (l, shift) in zip(mylcs[1:], p):
                l.timeshift = shift
        # We do a first source opt :
        fr2 = opt_source(mylcs, mysourcespline, verbose=False, trace=trace)
        # And then ML, and source once again :
        if optml:
            opt_ml(mylcs, mysourcespline, bokit=0, verbose=False)
            fr2 = opt_source(mylcs, mysourcespline, verbose=False, trace=False)

        return fr2

    def apply(p):
        """
        Does the same as errorfct, but on the real objects, not copies !
        """
        if p.shape == ():
            p = np.array([p])
        if movefirst:
            for (l, shift) in zip(lcs, p):
                l.timeshift = shift
        else:
            for (l, shift) in zip(lcs[1:], p):
                l.timeshift = shift

        fr2 = opt_source(lcs, sourcespline, verbose=False, trace=trace)
        if optml:
            opt_ml(lcs, sourcespline, bokit=0, verbose=False)
            fr2 = opt_source(lcs, sourcespline, verbose=False, trace=False)
        # These calls to opt_source will set the sourcepline.lastr2nostab !
        return fr2

    if movefirst:
        inip = np.array([l.timeshift for l in lcs])
    else:
        inip = np.array([l.timeshift for l in lcs[1:]])

    inir2 = errorfct(inip)

    if verbose:
        print("Starting time shift optimization ...")
        print("Initial pars : ", inip)
        print("Initial r2 : %f" % inir2)

    minout = spopt.fmin_powell(errorfct, inip, full_output=1, xtol=0.001, disp=verbose)
    popt = minout[0]

    finalr2 = apply(popt)  # This sets popt, and the optimal ML and source.

    if verbose:
        print("Final r2 : %f" % r2)
        print("Optimal pars : ", popt)

    return finalr2


def comb(*sequences):
    """
    http://code.activestate.com/recipes/502199/
    combinations of multiple sequences so you don't have
    to write nested for loops

    >>> from pprint import pprint as pp
    >>> pp(comb(['Guido','Larry'], ['knows','loves'], ['Phyton','Purl']))
    [['Guido', 'knows', 'Phyton'],
    ['Guido', 'knows', 'Purl'],
    ['Guido', 'loves', 'Phyton'],
    ['Guido', 'loves', 'Purl'],
    ['Larry', 'knows', 'Phyton'],
    ['Larry', 'knows', 'Purl'],
    ['Larry', 'loves', 'Phyton'],
    ['Larry', 'loves', 'Purl']]
    >>>
    """
    combinations = [[seq] for seq in sequences[0]]
    for seq in sequences[1:]:
        combinations = [cb + [item]
                        for cb in combinations
                        for item in seq]
    return combinations


def opt_ts_brute(lcs, sourcespline, movefirst=True, optml=False, r=2, step=1.0, verbose=True, trace=False):
    """

    If you use this again, implement mlsplflat, might be important.

    Given the current delays, I will explore a hypercube (r steps in each direction on each axis)
    of possible time shift combinations. And choose the shifts that gave the smallest chi2.

    For each trial shift, I optimize the sourcespline to fit all curves, and optionally also the ML of every curve.

    This is very slow, not really used.

    """
    origlcs = [l.copy() for l in lcs]
    origsourcespline = sourcespline.copy()

    if movefirst:
        origshifts = np.array([l.timeshift for l in lcs])
    else:
        origshifts = np.array([l.timeshift for l in lcs[1:]])

    relrange = np.linspace(-1.0 * r * step, 1.0 * r * step, int(2 * r + 1))
    absrangelist = [list(relrange + origshift) for origshift in origshifts]

    # We want a list of combinations to explore. Its length is (2*r + 1)**len(lcs)
    combilist = comb(*absrangelist)
    if movefirst:
        assert len(combilist) == (2 * r + 1) ** len(lcs)
    else:
        assert len(combilist) == (2 * r + 1) ** (len(lcs) - 1)

    combir2s = np.zeros(len(combilist))
    for i, absshifts in enumerate(combilist):
        if verbose:
            print("%i / %i" % (i + 1, len(combilist)))

        # We make internal copies :
        mylcs = [l.copy() for l in origlcs]
        mysourcespline = origsourcespline.copy()

        # We set the shifts of these internal copies :

        if movefirst:
            for (l, shift) in zip(mylcs, absshifts):
                l.timeshift = shift
        else:
            for (l, shift) in zip(mylcs[1:], absshifts):
                l.timeshift = shift
        # We do a first source opt :
        finalr2 = opt_source(mylcs, mysourcespline, verbose=False, trace=trace)
        # And then ML, and source once again :
        if optml:
            opt_ml(mylcs, mysourcespline, bokit=0, verbose=False)
            finalr2 = opt_source(mylcs, mysourcespline, verbose=False, trace=False)

        # And save the r2 :
        combir2s[i] = finalr2

    # Find the min
    minindex = np.argmin(combir2s)
    optabshifts = combilist[minindex]
    if verbose:
        print("Best shift at index %i :" % minindex)
        print(optabshifts)
    # And set the actual curves :
    if movefirst:
        for (l, shift) in zip(lcs, optabshifts):
            l.timeshift = shift
    else:
        for (l, shift) in zip(lcs[1:], optabshifts):
            l.timeshift = shift
    finalr2 = opt_source(lcs, sourcespline, verbose=False, trace=trace)
    if optml:
        opt_ml(lcs, sourcespline, bokit=0, verbose=False)
        finalr2 = opt_source(lcs, sourcespline, verbose=False, trace=False)

    return finalr2


def opt_ts_indi(lcs, sourcespline, method="fmin", crit="r2", optml=False, mlsplflat=False, brutestep=1.0, bruter=5,
                verbose=True):
    """
    We shift the curves one by one so that they match to the spline, using fmin or brute force.
    A bit special : I do not touch the spline at all ! Hence I return no r2.
    Idea is to have a fast ts optimization building block.

    The spline should be a shape common to the joined lcs.
    No need to work on copies, as we do not change the ML or spline *iteratively*, but
    only the ML -> nothing can go wrong.
    :param lcs : list of LightCurve
    :param sourcespline : Spline
    :param method :
    :param crit:
    :param optml:
    :param mlsplflat:
    :param brutestep: step size, in days
    :param bruter: radius in number of steps
    :param verbose: boolean, verbosity

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
            # print "Still using r2 !"
            elif crit == "tv":
                error = mltv([l], sourcespline)[0]
                print("Warning, using TV !")
            if verbose:
                print("%s %10.3f %10.3f" % (l.object, l.timeshift, error))
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
