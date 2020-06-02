"""
Higher level wrappers around the multiopt optimizers.
These attempt to be a as multi-purpose as possible, but please understand that there is no general "optimal" way to optimize
a spline fit. Depending on your data, custom optimizers might be needed. That's what we did for the TDC.

In principle we do not define the microlensing inside of the optimizer, we really just optimize it.

The functions must return a final optimal spline. This spline object also contains its final r2, saved into the pkl, and used for plots etc.

"""

from pycs3.gen.lc_func import getnicetimedelays
from pycs3.gen.spl_func import fit
from pycs3.spl.multiopt import opt_magshift, opt_ml, opt_source, opt_ts_indi, redistribflux


def opt_rough(lcs, nit=5, shifttime=True, crit="r2",
              knotstep=100, stabext=300.0, stabgap=20.0, stabstep=4.0, stabmagerr=-2.0,
              verbose=True):
    """
    Getting close to the good delays, as fast as possible : no BOK (i.e. knot positions are not free), only brute force without optml.
    Indeed with optml this tends to be unstable.

    .. note:: This function is here to correct for shift errors in the order of 20 days, hence its name.

    :param nit: a number of iterations (ok to leave default : 5)
        100 is default. Lower this to eg 50 if your curve has very short term intrinsic variations, increase it if your curve is very noisy.

    """
    if verbose:
        print("Starting opt_rough on initial delays :")
        print(getnicetimedelays(lcs, separator=" | "))

    opt_magshift(lcs)

    # If there is a curve without ML, we aim at this one first
    nomllcs = [l for l in lcs if l.ml is None]
    if len(nomllcs) == 0:
        # Spline though the first curve :
        if verbose:
            print("Aiming at first curve.")
        spline = fit([lcs[0]], knotstep=knotstep, stab=True, stabext=stabext,
                     stabgap=stabgap, stabstep=stabstep, stabmagerr=stabmagerr, bokit=0, verbose=False)
    else:
        # Spline through the fixed curves :
        if verbose:
            print("Aiming at curves %s." % (", ".join([l.object for l in nomllcs])))
        spline = fit(nomllcs, knotstep=knotstep, stab=True, stabext=stabext,
                     stabgap=stabgap, stabstep=stabstep, stabmagerr=stabmagerr, bokit=0, verbose=False)

    opt_ml(lcs, spline, bokit=0, splflat=True, verbose=True)

    # Spline to go through all curves :
    spline = fit(lcs, knotstep=knotstep, stab=True, stabext=stabext,
                 stabgap=stabgap, stabstep=stabstep, stabmagerr=stabmagerr, bokit=0, verbose=False)
    opt_ml(lcs, spline, bokit=0, splflat=True, verbose=False)
    opt_source(lcs, spline, dpmethod="extadj", bokit=0, verbose=False, trace=False)

    if verbose:
        print("First spline and ML opt done.")

    for it in range(nit):
        if shifttime:
            opt_ts_indi(lcs, spline, optml=False, method="brute", crit=crit, brutestep=1.0, bruter=20, verbose=False)
        opt_source(lcs, spline, dpmethod="extadj", bokit=0, verbose=False, trace=False)
        opt_ml(lcs, spline, bokit=0, splflat=True, verbose=False)
        opt_source(lcs, spline, dpmethod="extadj", bokit=0, verbose=False, trace=False)

        if verbose:
            print("%s    (Iteration %2i, r2 = %8.1f)" % (
                getnicetimedelays(lcs, separator=" | "), it + 1, spline.lastr2nostab))

    if verbose:
        print("Rough time shifts done :")
        print("%s" % (getnicetimedelays(lcs, separator=" | ")))

    return spline


def opt_fine(lcs, spline=None, nit=10, shifttime=True, crit="r2",
             knotstep=20, stabext=300.0, stabgap=20.0, stabstep=4.0, stabmagerr=-2.0,
             bokeps=10, boktests=10, bokwindow=None,
             distribflux=False, splflat=True, verbose=True):
    """
    Fine approach, we assume that the timeshifts are within 10 days, and ML is optimized.

    Default is True for the iterations, and we release the splines only at the end.
    If you put it to False, spline are left free at any stage. This should be fine if you leave one curve without ML.

    :param spline: If this is None, I will fit my own spline, but then I expect that your lcs already overlap (i.e., ML is set) !
        If you give me a spline, I will use it (in this case I disregard your splstep setting !)

    """

    if verbose:
        print("Starting opt_fine on initial delays :")
        print(getnicetimedelays(lcs, separator=" | "))

    if spline is None:
        spline = fit(lcs, knotstep=knotstep,
                     stab=True, stabext=stabext, stabgap=stabgap, stabstep=stabstep, stabmagerr=stabmagerr,
                     bokit=2, boktests=boktests, bokwindow=bokwindow, bokeps=bokeps, verbose=False)
    opt_ml(lcs, spline, bokit=2, splflat=splflat, verbose=False)

    if verbose:
        print("Iterations :")

    for it in range(nit):

        if verbose:
            print("Start")

        if shifttime:
            opt_ts_indi(lcs, spline, optml=True, mlsplflat=splflat, method="brute", crit=crit, brutestep=0.2, bruter=10,
                        verbose=False)
            if verbose:
                print("opt_ts_indi brute done")

        opt_source(lcs, spline, dpmethod="extadj", bokit=0, verbose=False, trace=False)

        if shifttime:
            opt_ts_indi(lcs, spline, optml=True, mlsplflat=splflat, method="fmin", crit=crit, verbose=False)
            if verbose:
                print("opt_ts_indi fine done")

        opt_source(lcs, spline, dpmethod="extadj", bokit=0, verbose=False, trace=False)
        opt_ml(lcs, spline, bokit=1, splflat=splflat, verbose=False)
        if verbose:
            print("opt_ml done")

        if distribflux:
            # This works only for doubles
            if len(lcs) != 2:
                raise RuntimeError("I can only run redistribflux on double lenses !")

            redistribflux(lcs[0], lcs[1], spline, verbose=True)

        opt_source(lcs, spline, dpmethod="extadj", bokit=1, verbose=False, trace=False)
        if verbose:
            print("opt_source BOK done")

        if verbose:
            print("%s    (Iteration %2i, r2 = %8.1f)" % (
                getnicetimedelays(lcs, separator=" | "), it + 1, spline.lastr2nostab))

    if verbose:
        print("Timeshift stabilization and releasing of splflat :")

    for it in range(5):
        if shifttime:
            opt_ts_indi(lcs, spline, optml=True, mlsplflat=False, method="fmin", crit=crit,
                        verbose=False)
        opt_source(lcs, spline, dpmethod="extadj", bokit=0, verbose=False, trace=False)
        if verbose:
            print("%s    (Iteration %2i, r2 = %8.1f)" % (
                getnicetimedelays(lcs, separator=" | "), it + 1, spline.lastr2nostab))

    return spline
