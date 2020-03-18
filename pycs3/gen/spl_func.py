"""
Module containing the function related to Spline operation
"""

from pycs3.gen.spl import Spline
from pycs3.gen.datapoints import DataPoints
import scipy.optimize as spopt
import numpy as np

def merge(lcs, olddp=None, splitup=True, deltat=0.000001, sort=True, stab=False,
          stabext=300.0, stabgap=30.0, stabstep=5.0, stabmagerr=2.0, stabrampsize=0, stabrampfact=1.0):
    """
    Factory function for DataPoints objects, starting from lightcurves.
    Takes a list of lightcurves and quickly concatenate the jds, mags, and magerrs.

    Instead of specifying all the stab point parameters, you can give me an old datapoints object,
    and I will reuse its settings... This is useful if you want to "update" the data points.

    If overlap is True, I will keep only points that are "covered" by all four lightcurves !
    This is useful when you want to build a first source spline, and your microlensing is messy at the borders.
    NOT YET IMPLEMENTED ...
    """

    jds = np.concatenate([l.getjds() for l in lcs])
    mags = np.concatenate([l.getmags() for l in lcs])
    magerrs = np.concatenate([l.getmagerrs() for l in lcs])

    if olddp is None:
        return DataPoints(jds, mags, magerrs, splitup=splitup, deltat=deltat, sort=sort,
                          stab=stab, stabext=stabext, stabgap=stabgap, stabstep=stabstep, stabmagerr=stabmagerr,
                          stabrampsize=stabrampsize, stabrampfact=stabrampfact)
    else:
        return DataPoints(jds, mags, magerrs, splitup=splitup, sort=sort,
                          deltat=olddp.deltat,
                          stab=olddp.stab, stabext=olddp.stabext, stabgap=olddp.stabgap, stabstep=olddp.stabstep,
                          stabmagerr=olddp.stabmagerr,
                          stabrampsize=olddp.stabrampsize, stabrampfact=olddp.stabrampfact)


def fit(lcs, knotstep=20.0, n=None, knots=None, stab=True,
        stabext=300.0, stabgap=20.0, stabstep=5.0, stabmagerr=-2.0, stabrampsize=0, stabrampfact=1.0,
        bokit=1, bokeps=2.0, boktests=5, bokwindow=None, k=3, verbose=True):
    """
    The highlevel function to make a spline fit.

    lcs : a list of lightcurves (I will fit the spline through the merged curves)

    Specify either
    knotstep : spacing of knots
    or
    n : how many knots to place
    or
    knots : give me actual initial knot locations, for instance prepared by seasonknots.

    stab : do you want to insert stabilization points ?
    stabext : number of days to the left and right to fill with stabilization points
    stabgap : interval of days considered as a gap to fill with stab points.
    stabstep : step of stab points
    stabmagerr : if negative, absolute mag err of stab points. If positive, the error bar will be stabmagerr times the median error bar of the data points.


    bokit : number of BOK iterations (put to 0 to not move knots)
    bokeps : epsilon of BOK
    boktests : number of test positions for each knot


    """
    dp = merge(lcs, stab=stab, stabext=stabext, stabgap=stabgap, stabstep=stabstep, stabmagerr=stabmagerr,
               stabrampsize=stabrampsize, stabrampfact=stabrampfact)

    s = Spline(dp, k=k, bokeps=bokeps, boktests=boktests, bokwindow=bokwindow)

    if knots is None:
        if n is None:
            s.uniknots(nint=knotstep, n=False)
        else:
            s.uniknots(nint=n, n=True)
    else:
        s.setintt(knots)

    for n in range(bokit):
        s.buildbounds(verbose=verbose)
        s.bok(bokmethod="BF", verbose=verbose)

    s.optc()
    s.r2(nostab=True)  # This is to set s.lastr2nostab

    return s


def seasonknots(lcs, knotstep, ingap, seasongap=60.0):
    """
    A little helper to get some knot locations inside of seasons only

    knotstep is for inside seasons
    ingap is the number of knots inside gaps.

    """
    knots = []

    # knotstep = 10
    dp = merge(lcs, splitup=True, deltat=0.000001, sort=True, stab=False)

    gaps = dp.jds[1:] - dp.jds[:-1]
    gapindices = list(np.arange(len(dp.jds) - 1)[gaps > seasongap])

    # knots inside of seasons :
    a = dp.jds[0]
    for gapi in gapindices:
        b = dp.jds[gapi]
        # print (a, b)
        knots.append(np.linspace(a, b, float(b - a) / float(knotstep)))
        a = dp.jds[gapi + 1]
    b = dp.jds[-1]
    knots.append(np.linspace(a, b, float(b - a) / float(knotstep)))

    # knots inside of gaps
    for gapi in gapindices:
        a = dp.jds[gapi]
        b = dp.jds[gapi + 1]
        knots.append(np.linspace(a, b, ingap + 2)[1:-1])

    knots = np.concatenate(knots)
    knots.sort()
    return knots


def r2(lcs, spline, nosquare=False):
    """
    I do not modify the spline (not even its datapoints) !
    Just evaluate the quality of the match, returning an r2 (without any stab points, of course).

    This is used if you want to optimize something on the lightcurves without touching the spline.

    Of course, I do not touch lastr2nostab or lastr2stab of the spline ! So this has really nothing
    to do with source spline optimization !


    """

    myspline = spline.copy()
    newdp = merge(lcs, stab=False)  # Indeed we do not care about stabilization points here.
    myspline.updatedp(newdp, dpmethod="leave")
    return myspline.r2(nostab=True, nosquare=nosquare)


def mltv(lcs, spline, weight=True):
    """
    Calculates the TV norm of the difference between a lightcurve (disregarding any microlensing !) and the spline.
    I return the sum over the curves in lcs.

    Also returns a abs(chi) like distance between the lcs without ML and the spline

    If weight is True, we weight the terms in sums according to their error bars.

    Idea : weight the total variation somehow by the error bars ! Not sure if needed, the spline is already weighted.
    """

    tv = 0.0
    dist = 0.0
    for l in lcs:

        # We have a spline, and a lightcurve
        lmags = l.getmags(noml=True)  # We get the mags without ML (but with mag and fluxshift !)
        ljds = l.getjds()  # Including any time shifts.

        # Evaluating the spline at those jds :
        splinemags = spline.eval(ljds)

        # The residues :
        res = lmags - splinemags

        # plt.plot(ljds, res, "r.")
        # plt.show()

        if not weight:
            tv += np.sum(np.fabs(res[1:] - res[:-1]))
            dist += np.sum(np.fabs(res))

        else:
            magerrs = l.getmagerrs()
            a = res[1:]
            aerrs = magerrs[1:]
            b = res[:-1]
            berrs = magerrs[:-1]

            vari = np.fabs(a - b)
            varierrs = np.sqrt(aerrs * aerrs + berrs * berrs)

            tv += np.sum(vari / varierrs)
            dist += np.sum(np.fabs(res) / np.fabs(magerrs))

    return tv, dist


def optcmltv(lcs, spline, verbose=True):
    """
    I will optimize the coefficients of the spline so to minimize the mltv.
    I do not use the microlensing of the lcs at all !

    Simple powell optimization, slow. A pity.

    Add BOK and time shifts in there and it might be bingo !

    Would be more efficient if we add knots on the fly
    """

    inic = spline.getc(m=2)

    def setc(c):
        spline.setc(c, m=2)

    def errorfct(c):
        setc(c)
        (tv, dist) = mltv(lcs, spline, weight=False)
        print("put weight")
        return tv + 0.1 * spline.tv()

    minout = spopt.fmin_powell(errorfct, inic, full_output=True, disp=verbose)
    copt = minout[0]

    # We find a common shift to all coeffs so that the level matches

    meanc = np.mean(spline.getc(m=2))
    meanmag = np.mean(np.concatenate([l.getmags(noml=True) for l in lcs]))

    setc(copt)
    spline.c += meanmag - meanc
