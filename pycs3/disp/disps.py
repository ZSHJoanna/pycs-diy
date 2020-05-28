"""
Defines dispersion functions. No optimization is done here !
We just calculate the dispersion.

* linintnp is the "default" one
* pelt95
* others should be ported into pycs from my f90 code :-(

They all share a common structure : give me 2 light curves as input, and you get a float describing the match.
By construction the dispersion is often not symmetric, i.e. if you switch lc1 and lc2 you will get a different result.
To symmetrize the techniques, use the provided symmetrize function.
I return a dictionnary, containing a field "d2" that gives the dispersion (among possible other fields).
Any microlensing is naturally taken into account, if present, simply using the getmags() method.

"""

import matplotlib.pyplot as plt
import numpy as np


def linintnp(lc1, lc2, interpdist=30.0, weights=True, usemask=True, plot=False):
    """
    This is the default one, fast implementation without loops.
    If usemask == True, it is a bit slower...
    If you do not change the mask, it's better to cutmask() it, then use usemask = False.
    """
    if usemask:
        lc1jds = lc1.getjds()[lc1.mask]  # lc1 is the "reference" curve
        lc2jds = lc2.getjds()[lc2.mask]  # lc2 is the curve that will get interpolated.
        lc1mags = lc1.getmags()[lc1.mask]
        lc2mags = lc2.getmags()[lc2.mask]
        lc1magerrs = lc1.magerrs[lc1.mask]
        lc2magerrs = lc2.magerrs[lc2.mask]
    else:
        lc1jds = lc1.getjds()  # lc1 is the "reference" curve
        lc2jds = lc2.getjds()  # lc2 is the curve that will get interpolated.
        lc1mags = lc1.getmags()
        lc2mags = lc2.getmags()
        lc1magerrs = lc1.magerrs
        lc2magerrs = lc2.magerrs

    # --- Mask creation : which points of lc1 can be taken into account ? ---

    # double-sampled version #

    interpdist = interpdist / 2.0

    # We double-sample the lc2jds:
    add_points = np.ediff1d(lc2jds) / 2.0 + lc2jds[:-1]
    lc2jdsa = np.sort(np.concatenate([lc2jds, add_points]))  # make this faster

    # We want to create of mask telling us which jds of lc1 can be interpolated on lc2,
    # without doing any python loop. We will build an auxiliary array, then interpolate it.
    lc2interpcode_lr = np.cast['int8'](
        np.ediff1d(lc2jdsa, to_end=interpdist + 1, to_begin=interpdist + 1) <= interpdist)  # len lc2 + 1
    lc2interpcode_sum = lc2interpcode_lr[1:] + lc2interpcode_lr[:-1]  # len lc2
    '''
    This last array contains ints :
        2 if both right and left distances are ok
        1 if only one of them is ok
        0 if both are two fare away (i.e. fully isolated point)
    We now interpolate on this (filling False for the extrapolation):
    '''
    lc1jdsmask = np.interp(lc1jds, lc2jdsa, lc2interpcode_sum, left=0, right=0) >= 1.0

    # --- Magnitude interpolation for all points, disregarding the mask (simpler -> faster) ---
    # These left and right values are a safety check : they should be masked, but if not, you will see them ...
    ipmagdiffs = np.interp(lc1jds, lc2jds, lc2mags, left=1.0e5, right=1.0e5) - lc1mags

    # --- Cutting out the mask
    okdiffs = ipmagdiffs[lc1jdsmask]
    n = len(okdiffs)

    if weights:
        # Then we do something similar for the errors :
        ipmagerrs = np.interp(lc1jds, lc2jds, lc2magerrs, left=1.0e-5, right=1.0e-5)
        ipsqrerrs = ipmagerrs * ipmagerrs + lc1magerrs * lc1magerrs  # interpolated square errors
        oksqrerrs = ipsqrerrs[lc1jdsmask]

        d2 = np.sum((okdiffs * okdiffs) / oksqrerrs) / n
    else:
        d2 = np.sum((okdiffs * okdiffs)) / n

    if plot:
        # Then we make a plot, essentially for testing purposes. Nothing important here !
        # This is not optimal, and actually quite slow.
        # Masked points are not shown at al

        plt.figure(figsize=(12, 8))  # sets figure size
        axes = plt.gca()

        # The points
        plt.plot(lc1jds, lc1mags, ".", color=lc1.plotcolour, label=lc1.object)
        plt.plot(lc2jds, lc2mags, ".", color=lc2.plotcolour, label=lc2.object)

        # Lines between lc2 points
        plt.plot(lc2jds, lc2mags, "-", color=lc2.plotcolour,
                 label=lc2.object)  # lines between non-masked lc2 points (= interpolation !)

        # Circles around non-used lc1 points
        plt.plot(lc1jds[lc1jdsmask == False], lc1mags[lc1jdsmask == False], linestyle="None", marker="o", markersize=8.,
                 markeredgecolor="black", markerfacecolor="None", color="black")  # circles around masked point

        # Dispersion "sticks"
        for (jd, mag, magdiff) in zip(lc1jds[lc1jdsmask], lc1mags[lc1jdsmask], okdiffs):
            plt.plot([jd, jd], [mag, mag + magdiff], linestyle=":", color="grey")

        # Something for astronomers only : we invert the y axis direction !
        axes.set_ylim(axes.get_ylim()[::-1])

        # And we make a title for that combination of lightcurves :
        # plt.title("Lightcurves", fontsize=20)
        plt.xlabel("Days", fontsize=16)
        plt.ylabel("Magnitude", fontsize=16)
        plt.title("%i interpolations" % n, fontsize=16)
        plt.show()

    return {'n': n, 'd2': d2}


def pelt95(lc1, lc2, decorlength=3.0, usemask=True, plot=False):
    """
    Dispersion method as described in Pelt Kayser Refsdal 93 & 95
    For now this corresponds to D3, the third form, i.e. using only strictly neighboring pairs,
    and no weighting.
    The fourth form uses *all* pairs within a given decorlength, non only neighboring.
    """
    if usemask:
        lc1jds = lc1.getjds()[lc1.mask]
        lc2jds = lc2.getjds()[lc2.mask]
        lc1mags = lc1.getmags()[lc1.mask]
        lc2mags = lc2.getmags()[lc2.mask]
        lc1magerrs = lc1.magerrs[lc1.mask]
        lc2magerrs = lc2.magerrs[lc2.mask]
    else:
        lc1jds = lc1.getjds()
        lc2jds = lc2.getjds()
        lc1mags = lc1.getmags()
        lc2mags = lc2.getmags()
        lc1magerrs = lc1.magerrs
        lc2magerrs = lc2.magerrs

    # Again, we want to avoid any loops !
    # So first we merge the lightcurves into a numpy array, introducing a last column that tracks the "origin" of each point.

    pts1 = np.column_stack((lc1jds, lc1mags, lc1magerrs, np.zeros(len(lc1jds))))
    pts2 = np.column_stack((lc2jds, lc2mags, lc2magerrs, np.ones(len(lc2jds))))
    pts = np.concatenate((pts1, pts2))

    # We sort this stuff according to the jds :
    sortedindices = np.argsort(pts[:, 0])
    pts = pts[sortedindices]

    # So now we have a big array of mixed points, with columns :
    # jds	mags	magerrs		1/0(dpending on origin)

    # Now we calculate all diffs between lines k+1 and k
    diffs = pts[1:] - pts[:-1]  # diffs = np.ediff1d(pts) # does not work... so we do it the hard way.
    # The trick is that we can infer good pairs from jd diffs and 1/0 diffs...

    # We do not care about the diffs of magnitude errors... instead we want to sum their squares
    sqrerrs = np.square(pts[1:, 2]) + np.square(pts[:-1, 2])
    # We can introduce this into the diffs array, in place of the error differences :
    diffs[:, 2] = sqrerrs

    # The faster way... selection of pairs :
    pairs = diffs[np.logical_and(diffs[:, 0] < decorlength, np.abs(diffs[:, 3]) > 0.1)]
    # And we calculate the sum terms :
    elements = np.square(pairs[:, 1]) / pairs[:, 2]

    n = len(elements)
    # d2 = np.sum(elements) / (n-1.0)
    d2 = np.sum(elements) / (2.0 * np.sum(1.0 / pairs[:, 2]))  # again, this is Pelt.
    # (dispersion estimator : sum a subset of half square-diffs)

    if plot:
        plt.figure(figsize=(12, 8))  # sets figure size
        axes = plt.gca()

        # The points
        plt.plot(lc1jds, lc1mags, ".", color=lc1.plotcolour, label=lc1.object)
        plt.plot(lc2jds, lc2mags, ".", color=lc2.plotcolour, label=lc2.object)

        for i, line in enumerate(diffs):
            if line[0] < decorlength and abs(line[3]) > 0.1:
                plt.plot([pts[i, 0], pts[i, 0] + line[0]], [pts[i, 1], pts[i, 1] + line[1]], linestyle="-",
                         color="gray")
                plt.plot([pts[i, 0] + line[0] / 2.0, pts[i, 0] + line[0] / 2.0], [pts[i, 1], pts[i, 1] + line[1]],
                         linestyle="-", color="black")

        axes.set_ylim(axes.get_ylim()[::-1])
        plt.xlabel("Days", fontsize=16)
        plt.ylabel("Magnitude", fontsize=16)
        plt.title("%i pairs" % n, fontsize=16)
        plt.show()

    return {"d2": d2, "n": n}


def symmetrize(lc1, lc2, dispersionmethod):
    """
    Calls your dispersion method on (A,B) and on (B,A) and returns the "average".
    """
    res1 = dispersionmethod(lc1, lc2)
    res2 = dispersionmethod(lc2, lc1)
    n = res1['n'] + res2['n']
    d2 = float(res1['n'] * res1['d2'] + res2['n'] * res2['d2']) / float(n)
    return {'n': n, 'd2': d2}
