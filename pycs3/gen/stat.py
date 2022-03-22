"""
Statistics related functions.
"""

import glob
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pycs3.gen.util as ut


def sf(l, binsize=200, ssf=False):
    """
    Structure function of a lightcurve

    For definition see for instance :
    De Vries, W.H. de, Becker, R., White, R., Loomis, C., 2005. Structure Function Analysis of Long-Term Quasar Variability. The Astronomical Journal 129, 615-615-629-629.
    :param l: LightCurve object
    :param binsize: float, binsize if ssf is False
    :param ssf: boolean,  ssf gives a 2D density plot, otherwise binned.
    """

    mags = l.getmags()
    jds = l.getjds()
    n = len(l)

    ja = np.ones((n, n)) * jds
    jam = ja - ja.transpose()
    jam = jam.flatten()
    keep = jam > 0.0
    jam = jam[keep]

    ma = np.ones((n, n)) * mags
    mam = ma - ma.transpose()
    mam = mam.flatten()
    mam = mam[keep]

    if ssf:  # stochastic structure function, we plot a 2d distribution

        jam = jam.flatten()
        mam = mam.flatten()

        plt.scatter(jam, mam, s=1.0)

        plt.xlabel("Delta t")
        plt.ylabel("Delta m")

        plt.show()

    else:  # we do a normal structure function, "variance of increments versus delta t" :
        mam = np.square(mam)

        order = np.argsort(jam)  # sorting according to the jd gaps
        jam = jam[order]
        mam = mam[order]

        m = len(jam)
        nbins = int(math.floor(m / binsize))
        jam = jam[0:nbins * binsize]  # cutting to the nearest last bin
        mam = mam[0:nbins * binsize]

        jam = jam.reshape((nbins, binsize))
        mam = mam.reshape((nbins, binsize))

        cjs = np.mean(jam, axis=1)
        cms = np.sqrt(np.mean(mam, axis=1) / float(binsize))

        plt.scatter(cjs, cms)

        plt.xlabel("Delta t")
        plt.ylabel("SF")

        plt.show()


def mad(data, axis=None):
    """
    Median absolute deviation
    :param data: array from which to compute the MAD
    :param axis: axis along to compute the MAD

    :return: float, MAD of the array

    """

    return np.median(np.absolute(data - np.median(data, axis)), axis)


def erf(x):
    """
    Error function. There is one in scipy, but this way we do it without scipy...

    scipy.special.erf(z)

    """
    # save the sign of x
    sign = 1
    if x < 0:
        sign = -1
    x = abs(x)

    # constants
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    # A&S formula 7.1.26
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)
    return sign * y  # erf(-x) = -erf(x)

def compute_dof_spline(rls, kn, knml):
    """
    Compute the number of degree of freedom of the spline optimiser, assuming all curves have the same microlensing.

    :param rls: list of residuals LightCurves. Use the subtract() fucntion to generate them.
    :param kn: float, knotstep of the spline
    :param knml: float, knotstep of the microlensing splines
    :return: float, number of degree of freedom
    """
    n_curve = len(rls)
    a = rls[0].jds[0]
    b = rls[0].jds[-1]
    nkn = int(float(b - a) / float(kn) - 2)  # number of internal knot
    if knml != 0:
        nknml = int(float(b - a) / float(knml) - 2)  # number of internal ml knot
    else:
        nknml = 0
    return (2 * nkn + n_curve) + n_curve * (2 * nknml + n_curve) + n_curve


def compute_chi2(rls, kn, knml):
    """
    Compute the chi2 of the spline fit, given a list residuals LightCurve
    :param rls: list of residuals LightCurves. Use the subtract() fucntion to generate them.
    :param kn: float, knotstep of the spline
    :param knml: float, knotstep of the microlensing splines

    :return: float, chi2 per degree of freedom of the fit.

    """
    chi2 = 0.0
    for rl in rls:
        chi2_c = np.mean((rl.getmags() ** 2) / rl.getmagerrs() ** 2)
        chi2 += chi2_c

    chi2_red = chi2 / compute_dof_spline(rls, kn, knml)
    return chi2_red


def runstest(residuals, autolevel=False, verbose=True):
    """
    One-sample runs test of randomness as presented in Practical Statistics for Astronomers by
    J. V. Wall and C. R. Jenkins, paragraph 5.3.3
    WARNING : ERROR IN THE BOOKS EXPECTATION FORMULA ! confirmed by author

    residuals is a numpy array of floats

    returns z (in units of sigmas)
    and p = assuming that the data are independent, the probability to get a result worse then what you got.

    """

    medianlevel = np.median(residuals)
    if autolevel:
        if verbose:
            print(("Leveling to median : %f" % medianlevel))
        residuals -= medianlevel

    bools = residuals > 0  # So residuals = 0 would be set to False, but
    bools = bools[np.abs(residuals) > 0.000001]  # we remove those very close to 0

    if verbose:
        print("Total : %i points / significant : %i points" % (len(residuals), len(bools)))

    n = len(bools)
    if n <= 20:
        print("WARNING : to few points for a meaningful runs test (or too close to 0.0)")

    intbools = bools.astype(int)
    nplus = np.sum(bools)
    nminus = n - nplus

    # And the number of runs
    changes = np.abs(intbools[1:] - intbools[:-1])
    nruns = np.sum(changes) + 1

    if verbose:
        print("     + (m) : %i" % nplus)
        print("     - (n) : %i" % nminus)
        print("  Runs (r) : %i" % nruns)

    # For large N, we can approximate the distribution of r by a gaussian :
    # Error from the book :
    # mur = (2.0 * nplus * nminus)/(nplus + nminus + 1.0)
    # sigmar = math.sqrt((2.0*nplus*nminus*(2.0*nplus*nminus - n))/(n*n*(n - 1.0)))
    # From corrected book and also wikipedia :
    mur = ((2.0 * nplus * nminus) / n) + 1.0
    sigmar = math.sqrt((mur - 1.0) * (mur - 2.0) / (n - 1.0))

    zruns = (nruns - mur) / sigmar

    # The probability to obtain a result worse than what you got :
    pruns = 1.0 - erf(abs(zruns) / math.sqrt(2.0))

    return {"zruns": zruns, "pruns": pruns, "nruns": nruns}


def subtract(lcs, model):
    """

    I return a list of residual light curves ("lcs - spline").
    Technically, "residual" light curves are nothing but normal lightcurves objects.
    Of course, I take into account any shifts or microlensing of your lcs.
    I do not modify my input arguments.

    :param lcs: grid to interpolate
    :param model: a Spline or a Rslc object, that have a eval() method
    :return: array, containing the interpolated values

    """

    rls = []

    for l in lcs:

        if l.hasmask():
            print("WARNING : I do not take into account the mask !")

        lp = l.copy()  # To avoid border effects

        lp.applyfluxshift()
        lp.applymagshift()
        if lp.ml is not None:
            lp.applyml()

        lp.mags -= model.eval(lp.getjds())

        rls.append(lp)

    return rls


def resistats(rl):
    """
    Give me a residual lightcurve, I return a dict with some descriptive stats about its magnitudes.
    """

    meanmag = np.mean(rl.getmags())
    stdmag = mad(rl.getmags())  # use median absolute deviation instead, for robustness to outliers
    runs = runstest(rl.getmags(), autolevel=False, verbose=False)

    out = {"mean": meanmag, "std": stdmag}
    out.update(runs)

    return out


def mapresistats(rls):
    """
    Return resistats of each residual curve.

    :param rls: list of residual LightCurve

    """
    return [resistats(rl) for rl in rls]


def anaoptdrawn(optoriglcs, optorigspline, simset="simset", optset="optset", npkl=1000, plots=True, nplots=3, r=0.11,
                plotjdrange=None, plotcurveindexes=None, showplot=False, directory="./", plotpath="./", id = "", resihist_figsize=None):
    """
    Not flexible but very high level function to analyse the spline-fit-residuals of drawn curves and comparing them to the
    real observations.
    This can be used to tune the parameters of the "drawing".
    .. warning:: The simset must have been optimized using spline fits, with option keepopt=True

    :param optoriglcs: optimized original curves
    :type optoriglcs: list
    :param optorigspline: spline that matches to these curves
    :type optorigspline: Spline
    :param simset: name of your simulation set
    :type simset: str
    :param optset: name of your optimisation
    :type optset: str
    :param plotcurveindexes: allows you to plot only a subset of lcs (smaller plots). Give a tuple like eg (0, 2, 3)
    :type plotcurveindexes: tuple
    :param npkl: I read only the first npkl pickle files.
    :type npkl: int
    :param plots: To choose if you want to produce the plots
    :type plots: bool
    :param nplots: number of mock curves to plot for the residual plots
    :type nplots: int
    :param r: radius around the mean to plot for the residuals histograms
    :type r: float
    :param plotjdrange: containing the two extremity of the period to plot
    :type plotjdrange: list
    :param showplot: False to save the figure in png, True to show it on the screen
    :type showplot: bool
    :param plotpath: directory to save the pngs, used if showplot is False
    :type plotpath: str
    :param directory: path to look for the simulation
    :type directory: str
    :param resihist_figsize: containing the dimension of the residuals histograms
    :type resihist_figsize: tuple
    :param id: to give an additionnal id name in the figure name.
    :type id: str
    :return: dictionnary containing the statistics about your run

    """
    print("Analysing the residuals of simset %s" % simset)

    # For each light curve we make a dict that we will use to store stuff
    curves = [{"optoriglc": optoriglc} for optoriglc in optoriglcs]

    # We compute the residuals of the original curve
    optorigrlcs = subtract(optoriglcs, optorigspline)
    for (curve, optorigrlc) in zip(curves, optorigrlcs):
        curve["optorigrlc"] = optorigrlc

    # We read all the optimized mock curves :
    pkls = sorted(glob.glob(os.path.join(directory, "sims_%s_opt_%s/*_opt.pkl" % (simset, optset))))
    print(os.path.join(directory, "sims_%s_opt_%s/*_opt.pkl" % (simset, optset)))

    optmocksplinelist = []
    optmocklcslist = []

    for (i, pkl) in enumerate(pkls):
        if i >= npkl:
            break
        opttweak = ut.readpickle(pkl, verbose=False)
        optmocksplinelist.extend(opttweak["optfctoutlist"])
        optmocklcslist.extend(opttweak["optlcslist"])

    assert len(optmocksplinelist) == len(optmocklcslist)

    print("We have %i simulated curves" % (len(optmocksplinelist)))

    # We compute all the residuals of the mock curves, and store them
    for curve in curves:
        curve["optmockrlclist"] = []

    for (optmocklcs, optmockspline) in zip(optmocklcslist, optmocksplinelist):
        assert len(optmocklcs) == len(optoriglcs)
        optmockrlcs = subtract(optmocklcs, optmockspline)

        for (curve, optmockrlc) in zip(curves, optmockrlcs):
            assert curve["optorigrlc"].object == optmockrlc.object
            curve["optmockrlclist"].append(optmockrlc)

    # We want to return the displayed statistics
    stats = []
    for curve in curves:
        curve["origresistats"] = resistats(curve["optorigrlc"])

        curve["mockresistats"] = list(map(resistats, curve["optmockrlclist"]))
        curve["meanmockresistats"] = dict(
            [[key, np.mean(np.array([el[key] for el in curve["mockresistats"]]))] for key in
             list(curve["origresistats"].keys())])
        curve["medmockresistats"] = dict(
            [[key, np.median(np.array([el[key] for el in curve["mockresistats"]]))] for key in
             list(curve["origresistats"].keys())])
        curve["stdmockresistats"] = dict([[key, np.std(np.array([el[key] for el in curve["mockresistats"]]))] for key in
                                          list(curve["origresistats"].keys())])

        print("++++++ %s ++++++" % curve["optorigrlc"].object)
        curve["zrunstxt"] = "zruns : %.2f (obs) vs %.2f +/- %.2f (sim)" % (
            curve["origresistats"]["zruns"], curve["meanmockresistats"]["zruns"], curve["stdmockresistats"]["zruns"])
        curve["sigmatxt"] = "sigma : %.4f (obs) vs %.4f +/- %.4f (sim)" % (
            curve["origresistats"]["std"], curve["meanmockresistats"]["std"], curve["stdmockresistats"]["std"])
        print(curve["zrunstxt"])
        print(curve["sigmatxt"])

        # return the original, mean and std of mocks zruns, then original, mean and std of mocks of sigma
        stats.append(
            [curve["origresistats"]["zruns"], curve["meanmockresistats"]["zruns"], curve["stdmockresistats"]["zruns"],
             curve["origresistats"]["std"], curve["meanmockresistats"]["std"], curve["stdmockresistats"]["std"]])

    # Now we proceed with making plots.
    # Resi and zruns histos combined into one nicer figure :

    if plots:
        if resihist_figsize is None:
            plt.figure(figsize=(3 * len(curves), 4))
        else:
            plt.figure(figsize=resihist_figsize)
        plt.subplots_adjust(left=0.02, bottom=0.12, right=0.98, top=0.98, wspace=0.08, hspace=0.37)

        # Resi histos :
        for (i, curve) in enumerate(curves):
            plt.subplot(2, len(curves), i + 1)
            plt.hist(np.concatenate([rlc.mags for rlc in curve["optmockrlclist"]]), 50, range=(-r, r),
                     facecolor='black', alpha=0.4, density=1, histtype="stepfilled")
            # Gaussian for the mock hist :
            plt.hist(curve["optorigrlc"].mags, 50, facecolor=curve["optorigrlc"].plotcolour, alpha=0.5, range=(-r, r),
                     density=1, histtype="stepfilled")
            plt.xlabel("Spline fit residuals [mag]")

            plt.text(-r + 0.1 * r, 0.8 * plt.gca().get_ylim()[1], curve["optorigrlc"].object, fontsize=18)
            plt.xlim(-r, r)
            plt.gca().get_yaxis().set_ticks([])

        # zruns histos :
        for (i, curve) in enumerate(curves):
            plt.subplot(2, len(curves), len(curves) + i + 1)

            plt.hist(np.array([el["zruns"] for el in curve["mockresistats"]]), 20, facecolor="black", alpha=0.4,
                     density=1, histtype="stepfilled")
            plt.axvline(curve["origresistats"]["zruns"], color=curve["optorigrlc"].plotcolour, linewidth=2.0, alpha=1.0)

            plt.xlabel(r"$z_{\mathrm{r}}$", fontsize=18)
            plt.gca().get_yaxis().set_ticks([])

        if showplot:
            plt.show()
        else :
            plt.savefig(os.path.join(plotpath, "%s_fig_anaoptdrawn_%s_%s_resihists.png" % (id, simset, optset)))

    # A detailed plot of some residuals, just for a few drawn curves

    if plots:
        for i in range(nplots):

            optmockrlcs = [curve["optmockrlclist"][i] for curve in curves]
            for l in optmockrlcs:
                l.plotcolour = "black"

            optorigrlcs = [curve["optorigrlc"] for curve in curves]

            if plotcurveindexes is not None:
                optorigrlcs = [optorigrlcs[index] for index in plotcurveindexes]
                optmockrlcs = [optmockrlcs[index] for index in plotcurveindexes]

            if not showplot :
                filn = os.path.join(plotpath, "%s_fig_anaoptdrawn_%s_%s_resi_%i.png" % (id, simset, optset, i + 1))
            else :
                filn = None

            plotresiduals([optorigrlcs, optmockrlcs], jdrange=plotjdrange, nicelabel=False, showlegend=False,
                          showsigmalines=False, errorbarcolour="#999999",
                          filename=filn)

    return stats


def plotresiduals(rlslist, jdrange=None, magrad=0.1, errorbarcolour="#BBBBBB",
                  showerrorbars=True, showlegend=True, nicelabel=True,
                  showsigmalines=True, filename=None, ax=None):
    """
    We plot the residual lightcurves in separate frames.

    The arguement rlslist is a *list* of *lists* of lightcurve objects.
    Ths sublists should have the same length, I'll choose my number of panels accordingly.
    The structure is : [[lca, lcb], [lca_sim1, lcb_sim1], ...]
    If you have only one lightcurve object, you can of course pass [[l]] ...

    I disregard the timeshift of the curves !
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker

    minorlocator = matplotlib.ticker.MultipleLocator(50)
    majorlocator = matplotlib.ticker.MultipleLocator(200)

    resminorlocator = matplotlib.ticker.MultipleLocator(0.01)
    resmajorlocator = matplotlib.ticker.MultipleLocator(0.05)

    eps = 0.001

    npanels = len(rlslist[0])
    if ax is None:
        fig = plt.figure(figsize=(12, 1.6 * npanels))  # sets figure size
        fig.subplots_adjust(left=0.07, right=0.99, top=0.95, bottom=0.17, hspace=0.05)
        ax = plt.gca()
        ihaveax = False
    else:
        ihaveax = True

    for i in range(npanels):  # i is the panel index

        rls = [rlslist[j][i] for j in range(len(rlslist))]  # j is the curve index.

        if ihaveax:
            ax0 = ax
        else:
            if i > 0:
                ax = plt.subplot(npanels, 1, i + 1, sharex=ax0, sharey=ax0)
            else:
                ax = plt.subplot(npanels, 1, i + 1)
                ax0 = ax

        for (j, rl) in enumerate(rls):

            stats = resistats(rl)

            label = "[%s/%s] (std: %.4f, zruns : %.3f)" % (rl.telescopename, rl.object, stats["std"], stats["zruns"])
            if nicelabel:
                label = "%s" % rl.object

            if showerrorbars:
                ax.errorbar(rl.jds, rl.getmags(), rl.magerrs, fmt=".", color=rl.plotcolour,
                            markeredgecolor=rl.plotcolour, ecolor=errorbarcolour, label=label, elinewidth=0.5)
            else:
                ax.plot(rl.jds, rl.getmags(), marker=".", markersize=3.0, linestyle="None",
                        markeredgecolor=rl.plotcolour, color=rl.plotcolour, label=label)

            if showsigmalines:
                ax.axhline(y=stats["std"], lw=0.5, color=rl.plotcolour)
                ax.axhline(y=-stats["std"], lw=0.5, color=rl.plotcolour)

            if nicelabel:
                ax.text(0.04 + (0.087 * j), 0.82, label, transform=ax.transAxes, color=rl.plotcolour)
            else:
                if not showlegend:
                    if j == 0:
                        ax.text(0.01, 0.81, rl.object, transform=ax.transAxes, color=rl.plotcolour, fontsize=17)

        ax.axhline(0, color="gray", dashes=(3, 3))

        ax.xaxis.set_minor_locator(minorlocator)
        ax.xaxis.set_major_locator(majorlocator)

        ax.yaxis.set_minor_locator(resminorlocator)
        ax.yaxis.set_major_locator(resmajorlocator)

        ax.set_ylim(-magrad + eps, magrad - eps)
        ax.set_ylim(ax.get_ylim()[::-1])

        if showlegend:
            ax.legend(numpoints=1, prop={'size': 10})

        # ax.set_ylabel("Residual [mag]")

        ax.set_xlabel("HJD - 2400000.5 [day]", fontsize=18)

        if i != npanels - 1:
            plt.setp(ax.get_xticklabels(), visible=False)
            ax.set_xlabel("")

    ax.text(0.015, 0.54, ' Spline Residuals [mag]', rotation=90, verticalalignment="center",
            horizontalalignment="center", transform=plt.gcf().transFigure, fontsize=16)

    if jdrange is not None:
        plt.xlim(jdrange[0], jdrange[1])
    else:
        plt.xlim(np.min(rlslist[0][0].jds) - 50, np.max(rlslist[0][0].jds) + 50)

    if filename and ihaveax:
        plt.savefig(filename)
    elif filename and not ihaveax:
        plt.savefig(filename)
    elif not filename and ihaveax:
        return
    else:
        plt.show()
