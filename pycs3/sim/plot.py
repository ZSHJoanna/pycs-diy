"""
Subpackage with functions to plot all kind of results from runs.
The delayplot function is now moved to :py:mod:`pycs3.tdcomb.plot`
"""

import logging
import math
import os

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
from matplotlib.ticker import MultipleLocator, MaxNLocator

import pycs3.gen.util
from pycs3.gen.stat import mad

logger = logging.getLogger(__name__)


class DelayContainer:
    """
    Stores the delay or error bar measurement(s) (one for each curve pair).
    This object is usually produced by the plot-functions ``hists`` or ``meanvstrue`` below.

    markers : [ 7 | 4 | 5 | 6 | 'o' | 'D' | 'h' | 'H' | '_' | '' | 'None' | ' ' | None | '8' | 'p' | ',' | '+' | '.' | 's' | '*' | 'd' | 3 | 0 | 1 | 2 | '1' | '3' | '4' | '2' | 'v' | '<' | '>' | '^' | '|' | 'x' | '$...$' | tuple | Nx2 array ]

    """

    def __init__(self, data, name="No name", objects=None, plotcolour="black", marker=None):
        """
        self.data is a list of dicts, the fields depend on if it's delays or errorbars

            * delays : label, mean, med, std
            * errorbars : label, tot, sys, ran, bias


        """
        self.data = data
        self.name = name
        self.objects = objects

        self.plotcolour = plotcolour
        self.marker = marker
        self.markersize = 5.0
        self.yshift = 0.0  # allows to "group" measurements


def normal(x, mu, sigma):
    """
    Plain normal distribution.
    You can directly apply me on numpy arrays x, mu, sigma.
    """

    return (1.0 / np.sqrt(2.0 * np.pi * sigma * sigma)) * np.exp(- (x - mu) ** 2 / (2 * sigma * sigma))


def hists(rrlist, r=10.0, nbins=100, showqs=True, showallqs=False, qsrange=None, title=None, xtitle=0.5, ytitle=0.95,
          titlesize=18, niceplot=False, displaytext=True, figsize=(16, 9), left=0.06, right=0.95, bottom=0.065,
          top=0.95, wspace=0.2, hspace=0.2, txtstep=0.04, majorticksstep=2, hideyaxis=True, trueshifts=None,
          filename=None, dataout=False, blindness=False, usemedian=False, outdir="./"):
    """
    Comparing the delay distributions from different run result objects.

    :param rrlist: a list of RunResults object.
    :param r: a range radius for the hists

    :param showqs: If True, I overplot the qs as scatter points.

    :param dataout: True means that I'll write the pkl file, containing a DelayContainer object needed to make the delayplot.

    :param usemedian: if True, use the median and median absolute deviation instead of mean and std. Default is False.

    .. warning:: To avoid rewriting newdelayplot, if usemedian is True then I write the median and mad in the mean and std fields of the pickles. This is dangerous (and a bit stupid and lazy), but since hists() and newdelayplot() are usually called one after the other it should not create too much confusion.

    """

    n = rrlist[0].nimages()

    labels = rrlist[0].labels

    # To get some fixed ranges for the histograms, we will use the center of the histos :
    reftrueshifts = 0.5 * (np.max(np.array([rr.getts()["center"] for rr in rrlist]), axis=0) + np.min(
        np.array([rr.getts()["center"] for rr in rrlist]), axis=0))

    for rr in rrlist:
        if rr.labels != labels:
            raise RuntimeError("Don't ask me to overplot runresults of different curves !")

        rr.trues = rr.gettruets()  # To store this, avoids calculating it again and again.

        rr.tmpdata = []

    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top, wspace=wspace, hspace=hspace)

    axisNum = 0
    for i in range(n):  # [A, B, C, D]
        for j in range(n):

            if (i == 0) or (j == n - 1):
                continue  # No plot

            axisNum += 1
            if j >= i:
                continue

            ax = plt.subplot(n - 1, n - 1, axisNum)

            # Delay label, used not only for display purposes, but also for the output pkl.
            delaylabel = "%s%s" % (labels[j], labels[i])

            if i == n - 1:
                if n == 2:  # Only one panel -> we write the object names into the xlabel
                    plt.xlabel("Delay %s%s [day]" % (labels[j], labels[i]), fontsize=14)
                else:
                    plt.xlabel("Delay [day]", fontsize=14)
            if showqs:
                axscatter = ax.twinx()

            # Ranges to plot
            reftruedelay = reftrueshifts[i] - reftrueshifts[j]
            histrange = (reftruedelay - r, reftruedelay + r)

            for irr, rr in enumerate(rrlist):
                # We will express the delays "i - j"
                if rr.plottrue:
                    delays = rr.truetsarray[:, i] - rr.truetsarray[:, j]
                else:
                    delays = rr.tsarray[:, i] - rr.tsarray[:, j]

                meddelay = np.median(delays)
                maddelay = mad(delays)
                meandelay = np.mean(delays)
                stddelay = np.std(delays)

                # We save these :
                if usemedian:
                    rr.tmpdata.append({"label": delaylabel, "mean": meddelay, "med": meddelay, "std": maddelay})
                else:
                    rr.tmpdata.append({"label": delaylabel, "mean": meandelay, "med": meddelay, "std": stddelay})

                (counts, bins, patches) = ax.hist(delays, bins=nbins, range=histrange, histtype="bar",
                                                  color=rr.plotcolour, alpha=0.4, lw=0, density=1.)

                if niceplot:
                    majorLocator = MultipleLocator(majorticksstep)
                    minorLocator = MultipleLocator(1.0)
                    ax.xaxis.set_major_locator(majorLocator)
                    ax.xaxis.set_minor_locator(minorLocator)
                    ax.yaxis.set_ticks([])

                if showqs and not rr.plottrue:
                    if showallqs:
                        axscatter.scatter(delays, rr.qs, s=1, facecolor=rr.plotcolour, lw=0)
                    else:
                        axscatter.scatter(delays[::5], rr.qs[::5], s=1, facecolor=rr.plotcolour, lw=0)

                    if qsrange:
                        axscatter.set_ylim(qsrange)
                    if niceplot:
                        majorLocator = MultipleLocator(500)
                        axscatter.yaxis.set_major_locator(majorLocator)

                # We plot the true shifts (if available) as a straight line individually for each rr :
                if rr.trues["type"] == "same":
                    truedelay = rr.trues["center"][i] - rr.trues["center"][j]
                    plt.axvline(x=truedelay, linewidth=1, linestyle="--", color=rr.plotcolour)

                if displaytext:

                    if getattr(rr, "plotgauss", False):
                        x = np.linspace(histrange[0], histrange[1], 100)
                        y = normal(x, meandelay, stddelay)
                        ax.plot(x, y, linestyle="-", color=rr.plotcolour)

                    if not usemedian:
                        delaytext = r"%+.1f $\pm$ %.1f" % (meandelay, stddelay)
                    else:
                        delaytext = r"%+.1f $\pm$ %.1f" % (meddelay, maddelay)

                    ax.annotate(delaytext, xy=(0.04, 0.7 - 0.12 * irr), xycoords='axes fraction', color=rr.plotcolour,
                                fontsize=10)

            plt.xlim(histrange)

            # We increase ylim by 30% if it
            ylims = list(ax.get_ylim())
            if n == 2:  # single panel
                ylims[1] *= 1.4
            else:
                ylims[1] *= 1.1
            ax.set_ylim(ylims)

            # hide y axis if wanted to
            if hideyaxis:
                ax.set_yticks([])

            # make the ticks a little bit bigger than default
            plt.xticks(fontsize=13)

            # enforce blindness if wanted, by modifying the xticks labels (not touching the data)
            if blindness:
                labels = ax.xaxis.get_ticklabels()
                locs = ax.xaxis.get_ticklocs()
                meanloc = np.mean(locs)
                blindlabels = []
                for loc, label in zip(locs, labels):
                    blindlabels.append(str(loc - meanloc))
                ax.xaxis.set_ticklabels(blindlabels)

            # Looked ok on big plots :
            # plt.annotate(delaylabel, xy=(0.03, 0.88),  xycoords='axes fraction', fontsize=12, color="black")
            if n != 2:  # otherwise we have only one single panel
                plt.annotate(delaylabel, xy=(0.05, 0.84), xycoords='axes fraction', fontsize=14, color="black")

            if trueshifts is not None:
                truedelay = trueshifts[i] - trueshifts[j]
                plt.axvline(truedelay, color="gray", linestyle="--", dashes=(3, 3), zorder=-20)

    if dataout:
        for rr in rrlist:
            dc = DelayContainer(data=rr.tmpdata, name=rr.name, plotcolour=rr.plotcolour, objects=labels[:])
            pycs3.gen.util.writepickle(dc, os.path.join(outdir, "%s_delays.pkl" % rr.autoname))
            rr.tmpdata = None

    labelspacetop = 0.0
    labelspaceright = 0.0
    if n == 2:
        labelspacetop = 0.04
        labelspaceright = 0.04

    for irr, rr in enumerate(rrlist):

        if niceplot:
            labeltxt = "%s" % (getattr(rr, 'name', 'NoName'))
            plt.figtext(x=right - labelspaceright, y=top - labelspacetop - txtstep * irr, s=labeltxt,
                        verticalalignment="top", horizontalalignment="right", color=rr.plotcolour, fontsize=15)
        else:
            labeltxt = "%s (%s, %i) " % (
                getattr(rr, 'name', 'NoName'), "Truth" if rr.plottrue else "Measured", rr.tsarray.shape[0])
            plt.figtext(x=right - labelspaceright, y=top - labelspacetop - txtstep * irr, s=labeltxt,
                        verticalalignment="top", horizontalalignment="right", color=rr.plotcolour, fontsize=15)

        rr.print_shifts()

    if title is not None:
        plt.figtext(x=xtitle, y=ytitle, s=title, horizontalalignment="center", color="black", fontsize=titlesize)

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)


def newcovplot(rrlist, r=6, rerr=3, nbins=10, nbins2d=3, binclip=True, binclipr=10.0, figsize=(13, 13), left=0.06,
               right=0.97, top=0.97, bottom=0.04, wspace=0.3, hspace=0.3, method='indepbin', minsamples=10,
               showplots=True, printdetails=True, printcovmat=True, detailplots=False, filepath=None, verbose=True):
    """
    Covariance scatter of all measurement errors.
    Give me a single runresults object (from a sim, with known true delays).
    """

    assert (method in ['depbin', 'indepbin'])
    retdict = {}  # we put all the intermediate products in a dict that we return

    nimages = rrlist[0].nimages()
    imginds = np.arange(nimages)
    labels = rrlist[0].labels

    if nimages == 4:  # then it's a quad
        covmatsize = 6
    elif nimages == 3:  # then it's a folded quad
        covmatsize = 3
    else:  # then it's a double
        logger.info("This function does not work for doubles")
        logger.info(
            "I kindly remind you that the covariance between a variable and itself is called variance, and there are simpler functions to compute that in PyCS. Try newdelayplot for instance.")

    couplelist = [(i, j) for j in imginds for i in imginds if i > j]
    ncouples = len(couplelist)

    tderrsdicts = []
    # rrlist is just a list of rr, we treat them one after the other
    for rr in rrlist:
        # for each rr, we compute the error from the true delay
        truetsslist = rr.truetsarray
        tsslist = rr.tsarray - truetsslist
        for ind, tss in enumerate(tsslist):
            tderrs = []
            truetds = []
            for (i, j) in couplelist:
                tderrs.append(tss[i] - tss[j])
                truetds.append(truetsslist[ind][i] - truetsslist[ind][j])
            tderrsdicts.append({"tderrs": tderrs, "truetds": truetds})

    # tderrsdict contains the errors on the true delays, as well as the true delays for each simulation

    # figure 1: general covariance plot for each pair of delays. Diagonal elements are the same than newdelayplot, off-diagonal elements are covariance for all the runresults
    allcovplot = plt.figure(figsize=figsize)
    allcovplot.subplots_adjust(left=left, right=right, bottom=bottom, top=top, wspace=wspace, hspace=hspace)
    # figure 2: covariance computed in each bin, for each pair of delays. Diagonal elements are the same than newdelayplot, off diagonal elements are colored tiles of covariance per true delays bins, with points overplotted.
    bincovplot = plt.figure(figsize=figsize)
    bincovplot.subplots_adjust(left=left, right=right, bottom=bottom, top=top, wspace=wspace, hspace=hspace)

    axisNum = 0

    # create the empty covariance matrix
    covmat = []
    for ind in range(len(couplelist)):
        covmat.append(np.zeros(len(couplelist)))
    indepbins = np.zeros(len(couplelist))
    depbins = np.zeros(len(couplelist))
    rranges = np.zeros(len(couplelist))
    retdict["delay"] = {}  # dict in a dict !
    for ii, i in enumerate(couplelist):  # (0, 1), (0, 2) ...
        delaylabel = "%s%s" % (labels[i[1]], labels[i[0]])
        retdict["delay"]["%s" % delaylabel] = {}  # dict in a dict in a dict ! dictception !!

        xtderrs = [tderrsdict["tderrs"][ii] for tderrsdict in tderrsdicts]
        xtruetds = [tderrsdict["truetds"][ii] for tderrsdict in tderrsdicts]
        maxx = np.max(xtruetds)
        minx = np.min(xtruetds)

        # fill the diagonal element
        ax1 = allcovplot.add_subplot(ncouples, ncouples, covmatsize * ii + (ii + 1))
        ax2 = bincovplot.add_subplot(ncouples, ncouples, covmatsize * ii + (ii + 1))
        majorLocator = MultipleLocator(1.0)

        for ax in [ax1, ax2]:
            ax.yaxis.set_major_locator(majorLocator)
            ax.xaxis.set_major_locator(MaxNLocator(5))
            if ii == len(couplelist) - 1:
                ax1.set_xlabel('True Delay [day]')
                ax2.set_ylabel('Measurement error [day]', labelpad=-10)

        # way 1 - binning independent of xtruedelays distribution. User choose the plot range. Similar to newdelayplot()

        reftrueshifts = np.mean([rr.gettruets()["center"] for rr in rrlist], axis=0)
        reftruedelay = reftrueshifts[i[0]] - reftrueshifts[i[1]]
        plotrange = (reftruedelay - r, reftruedelay + r)
        binlims = np.linspace(plotrange[0], plotrange[1], nbins + 1)

        # needed for binvals:
        xtderrs = np.array(xtderrs)

        digitized = np.digitize(xtruetds, binlims)
        binvals = [xtderrs[digitized == bini] for bini in range(1, len(binlims))]
        binstds = list(map(np.std, binvals))
        binmeans = list(map(np.mean, binvals))

        if binclip:
            for (bini, binvalarray) in enumerate(binvals):

                keep = np.logical_and(binvalarray < binclipr, binvalarray > -binclipr)
                if np.sum(keep == False) != 0:
                    n_kick = np.sum(keep == False)
                    logging.info("Kicking %i points." % (n_kick))
                    if n_kick > 0.2 * len(binvalarray):
                        logging.warning(
                            'Kicking more than 20 percent of your curves. Something went probably wrong ! Please consider using other estimator parameters.')
                binvals[bini] = binvalarray[keep]
            binstds = list(map(np.std, binvals))
            binmeans = list(map(np.mean, binvals))

        syserror = np.max(np.fabs(binmeans))
        randerror = np.max(binstds)
        toterror = np.sqrt(syserror * syserror + randerror * randerror)
        indepbins[ii] = toterror

        retdict["delay"]["%s" % delaylabel][
            "indep"] = {}  # dict in a dict in a dict in a dict ! we need to go deeper !!!
        retdict["delay"]["%s" % delaylabel]["indep"]["syserror"] = syserror
        retdict["delay"]["%s" % delaylabel]["indep"]["randerror"] = randerror
        retdict["delay"]["%s" % delaylabel]["indep"][
            "toterror"] = toterror  # that's already in the covariance matrix...

        # Plot the result !
        line = np.linspace(plotrange[0], plotrange[1], 100)
        zeros = np.zeros(100)
        width = binlims[1] - binlims[0]
        for ax in [ax1, ax2]:
            ax.plot(line, zeros, color="black", lw=0.5)
            ax.bar(binlims[:-1], binmeans, yerr=binstds, width=width, color=rr.plotcolour, ecolor=rr.plotcolour,
                   error_kw={"capsize": 2.5, "capthick": 0.5, "markeredgewidth": 0.5}, edgecolor=rr.plotcolour,
                   alpha=0.2)
            ax.set_ylim((-rerr, rerr))
            if figsize[0] > 8:
                ax.annotate(delaylabel, xy=(0.9, 0.05), xycoords='axes fraction', ha="center")  # x axis
            else:
                ax.annotate(delaylabel, xy=(0.78, 0.08), xycoords='axes fraction', ha="center")
            ax.set_xlim(plotrange)

            majorLocator = MultipleLocator(int(r / 2.0) + 1)
            ax.xaxis.set_major_locator(majorLocator)

            ax.set_title(r'sys=%.2f | ran=%.2f' % (syserror, randerror) + '\n' + 'tot=%.2f' % toterror, fontsize=10)

        # way 2 - binning dependent on the xtruedelays samples: min and max vals corresponds to the extremas of xtruedelays distribution
        xbinvals = np.linspace(minx, maxx, num=nbins + 1, endpoint=True)
        rranges[ii] = maxx - minx

        binmeans = []
        binstds = []
        for indx, xbinval in enumerate(xbinvals[:nbins]):
            subsamples = []
            for (ind, xtruetd) in enumerate(xtruetds):
                if xbinval < xtruetd < xbinvals[indx + 1]:
                    subsamples.append(xtderrs[ind])
            binmeans.append(np.mean(subsamples))
            binstds.append(np.std(subsamples))
        syserror = np.max(np.fabs(binmeans))
        randerror = np.max(binstds)
        toterror = np.sqrt(syserror * syserror + randerror * randerror)
        depbins[ii] = toterror

        retdict["delay"]["%s" % delaylabel]["dep"] = {}
        retdict["delay"]["%s" % delaylabel]["dep"]["syserror"] = syserror
        retdict["delay"]["%s" % delaylabel]["dep"]["randerror"] = randerror
        retdict["delay"]["%s" % delaylabel]["dep"]["toterror"] = toterror

        # We let the user choose which method he prefers
        # Dear user, be CAREFUL with your choice !

        if method == 'depbin':
            if ii == 0 and verbose:
                logger.info("You chose a binning depending on the sample values")
            covmat[ii][ii] = depbins[ii]
        elif method == 'indepbin':  # that should be the default value
            if ii == 0 and verbose: logger.info("You chose a binning independent of the sample values")
            covmat[ii][ii] = indepbins[ii]

        ### fill the off-diagonal elements
        retdict["cov"] = {}
        for jj, j in enumerate(couplelist):
            axisNum += 1
            if (ii == 0) or (jj == ncouples - 1):
                continue  # No plot
            if jj >= ii:
                continue
            xdelaylabel = "%s%s" % (labels[i[1]], labels[i[0]])
            ydelaylabel = "%s%s" % (labels[j[1]], labels[j[0]])
            retdict["cov"]["%s-%s" % (ydelaylabel, xdelaylabel)] = {}

            if detailplots:
                # figure 3: for each pair, plot the covariance in each bin. One figure per pair
                bincovplot2 = plt.figure(figsize=figsize)
                bincovplot2.subplots_adjust(left=left, right=right, bottom=bottom, top=top, wspace=wspace,
                                            hspace=hspace)

            ytderrs = [tderrsdict["tderrs"][jj] for tderrsdict in tderrsdicts]
            ytruetds = [tderrsdict["truetds"][jj] for tderrsdict in tderrsdicts]

            ax1 = allcovplot.add_subplot(ncouples, ncouples, axisNum)
            ax2 = bincovplot.add_subplot(ncouples, ncouples, axisNum)

            majorLocator = MultipleLocator(2.0)
            ax1.set_xlim(-rerr, rerr)
            ax1.set_ylim(-rerr, rerr)
            ax1.xaxis.set_major_locator(majorLocator)
            ax1.yaxis.set_major_locator(majorLocator)
            ax1.axhline(0, color="black")
            ax1.axvline(0, color="black")

            ax2.xaxis.set_major_locator(MaxNLocator(3))
            ax2.yaxis.set_major_locator(MaxNLocator(3))

            if axisNum == ncouples * (ncouples - 1) + 1:
                ax1.set_xlabel('Measurement error [day]')
                ax1.set_ylabel('Measurement error [day]')

                ax2.set_xlabel('True delay [day]')
                ax2.set_ylabel('True delay [day]')

            ## binning independent of xtrudelays and ytruedelays distribution. Same plotrange as diagonal elements, but 2d binning
            retdict["cov"]["%s-%s" % (ydelaylabel, xdelaylabel)]["indep"] = {}
            xbinlims2d = np.linspace(plotrange[0], plotrange[1], nbins2d + 1)

            yreftruedelay = reftrueshifts[j[0]] - reftrueshifts[j[1]]
            yplotrange = (yreftruedelay - r, yreftruedelay + r)
            ybinlims2d = np.linspace(yplotrange[0], yplotrange[1], nbins2d + 1)

            xcoordsan = []
            ycoordsan = []
            colorsan = []
            covsindep = []

            for indx, xbinlim in enumerate(xbinlims2d[:nbins2d]):
                for indy, ybinlim in enumerate(ybinlims2d[:nbins2d]):
                    subsamples = []
                    for (ind, xtruetd), ytruetd in zip(enumerate(xtruetds), ytruetds):
                        if xbinlim < xtruetd < xbinlims2d[indx + 1] and ybinlim < ytruetd < ybinlims2d[indy + 1]:
                            subsamples.append((xtderrs[ind], ytderrs[ind]))

                    if len(subsamples) > minsamples:
                        covval = np.cov(subsamples, rowvar=False)[0][1]
                        colorsan.append("black")
                    else:
                        covval = 0
                        colorsan.append('crimson')

                    # save the plotting coords, to bold plot the biggest covval later...
                    xcoordsan.append(xbinlim + (xbinlims2d[indx + 1] - xbinlim) / 2)
                    ycoordsan.append(ybinlim + (ybinlims2d[indy + 1] - ybinlim) / 2)

                    # colorize the regions according to the covariance value
                    maxval = 0.5
                    alpha = min(np.abs(covval / maxval), 1.0)
                    from matplotlib.patches import Rectangle
                    rect = Rectangle((xbinlim, ybinlim), xbinlims2d[indx + 1] - xbinlim, ybinlims2d[indy + 1] - ybinlim,
                                     color=rrlist[0].plotcolour, alpha=alpha)
                    ax2.add_patch(rect)

                    xdelaylabeldet = "%s%s [%.1f , %.1f]" % (labels[i[1]], labels[i[0]], xbinlim, xbinlims2d[indx + 1])
                    ydelaylabeldet = "%s%s [%.1f , %.1f]" % (labels[j[1]], labels[j[0]], ybinlim, ybinlims2d[indy + 1])

                    retdict["cov"]["%s-%s" % (ydelaylabel, xdelaylabel)]["indep"][
                        "%s-%s" % (ydelaylabeldet, xdelaylabeldet)] = covval
                    covsindep.append(covval)

                    if detailplots:
                        # add an Axes on the figure for each bin, and plot the errors
                        # mapping the maptlotlib indice is a bit tricky:
                        # if we use nbins2dx and nbins2dy: nbins2dx*nbins2dy - (nbins2dx-1-indx) - (nbins2dy*indy)
                        spind = nbins2d * nbins2d - (nbins2d - 1 - indx) - (nbins2d * indy)
                        ax3 = bincovplot2.add_subplot(nbins2d, nbins2d, spind)
                        ax3.set_xlim(-rerr, rerr)
                        ax3.set_ylim(-rerr, rerr)
                        ax3.xaxis.set_major_locator(majorLocator)
                        ax3.yaxis.set_major_locator(majorLocator)
                        ax3.axhline(0, color="black")
                        ax3.axvline(0, color="black")
                        ax3.set_xlabel('Measurement error [day]')
                        ax3.set_ylabel('Measurement error [day]')

                        showdensity = True
                        bins = 10
                        if showdensity:
                            cmap = colors.LinearSegmentedColormap.from_list('custom', ['white', rrlist[0].plotcolour],
                                                                            gamma=1.0)
                            ax3.hexbin([s[0] for s in subsamples], [s[1] for s in subsamples], gridsize=bins,
                                       extent=(-rerr, rerr, -rerr, rerr), mincnt=1, cmap=cmap, edgecolor="none")
                        showpoints = True
                        if showpoints:
                            ax3.scatter([s[0] for s in subsamples], [s[1] for s in subsamples], s=5,
                                        facecolor=rrlist[0].plotcolour, lw=0, alpha=0.5)
                        showcontour = True
                        if showcontour:
                            H, xedges, yedges = np.histogram2d([s[0] for s in subsamples], [s[1] for s in subsamples],
                                                               range=[[-r, r], [-r, r]], bins=(bins, bins))
                            extent = [yedges[0], yedges[-1], xedges[0], xedges[-1]]
                            data = np.vstack((xtderrs, ytderrs))
                            kde = scipy.stats.kde.gaussian_kde(data)
                            grid = np.mgrid[-r:r:1j * bins, -r:r:1j * bins]
                            grid_coords = np.append(grid[0].reshape(-1, 1), grid[1].reshape(-1, 1), axis=1)
                            z = kde(grid_coords.T)
                            z = z.reshape(bins, bins)
                            levels = [np.max(z) * 0.45]
                            cset = ax3.contour(grid[0], grid[1], z, levels=levels, origin="lower",
                                               colors=rrlist[0].plotcolour, extent=extent, linewidth=0.5)

                        if figsize[0] > 8:
                            ax3.annotate(xdelaylabeldet, xy=(0.77, 0.05), xycoords='axes fraction', ha="center")
                            ax3.annotate(ydelaylabeldet, xy=(0.04, 0.90), xycoords='axes fraction', ha="left",
                                         rotation=90.0)

            if detailplots and filepath is not None:
                bincovplot2.savefig(os.path.join(filepath, "bincov_%s%s-vs-%s%s.png" % (
                    labels[j[1]], labels[j[0]], labels[i[1]], labels[i[0]])))

            mincovindep = np.min(covsindep)
            maxcovindep = np.max(covsindep)

            if abs(mincovindep) > maxcovindep:
                extcovindep = mincovindep
            else:
                extcovindep = maxcovindep

            mind = covsindep.index(extcovindep)
            for ind, val in enumerate(covsindep):
                if ind == mind:
                    ax2.annotate("%.2f" % val, xy=(xcoordsan[ind], ycoordsan[ind]), ha="center", va='center',
                                 color='darkblue', fontsize=14)
                else:
                    ax2.annotate("%.2f" % val, xy=(xcoordsan[ind], ycoordsan[ind]), ha="center", va='center',
                                 color=colorsan[ind])

            # plotting ax2 uses the 2d binning
            for ind, xbinlim in enumerate(xbinlims2d):
                ax2.axvline(xbinlim, linestyle='--', color='black', alpha=0.5)
                ax2.axhline(ybinlims2d[ind], linestyle='--', color='black', alpha=0.5)
            showpoints = False
            if showpoints:
                ax2.scatter(xtruetds, ytruetds, s=2, facecolor=rrlist[0].plotcolour, lw=0, alpha=0.1)

            ax2.set_xlim(plotrange)
            ax2.set_ylim(yplotrange)

            # plotting ax1 is pretty basic, that's only the points
            retdict["cov"]["%s-%s" % (ydelaylabel, xdelaylabel)]["dep"] = {}
            showdensity = True
            bins = 10
            if showdensity:
                cmap = colors.LinearSegmentedColormap.from_list('custom', ['white', rrlist[0].plotcolour], gamma=1.0)
                ax1.hexbin(xtderrs, ytderrs, gridsize=bins, extent=(-rerr, rerr, -rerr, rerr), mincnt=1, cmap=cmap,
                           edgecolor="none")

            showpoints = False
            if showpoints:
                ax1.scatter(xtderrs, ytderrs, s=2, facecolor=rrlist[0].plotcolour, lw=0)

            showcontour = True
            if showcontour:
                H, xedges, yedges = np.histogram2d(xtderrs, ytderrs, range=[[-r, r], [-r, r]], bins=(bins, bins))
                extent = [yedges[0], yedges[-1], xedges[0], xedges[-1]]

                data = np.vstack((xtderrs, ytderrs))
                kde = scipy.stats.kde.gaussian_kde(data)

                grid = np.mgrid[-r:r:1j * bins, -r:r:1j * bins]
                grid_coords = np.append(grid[0].reshape(-1, 1), grid[1].reshape(-1, 1), axis=1)

                z = kde(grid_coords.T)
                z = z.reshape(bins, bins)

                levels = [np.max(z) * 0.45]
                cset = ax1.contour(grid[0], grid[1], z, levels=levels, origin="lower", colors=rrlist[0].plotcolour,
                                   extent=extent)

            if figsize[0] > 8:
                ax1.annotate(xdelaylabel, xy=(0.9, 0.05), xycoords='axes fraction', ha="center")  # x axis
                ax1.annotate(ydelaylabel, xy=(0.06, 0.85), xycoords='axes fraction', ha="left", rotation=90.0)  # y axis
            else:
                ax1.annotate(xdelaylabel, xy=(0.78, 0.08), xycoords='axes fraction', ha="center")  # x axis
                ax1.annotate(ydelaylabel, xy=(0.08, 0.76), xycoords='axes fraction', ha="left", rotation=90.0)  # y axis

            meancov = np.cov([(xtderr, ytderr) for xtderr, ytderr in zip(xtderrs, ytderrs)], rowvar=False)[0][1]
            ax2.set_title('%s vs %s | mean = %.2f' % (ydelaylabel, xdelaylabel, meancov), fontsize=10)

            ## binning dependent of true delays, for comparision
            xbinvals = np.linspace(minx, maxx, num=nbins2d + 1, endpoint=True)
            maxy = np.max(ytruetds)
            miny = np.min(ytruetds)
            ybinvals = np.linspace(miny, maxy, num=nbins2d + 1, endpoint=True)

            covsdep = []
            for indx, xbinval in enumerate(xbinvals[:nbins2d]):
                for indy, ybinval in enumerate(ybinvals[:nbins2d]):
                    subsamples = []
                    for (ind, xtruetd), ytruetd in zip(enumerate(xtruetds), ytruetds):
                        if xbinval < xtruetd < xbinvals[indx + 1] and ybinval < ytruetd < ybinvals[indy + 1]:
                            subsamples.append((xtderrs[ind], ytderrs[ind]))

                    xdelaylabeldet = "%s%s [%.1f , %.1f]" % (labels[i[1]], labels[i[0]], xbinval, xbinvals[indx + 1])
                    ydelaylabeldet = "%s%s [%.1f , %.1f]" % (labels[j[1]], labels[j[0]], ybinval, ybinvals[indy + 1])
                    if len(subsamples) > minsamples:
                        covvaldep = np.cov(subsamples, rowvar=False)[0][1]
                    else:
                        covvaldep = 0.0
                    retdict["cov"]["%s-%s" % (ydelaylabel, xdelaylabel)]["dep"][
                        "%s-%s" % (ydelaylabeldet, xdelaylabeldet)] = covvaldep
                    covsdep.append(covvaldep)

            # We do NOT want the min or max in the final covmat but the mean on all samples.
            # do NOT take the mean of covsdep, some samples are not in !!
            covdep = meancov
            covindep = meancov

            if method == "depbin":
                covmat[ii][jj] = covdep
                covmat[jj][ii] = covdep
            elif method == "indepbin":
                covmat[ii][jj] = covindep
                covmat[jj][ii] = covindep

            if verbose:
                logger.info("-" * 15)
                logger.info(f"{i} , {j}")
                logger.info(f"{covdep}, {covindep}")

    axinv = bincovplot.add_subplot(ncouples, ncouples, 2, frameon=False)
    axinv.set_xticklabels([])
    axinv.set_yticklabels([])
    axinv.set_xticks([])
    axinv.set_yticks([])

    # and annotate

    text = 'True delay plot range: +- %i [days]' % r + '\n\n'
    text += 'Measurement error plot range: +- %.1f [days]' % rerr + '\n\n'
    text += '1D binning:  %i bins' % nbins + '\n\n'
    text += '2D binning:  %ix%i bins' % (nbins2d, nbins2d) + '\n\n'
    text += 'Min. number of samples in 2D binning:  %i samples' % minsamples + '\n\n\n\n'

    if printdetails:
        if len(covmat[0]) == 6:
            mylist = [str(e) for e in covmat[0]] + \
                     [str(e) for e in covmat[1]] + \
                     [str(e) for e in covmat[2]] + \
                     [str(e) for e in covmat[3]] + \
                     [str(e) for e in covmat[4]] + \
                     [str(e) for e in covmat[5]]
            mylist = [float(e) for e in mylist]
        else:
            logger.warning("Cov. matrix display not defined for matrices other than 6x6 !")
            printcovmat = False

        if printcovmat:
            text += '     AB        AC        AD        BC        BD        CD \n'
            text += '     ' + '-----' * 12 + '\n'
            text += 'AB | %.2f    %.2f    %.2f    %.2f    %.2f    %.2f \n     |\n' \
                    'AC | %.2f    %.2f    %.2f    %.2f    %.2f    %.2f \n     |\n' \
                    'AD | %.2f    %.2f    %.2f    %.2f    %.2f    %.2f \n     |\n' \
                    'BC | %.2f    %.2f    %.2f    %.2f    %.2f    %.2f \n     |\n' \
                    'BD | %.2f    %.2f    %.2f    %.2f    %.2f    %.2f \n     |\n' \
                    'CD | %.2f    %.2f    %.2f    %.2f    %.2f    %.2f \n     |\n' \
                    % (mylist[0], mylist[1], mylist[2], mylist[3], mylist[4], mylist[5]
                       , mylist[6], mylist[7], mylist[8], mylist[9], mylist[10], mylist[11]
                       , mylist[12], mylist[13], mylist[14], mylist[15], mylist[16], mylist[17]
                       , mylist[18], mylist[19], mylist[20], mylist[21], mylist[22], mylist[23]
                       , mylist[24], mylist[25], mylist[26], mylist[27], mylist[28], mylist[29]
                       , mylist[30], mylist[31], mylist[32], mylist[33], mylist[34], mylist[35])

            axinv.annotate(text, xy=(0.7 * (ncouples - 1), -2.0), xycoords='axes fraction', ha="left")
        else:
            axinv.annotate(text, xy=(0.7 * (ncouples - 1), -1.0), xycoords='axes fraction', ha="left")

    retdict["r"] = r
    retdict["rerr"] = rerr
    retdict["nbins"] = nbins
    retdict["nbins2d"] = nbins2d
    retdict["minsamples"] = minsamples

    if filepath is not None:
        bincovplot.savefig(os.path.join(filepath, "bincov.png"))
        allcovplot.savefig(os.path.join(filepath, "allcov.png"))

    else:
        if showplots:
            plt.show()

    # now let's compare indepbins and depbins
    if verbose:
        logger.info("-" * 35)
        logger.info("nbins = %i" % nbins)
        logger.info("indepbins - r = %.1f" % r)
        logger.info("depbins - r(max-min) = %.1f"%np.mean(rranges))
        logger.info("-" * 35)
        logger.info("pair - indepbins - depbins - diff")
        logger.info("-" * 35)
        logger.info("AB - %.2f - %.2f - %.1f%%" % (indepbins[0], depbins[0],
                                                   (max(indepbins[0], depbins[0]) - min(indepbins[0],
                                                                                        depbins[0])) / max(
                                                       indepbins[0], depbins[0]) * 100))
        logger.info("AC - %.2f - %.2f - %.1f%%" % (indepbins[1], depbins[1],
                                                   (max(indepbins[1], depbins[1]) - min(indepbins[1],
                                                                                        depbins[1])) / max(
                                                       indepbins[1], depbins[1]) * 100))
        if nimages == 4:
            logger.info("BC - %.2f - %.2f - %.1f%%" % (indepbins[3], depbins[3],
                                                       (max(indepbins[3], depbins[3]) - min(indepbins[3],
                                                                                            depbins[3])) / max(
                                                           indepbins[3], depbins[3]) * 100))
            logger.info("AD - %.2f - %.2f - %.1f%%" % (indepbins[2], depbins[2],
                                                       (max(indepbins[2], depbins[2]) - min(indepbins[2],
                                                                                            depbins[2])) / max(
                                                           indepbins[2], depbins[2]) * 100))
            logger.info("BD - %.2f - %.2f - %.1f%%" % (indepbins[4], depbins[4],
                                                       (max(indepbins[4], depbins[4]) - min(indepbins[4],
                                                                                            depbins[4])) / max(
                                                           indepbins[4], depbins[4]) * 100))
            logger.info("CD - %.2f - %.2f - %.1f%%" % (indepbins[5], depbins[5],
                                                       (max(indepbins[5], depbins[5]) - min(indepbins[5],
                                                                                            depbins[5])) / max(
                                                           indepbins[5], depbins[5]) * 100))
        elif nimages == 3:
            logger.info("BC - %.2f - %.2f - %.1f%%" % (indepbins[2], depbins[2],
                                                       (max(indepbins[2], depbins[2]) - min(indepbins[2],
                                                                                            depbins[2])) / max(
                                                           indepbins[2], depbins[2]) * 100))
        logger.info("-" * 35)

    retdict["covmat"] = covmat
    return retdict


def measvstrue(rrlist, r=10.0, nbins=10, plotpoints=True, alphapoints=1.0, plotrods=True, alpharods=0.2,
               ploterrorbars=True, sidebyside=True, errorrange=None, binclip=False, binclipr=10.0, title=None,
               xtitle=0.75, ytitle=0.95, titlesize=30, figsize=(10, 6), left=0.06, right=0.97, top=0.99, bottom=0.08,
               wspace=0.15, hspace=0.3, txtstep=0.04, majorticksstep=2, displayn=True, filename=None, dataout=False,
               tweakeddisplay=False, blindness=False, outdir="./"):
    """

    Plots measured delays versus true delays

    :param r: radius of simulation input delays to plot (x axis range)
    :param nbins: number of bins for the bar plot within this range.
    :param plotpoints: should I plot the points (scatter plot) ?
    :param plotrods: should I plot the avg within each bin ?
    :param ploterrorbars: should I add errorbars upon the bar plot ?
    :param sidebyside: should I plot bars side by side, or overplot them ?
    :param errorrange: radius of measurement errors to plot (y axis range). You can also give a tuple (low, high), to make asymetric plots.
    :param binclip: should I clip errors larger than binclipr days (catastrophic failures of methods) ?
    :param binclipr: see binclip ...

    """

    n = rrlist[0].nimages()

    labels = rrlist[0].labels

    # To get some fixed ranges for the histograms, we will use the first element of rrlist.
    reftrueshifts = np.round(rrlist[0].gettruets()["center"])

    for rr in rrlist:
        if rr.labels != labels:
            raise RuntimeError("Don't ask me to overplot runresults of different curves !")

        rr.trues = rr.gettruets()  # To store this, avoids calculating it again and again.

        rr.tmpdata = []

    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top, wspace=wspace, hspace=hspace)

    axisNum = 0
    for i in range(n):
        for j in range(n):
            if (i == 0) or (j == n - 1):
                continue  # No plot
            axisNum += 1
            if j >= i:
                continue

            ax = plt.subplot(n - 1, n - 1, axisNum)

            minorLocator = MultipleLocator(1.0)
            majorLocator = MultipleLocator(majorticksstep)
            ax.xaxis.set_minor_locator(minorLocator)
            ax.xaxis.set_major_locator(majorLocator)
            if tweakeddisplay:
                from matplotlib.ticker import MaxNLocator
                locator = MaxNLocator(prune='both', nbins=6)
                ax.yaxis.set_major_locator(locator)
            ax.yaxis.set_minor_locator(MultipleLocator(1.0))

            reftruedelay = reftrueshifts[i] - reftrueshifts[j]
            plotrange = (reftruedelay - r, reftruedelay + r)

            # Identity line :
            line = np.linspace(plotrange[0], plotrange[1], 100)
            zeros = np.zeros(100)
            plt.plot(line, zeros, color="black", lw=0.5)

            # Delay label
            delaylabel = "%s%s" % (labels[j], labels[i])

            # Preparing the bins :
            binlims = np.linspace(plotrange[0], plotrange[1], nbins + 1)

            for irr, rr in enumerate(rrlist):  # We go through the different runresult objects
                # We will express the delays "i - j"
                truedelays = rr.truetsarray[:, i] - rr.truetsarray[:, j]
                measdelays = rr.tsarray[:, i] - rr.tsarray[:, j]

                resis = measdelays - truedelays

                # A simple scatter plot of the residues :
                if plotpoints:
                    ax.scatter(truedelays, resis, s=2, facecolor=rr.plotcolour, lw=0, alpha=alphapoints)

                # We bin those :
                digitized = np.digitize(truedelays, binlims)

                binvals = [resis[digitized == bini] for bini in range(1, len(binlims))]
                binstds = list(map(np.std, binvals))
                binmedians = list(map(np.median, binvals))
                binmeans = list(map(np.mean, binvals))

                if binclip:
                    for (bini, binvalarray) in enumerate(binvals):
                        keep = np.logical_and(binvalarray < binclipr, binvalarray > -binclipr)
                        if np.sum(keep == False) != 0:
                            n_kick = np.sum(keep == False)
                            logging.info("Kicking %i points." % (n_kick))
                            if n_kick > 0.2 * len(binvalarray):
                                logging.warning(
                                    'Kicking more than 20 percent of your curves. Something went probably wrong ! Please consider using other estimator parameters.')
                        binvals[bini] = binvalarray[keep]
                    binstds = list(map(np.std, binvals))
                    binmedians = list(map(np.median, binvals))
                    binmeans = list(map(np.mean, binvals))

                # We save the maximum sys and ran error :

                syserror = np.max(np.fabs(binmeans))
                randerror = np.max(binstds)
                toterror = np.sqrt(syserror * syserror + randerror * randerror)

                bias = np.mean(binmeans)  # The signed bias
                rr.tmpdata.append({
                    "label": delaylabel,
                    "sys": syserror,
                    "ran": randerror,
                    "tot": toterror,
                    "bias": bias
                })

                width = binlims[1] - binlims[0]

                if plotrods:
                    if not sidebyside:
                        if ploterrorbars:
                            ax.bar(binlims[:-1] + width / 2.0, binmeans, yerr=binstds, width=width, color=rr.plotcolour,
                                   ecolor=rr.plotcolour,
                                   error_kw={"capsize": 2.5, "capthick": 0.5, "markeredgewidth": 0.5},
                                   edgecolor=rr.plotcolour, alpha=alpharods)
                        else:
                            ax.bar(binlims[:-1] + width / 2.0, binmeans, width=width, color=rr.plotcolour,
                                   edgecolor=rr.plotcolour, alpha=alpharods)
                    else:
                        width = width / len(rrlist)
                        squeezefactor = 1.0
                        plotwidth = squeezefactor * width
                        offset = width * (1.0 - squeezefactor) / 2.0

                        if ploterrorbars:
                            ax.bar(binlims[:-1] + width / 2.0 + offset + irr * plotwidth, binmeans, yerr=binstds,
                                   width=plotwidth, color=rr.plotcolour, ecolor=rr.plotcolour,
                                   error_kw={"capsize": 2.5, "capthick": 0.5, "markeredgewidth": 0.5},
                                   edgecolor=rr.plotcolour, alpha=alpharods, linewidth=0)
                        else:
                            ax.bar(binlims[:-1] + width / 2.0 + offset + irr * plotwidth, binmeans, width=plotwidth,
                                   color=rr.plotcolour, edgecolor=rr.plotcolour, alpha=alpharods)

            # That's it for the different runresult objects, back to the common stuff for this particular panel :
            if sidebyside:
                for binlim in binlims:
                    plt.axvline(binlim, lw=0.5, color="#AAAAAA", zorder=-30)

            # Just on 2 plots :
            if tweakeddisplay:
                if i == n - 1:
                    plt.xlabel("True delay [day]", fontsize=18)
                if j == 0 and i == int(math.floor(n / 2.0)):
                    plt.ylabel("Delay measurement error [day]", fontsize=18, y=-0.10)
                plt.xticks(fontsize=15)
                plt.yticks(fontsize=15)
            else:
                if i == n - 1:
                    plt.xlabel("True delay [day]", fontsize=16)
                if j == 0 and i == int(math.floor(n / 2.0)):
                    plt.ylabel("Delay measurement error [day]", fontsize=16)
                plt.xticks(fontsize=13)
                plt.yticks(fontsize=13)
                xmin, xmax = ax.get_xlim()
                ax.set_xticks(np.round(np.linspace(xmin, xmax, 10)))

            plt.xlim(plotrange)
            if errorrange is not None:
                if hasattr(errorrange, '__iter__'):  # then its a tuple or list
                    plt.ylim((errorrange[0], errorrange[1]))
                else:
                    plt.ylim((-errorrange, errorrange))

            if n != 2:  # otherwise we have only 1 delay and panel
                plt.annotate(delaylabel, xy=(0.03, 0.88 - txtstep), xycoords='axes fraction', fontsize=14,
                             color="black")

            # enforce blindness if wanted, by modifying the xticks labels (not touching the data)
            if blindness:
                labels = ax.xaxis.get_ticklabels()
                locs = ax.xaxis.get_ticklocs()
                meanloc = np.mean(locs)
                blindlabels = []
                for loc, label in zip(locs, labels):
                    blindlabels.append(str(loc - meanloc))
                ax.xaxis.set_ticklabels(blindlabels)

        # That's it for this panel, back to the total figure :

    if dataout:
        for rr in rrlist:
            dc = DelayContainer(data=rr.tmpdata, name=rr.name, plotcolour=rr.plotcolour, objects=labels[:])
            pycs3.gen.util.writepickle(dc, os.path.join(outdir, "%s_errorbars.pkl" % rr.autoname))
            rr.tmpdata = None

    labelspacetop = 0.0
    labelspaceright = 0.0
    if n == 2:
        labelspacetop = 0.04
        labelspaceright = 0.04

    for irr, rr in enumerate(rrlist):

        if displayn:
            labeltxt = "%s (%i) " % (getattr(rr, 'name', 'NoName'), rr.tsarray.shape[0])
        else:
            labeltxt = "%s" % (getattr(rr, 'name', 'NoName'))
        if not tweakeddisplay:
            plt.figtext(x=right - labelspaceright, y=top - labelspacetop - txtstep * irr, s=labeltxt,
                        verticalalignment="top", horizontalalignment="right", color=rr.plotcolour, fontsize=15)
        else:
            plt.figtext(x=0.54, y=0.8325 - txtstep * irr, s=labeltxt, verticalalignment="top",
                        horizontalalignment="left", color=rr.plotcolour, fontsize=17)

    if title is not None:
        plt.figtext(x=xtitle, y=ytitle, s=title, horizontalalignment="center", color="black", fontsize=titlesize)

    if filename is None:  # pragma: no cover
        plt.show()
    else:
        plt.savefig(filename)
