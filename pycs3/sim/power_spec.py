"""
Module to define the PowerSpectrum class and associated functions. This can be used to measure the slope of the PowerSpectrum
and adjust the argument in the py:mod:`pycs3.sim.twk.tweakspl` function, when using ``colored_noise`` for generating the mock curves.
"""
import copy as pythoncopy
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as spopt
import logging
logger = logging.getLogger(__name__)


def window_hanning(x):
    """return x times the hanning window of len(x)"""
    return np.hanning(len(x)) * x


class PowerSpectrum:
    """
    A class representing a power spectrum of a Source object.
    """
    def __init__(self, source, flux=False):
        """
        Constructor, simply takes a Source object and calculates the power spectrum

        The Source is always expressed in magnitudes, but you might want to get a power spectrum in terms
        of flux or "counts".
        Put flux = True, and I will convert your magnitudes into fluxes before calculating the power spectrum.
        """

        self.flux = flux
        self.n = len(source.ijds)  # Even by construction.

        if not flux:
            x = source.imags
        else:
            x = 10.0 ** (-0.4 * source.imags)

        # Note that these frequencies of self.f have physical units, they are directly in days^-1 !

        # An alternative using only fft by myself, strongly "inspired" by the mlab psd above
        # (It gives the same results, of course, but avoids the stupid figure)

        # The frequencies, and preparing a window :
        self.f = np.linspace(0, 0.5 / source.sampling, int(self.n / 2 + 1))
        windowvals = window_hanning(np.ones(self.n))

        # The FFT and power :
        fx = np.fft.rfft(windowvals * x)
        p = np.abs(fx) ** 2

        # Scale the spectrum by the norm of the window to compensate for
        # windowing loss; see Bendat & Piersol Sec 11.5.2.
        p *= 1 / (np.abs(windowvals) ** 2).sum()

        # Also include scaling factors for one-sided densities and dividing by the
        # sampling frequency, if desired. Scale everything, except the DC component
        # and the NFFT/2 component:
        p[1:-1] *= 2.0 * source.sampling
        # But do scale those components by Fs, if required
        p[[0, -1]] *= source.sampling

        self.p = p

        self.plotcolour = source.plotcolour

        self.slope = None
        self.name = source.name

    def __str__(self):
        if self.flux:
            return "%s(flux)" % str(self.name)
        else:
            return "%s(mag)" % str(self.name)

    def copy(self):
        """
        Return a copy of itself.

        """
        return pythoncopy.deepcopy(self)

    def calcslope(self, fmin=1. / 1000.0, fmax=1. / 2.0):
        """
        Measures the slope of the PS, between fmin and fmax.
        All info about this slope is sored into the dictionary self.slope.
        This is just fitting the powerspectrum as given by the constructor.

        """
        if fmin is None:
            fmin = self.f[1]
        if fmax is None:
            fmax = self.f[-1]

        reg = np.logical_and(self.f <= fmax, self.f >= fmin)

        fitx = np.log10(self.f[reg]).flatten()
        fity = np.log10(self.p[reg]).flatten()

        if not np.all(np.isfinite(fity)):
            logger.info("Skipping calcsclope for flat function !")
            return

        self.slope = {}

        def func(x, m, h):
            return m * x + h

        popt = spopt.curve_fit(func, fitx, fity, p0=[0.0, 0.0])[0]
        sepfit = func(fitx, popt[0], popt[1])  # we evaluate the linear fit.

        self.slope["slope"] = popt[0]
        self.slope["f"] = 10.0 ** fitx
        self.slope["p"] = 10.0 ** sepfit
        self.slope["fmin"] = fmin
        self.slope["fmax"] = fmax


def psplot(pslist, nbins=0, filename=None, figsize=(12, 8), showlegend=True):
    """
    Plots a list of PowerSpectrum objects.
    If the PS has a slope, it is plotted as well.

    if nbins > 0, I bin the spectra.

    add option for linear plot ?
    """

    plt.figure(figsize=figsize)
    for ps in pslist:

        if not np.all(np.isfinite(np.log10(ps.p))):
            logger.info("No power to plot (probably flat curve !), skipping this one.")
            continue
        # We bin the points

        if nbins > 0:
            logf = np.log10(ps.f[1:])  # we remove the first one
            logbins = np.linspace(np.min(logf), np.max(logf), nbins + 1)  # So nbins +1 numbers here.
            bins = 10 ** logbins
            bincenters = 0.5 * (bins[:-1] + bins[1:])  # nbins centers
            logbins[0] -= 1.0
            logbins[-1] += 1.0
            binindexes = np.digitize(logf, logbins)  # binindexes go from 1 to nbins+1
            binvals = []
            binstds = []
            for i in range(1, nbins + 1):
                vals = ps.p[1:][binindexes == i]
                binvals.append(np.mean(vals))
                binstds.append(np.std(vals) / np.sqrt(vals.size))

            bincenters = np.array(bincenters)
            binvals = np.array(binvals)
            binstds = np.array(binstds)

            plt.loglog(bincenters, binvals, marker=".", linestyle="-", color=ps.plotcolour, label="%s" % ps)

        else:
            plt.loglog(ps.f, ps.p, marker=".", linestyle="None", color=ps.plotcolour, label="%s" % ps)
        if ps.slope is not None:
            plt.loglog(ps.slope["f"], ps.slope["p"], marker="None", color=ps.plotcolour,
                       label="Slope %s = %.3f" % (ps, ps.slope["slope"]))
            plt.axvline(ps.slope["fmin"], color=ps.plotcolour, dashes=(5, 5))
            plt.axvline(ps.slope["fmax"], color=ps.plotcolour, dashes=(5, 5))

    plt.xlabel("Frequency [1/days]")
    plt.ylabel("Power")

    if showlegend:
        plt.legend()

    if filename:
        plt.savefig(filename)
    else: # pragma: no cover
        plt.show()
