"""
Stuff to represent a light curve as points on a regular grid, and power spectrum plots.
This allows e.g. to tweak its power spectrum by adding "correlated noise", resample it, etc.

"""
import copy as pythoncopy

import matplotlib.pyplot as plt
import numpy as np
import pycs3.gen.datapoints
import pycs3.gen.spl
import scipy.interpolate as si
import logging
logger = logging.getLogger(__name__)


class Source:
    """
    A class representing a "source" lightcurve, i.e. any artificial signal whose powerspectrum we can modify,
    and still evaluate the curve at given irregular jds.

    To do this, we work with "internal regular arrays" (and do FFTs and so on on these), and then interpolate these
    fine arrays to get back values for given jds.

    We store the data in magnitudes.

    Ideas : use the mask carried around by self.inispline.datapoints to define regions that
    really matter for the power spectum and stuff.

    """

    def __init__(self, spline=None, name="Source", range=(0, 10000), sampling=0.2):
        """
        range are in jds, only used if you don't give me a sourcespline.
        sampling is in days.

        I will store the spline as inispline, but I will not modify it, so no need to pass me a copy.

        """
        if spline is not None:
            self.inispline = spline

            self.jdmin = spline.datapoints.jds[0]
            self.jdmax = spline.datapoints.jds[-1]
            self.plotcolour = spline.plotcolour
        else:

            # Some default spline to play with and for tests :
            self.jdmin = range[0]
            self.jdmax = range[-1]
            jds = np.linspace(self.jdmin, self.jdmax, 100)
            mags = np.zeros(jds.shape)
            magerrs = 0.1 * np.ones(jds.shape)
            dp = pycs3.gen.datapoints.DataPoints(jds, mags, magerrs, splitup=False, sort=False)
            self.inispline = pycs3.gen.spl.Spline(dp)
            self.inispline.uniknots(3)  # Puts coeffs to 0, that's ok.
            self.plotcolour = "grey"

        self.name = name

        self.sampling = sampling  # Number of days between points of the internal arrays.
        # Note that we want an even number of internal points.
        # Hence, this sampling will be adapted a bit so to make the number of points even.
        self.setup()

    def setup(self):
        """
        Sets the internal regular arrays by sampling the spline.
        We update self.sampling, so to get an even number of points (for FFTs).
        """
        nsteps = int(((self.jdmax - self.jdmin) / self.sampling) / 2.0) * 2
        self.sampling = float(self.jdmax - self.jdmin) / float(nsteps)
        self.ijds = np.linspace(self.jdmin, self.jdmax, nsteps)
        self.imags = self.inispline.eval(jds=self.ijds)

    def __str__(self):
        return "%s (%i)" % (str(self.name), len(self.ijds))

    def copy(self):
        """
        Return a copy of itself.

        """
        return pythoncopy.deepcopy(self)

    def setmean(self, mean=-12.0):
        """
        Shifts the magnitudes so that their mean value are at the specified level.
        Don't do this, just for testing purposes !
        When doing flux PS, this scales the power, that's all.
        """
        self.imags += mean - np.mean(self.imags)

    def addgn(self, sigma=0.1, seed=None):
        """
        Don't do this, just for testing purposes ! There is no reason to add white noise to the source !
        """
        rs = np.random.RandomState(seed)  # we create a random state object, to control the seed.
        self.imags += rs.standard_normal(self.imags.size) * sigma

    def addrw(self, sigma=0.1, seed=None):
        """
        Add a random walk, also for experiment (power law param is -2)
        """
        rs = np.random.RandomState(seed)  # we create a random state object, to control the seed.
        randvec = rs.standard_normal(self.imags.size) * sigma
        self.imags += np.cumsum(randvec)

    def addplaw2(self, beta=-3.0, sigma=0.01, flux=False, fmin=None, fmax=None, hann=False, seed=None):
        """
        Next version, better
        Adds noise according to a power law PSD.
        See Timmer & Koenig 1995

        power law would be -2

        if hann, we soften the window (Hann window instead of tophat).

        """
        # To simplify, we will generate a symmetric curve and use only half of it. Hence this symmetric curve will be twice as long.

        n = self.imags.size  # The real number of points in our curve. This is even by construction.
        n2 = 2 * n  # This is even more even.

        # The positive FFT frequencies, in relative units :
        freqs2 = np.linspace(0, 0.5, int(n2 / 2 + 1))  # The 0 frequency is included, but we will leave it's power at 0.

        # To help representing what we do, here are the same in days^-1 :
        freqs = np.linspace(0, 0.5 / self.sampling, int(n2 / 2 + 1))  # Our associated frequencies, in units of days^-1

        # Now we generate the random coefficents for those freqs.
        rs = np.random.RandomState(seed)  # we create a random state object, to control the seed.

        # Complex and imaginary part
        rspecs = rs.standard_normal(int(n2 / 2 + 1))  # same length as freqs2, associated
        rspecs[1:] *= freqs2[1:] ** (beta / 2.0)
        ispecs = rs.standard_normal(int(n2 / 2 + 1))
        ispecs[1:] *= freqs2[1:] ** (beta / 2.0)
        rspecs[0] = 0.0  # To get 0 power for the 0 frequency.
        ispecs[0] = 0.0  # To get 0 power for the 0 frequency.

        # As we work with even number of signals, the Nyquist frequency term is real :
        ispecs[-1] = 0
        specs = rspecs + 1j * ispecs

        # We now build a mask
        if fmin is None:  # if fmin is None
            fmin = freqs[0] - 1.0
        if fmax is None:
            fmax = freqs[-1] + 1.0
        windowmask = np.logical_and(freqs <= fmax, freqs >= fmin)  # We will later set everything outside of this to 0.0

        outofwindowmask = windowmask == False  # we invert the mask
        specs[outofwindowmask] = 0.0

        if hann:
            bell = np.zeros(freqs2.shape)
            bell[windowmask] = np.hanning(np.sum(windowmask))
            specs *= bell

        # We add the points to the values
        if flux:
            logger.warning("Working with fluxes, check that sigma is scaled for fluxes and not magnitude.")
            iflux = 10.0 ** (-0.4 * self.imags)  # The fluxes
            iflux += sigma * np.fft.irfft(specs)[:int(n2 / 2)]  # We add our "noise"
            self.imags = -2.5 * np.log10(iflux)  # and get back to fluxes
        else:
            self.imags -= sigma * np.fft.irfft(specs)[:int(n2 / 2)]  # -, to make it coherent with the fluxes.

    def eval(self, jds):
        """
        I interpolate linearly my ijds/imags to give you an array of mags corresponding to your jds
        This could in principle be done using the spline object made by the function below, but this is safer and faster.
        """
        if np.min(jds) < self.jdmin or np.max(jds) > self.jdmax:
            raise RuntimeError("Sorry, your jds are out of bound !")

        f = si.interp1d(self.ijds, self.imags, kind="linear", bounds_error=True)
        return f(jds)

    def generate_spline(self):
        """
        I return a new pycs.gen.spl.Spline object corresponding to the source.
        So this is a bit the inverse of the constructor.
        You can then put this spline object as ML of a lightcurve, as source spline, or whatever.

        Note that my output spline has LOTs of knots... it is an interpolating spline, not a regression spline !


        ..note:: This spline is a priori for display purposes only. To do an interpolation, it might be safer (and faster) to use
            the above linear interpolation eval() function.
            But making a plot, you'll see that this spline seems well accurate.
        """

        x = self.ijds.copy()
        y = self.imags.copy()
        magerrs = np.zeros(len(x))

        out = si.splrep(x, y, w=None, xb=None, xe=None, k=3, task=0, s=0.0, t=None, full_output=True, per=False, quiet=True)

        tck = out[0]

        # From this we want to build a real Spline object.
        datapoints = pycs3.gen.datapoints.DataPoints(x, y, magerrs, splitup=False, sort=False, stab=False)
        outspline = pycs3.gen.spl.Spline(datapoints, t=tck[0], c=tck[1], k=tck[2], plotcolour=self.plotcolour)

        outspline.knottype = "MadeBySource"
        outspline.showknots = False  # Usually we have a lot of them, slows down.
        return outspline


def sourceplot(sourcelist, filename=None, figsize=(12, 8), showlegend=True, showspline=True, marker=None):
    """
    I show you a plot of a list of Source objects.
    """

    plt.figure(figsize=figsize)
    for s in sourcelist:
        if marker is None:
            plt.plot(s.ijds, s.imags, marker="None", color=s.plotcolour, linestyle="-", label="%s" % s.name)
        else:
            plt.plot(s.ijds, s.imags, marker=marker, color=s.plotcolour, linestyle="none", label="%s" % s.name)

        if showspline:
            spline = s.generate_spline()
            xs = np.arange(s.ijds[0], s.ijds[-1], 0.02)
            ys = spline.eval(jds=xs)
            plt.plot(xs, ys, "-", color="red", zorder=+20, label="%s.spline()" % s.name)

            plt.plot()

    # Something for astronomers only : we invert the y axis direction !
    axes = plt.gca()
    axes.set_ylim(axes.get_ylim()[::-1])
    plt.xlabel("HJD - 2400000.5 [days]", fontsize=14)
    plt.ylabel("Magnitude (relative)", fontsize=14)

    if showlegend:
        plt.legend()

    if filename:
        plt.savefig(filename)
    else:
        plt.show()
