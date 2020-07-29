"""
Microlensing represented by splines.
"""

import copy as pythoncopy

import numpy as np
from pycs3.gen.datapoints import DataPoints
from pycs3.gen.spl import Spline


class SplineML:
    """
    A lightcurve can have such a microlensing object as attribute "ml".
    We have only one spline per lightcurve (i.e., no seasons).

    The datapoint mags = what has to be added to the lightcurve so that it matches the sourcespline (or something else).
    This is the same convention set as for ML by polynoms.
    """

    def __init__(self, spline):
        """
        :param spline: Spline object that represent the microlensing
        :param mltype: Type of Microlensing
        :type mltype: str
        """

        self.spline = spline
        self.mltype = "spline"

    def __str__(self):
        return "|" + str(self.spline) + "|"

    def copy(self):
        """
        Return a copy of itself

        """
        return pythoncopy.deepcopy(self)

    def checkcompatibility(self, lightcurve):
        """
        It should be checked here that the self.datapoints are compatible with the lightcurve
        (i.e. same "relative" jds)
        """
        if not np.all(np.fabs(self.spline.datapoints.jds[self.spline.datapoints.mask] - (
                lightcurve.jds - lightcurve.jds[0])) < 0.001): # pragma: no cover
            raise RuntimeError("Ouch, this lightcurve is no longer compatible with its SplineML datapoints !")

    def settargetmags(self, lightcurve, sourcespline):
        """
        We update the self.spline.datapoints.mags so that a fit will results in a microlensing approximation.

        Sourcespline is a spline object that is the reference where you want your lightcurve to go.

        Only the mags are updated !
        Indeed for this microlensing spline I don't care if your lightcurve was shifted in time.
        But of course a shift in time with respect to the sourcespline will give you different mags !

        I hope you did not remove points or other tweaks.

        @todo: treat magerrs with a getter method !

        EDIT : Let's try to add some stabilization points in the datapoints...
        The constructor of the SplineML now defines how to calculate these stabilisation points.
        These settings are permanent for one SplineML, hence like without any stabpoints, all we need to do
        is to update the mags.
        No need to update the stab point jds, or the datapoints.mask etc.
        But, to update the mags, we need to calculate the mags for these stab points.

        """

        self.checkcompatibility(lightcurve)

        jdoffset = lightcurve.getjds()[0]  # That's the only stuff we'll use from the lightcurve.

        # Evaluating the sourcespline at the real jds :
        sourcerealjds = self.spline.datapoints.jds[self.spline.datapoints.mask] + jdoffset
        sourcerealmags = sourcespline.eval(sourcerealjds)

        # Getting the real mags :
        realpointmags = lightcurve.getmags(noml=True)  # Note that we do not want to include the ML here of course !
        realtargetmags = sourcerealmags - realpointmags

        # And now we need to interpolate the values for the stab points.
        # Is it better to interpolate the real mags, or directly the correction (the targetmags) ?
        # Perhaps better to interpolate the correction, as we do not want the ML to follow the intrinsic spline
        # anyway. We assume the ML is a simple as possible.

        sourcestabjds = self.spline.datapoints.jds[self.spline.datapoints.mask is False] + jdoffset

        stabtargetmags = np.interp(sourcestabjds, sourcerealjds, realtargetmags, left=100.0, right=100.0)
        # left and right should never be needed ...

        # The update itself :
        self.spline.datapoints.mags[self.spline.datapoints.mask] = realtargetmags
        self.spline.datapoints.mags[self.spline.datapoints.mask is False] = stabtargetmags

        # The magerrs of the stabpoints do not change, we update just the real magerrs, in case.
        self.spline.datapoints.magerrs[self.spline.datapoints.mask] = lightcurve.magerrs

    def replacespline(self, newspline):
        """
        If you want to change the spline of this SplineML object
        I should maybe check that you didn't change stab attribute stuff here ?
        """
        olddp = self.spline.datapoints
        self.spline = newspline.copy()
        # And now we want to update the splines datapoints, by principle. Even if typically you will not fit this spline
        # anymore, we need at least to be able to evaluate it !
        self.spline.updatedp(olddp, dpmethod="extadj")

    def reset(self):
        """
        Puts all coeffs back to zero, and redistributes the knots uniformly.
        The number of knots does not change of course !
        """
        self.spline.reset()

    def calcmlmags(self, lightcurve):
        """
        Required by lc (for getmags, applyml, etc...)
        Returns a mags-like vector containing the mags to be added to the lightcurve.
        :param lightcurve: lightCurve object, not used, this is to match the polyml.Microlensing.calcmlmags()
        """

        return self.spline.eval(nostab=True)  # Of course, we do not want to evaluate the stab points !

    def smooth(self, lightcurve, n=1000):
        """
        Returns a dict of points to plot when displaying this microlensing.
        Just for plots.

        Warning : we suppose here that lightcurve is the same as was used to build the ml !

        Here we directly return stuff that can be plotted, so we *do* care about timeshifts.

        n is the number of points you want. the more the smoother.
        """
        smoothtime = np.linspace(self.spline.datapoints.jds[0], self.spline.datapoints.jds[-1], int(n))
        smoothml = self.spline.eval(jds=smoothtime)
        refmag = np.median(lightcurve.getmags())

        # We also return the knots, in the form of points upon the smooth curve
        # These can then be plotted with errorbars, for instance.

        knotjds = self.spline.getinttex()
        knotmags = self.spline.eval(jds=knotjds)

        # Setting the correct offset, so that the microlensing is shown along with the lightcurve :
        jdref = lightcurve.getjds()[0]
        smoothtime += jdref
        knotjds += jdref

        return {"n": n, "jds": smoothtime, "ml": smoothml, "refmag": refmag, "knotjds": knotjds, "knotmags": knotmags}

def addtolc(lc, n=5, knotstep=None, stab=True, stabgap=30.0, stabstep=3.0, stabmagerr=1.0,
            bokeps=10.0, boktests=10, bokwindow=None):
    """
    Adds a SplineML to the lightcurve.
    SplineML splines have NO external stabilization points (stabext = 0.0) !
    We ignore any mask of the lc, cut it before putting this ML.
    This is just about putting things in place. Of course we cannot optimize anything here !

    If targetlc is not None, then pass me another light curve that already have a microlensing spline. I will "copy" that microlensing to your lc and adjust the knots coefficients accordingly.

    The stab stuff inserts stab points into gaps only.


    We use uniknots, n is the number of uniform intervals you want.
    Or specify knotstep : if this is specified, I don't use n.

    ::

        pycs.gen.splml.addtolc(l, knotstep=200)

    """
    lcjds = lc.getjds()
    jdoffset = lcjds[0]  # We can do this, as jds is sorted.
    lcjds -= jdoffset  # so the first true datapoint of a ML spline is at 0.0

    dpmags = np.zeros(len(lcjds))
    dpmagerrs = np.ones(len(lcjds))

    dp = DataPoints(lcjds, dpmags, dpmagerrs, splitup=True, sort=True,
                    stab=stab, stabext=0.0, stabgap=stabgap, stabstep=stabstep, stabmagerr=stabmagerr)

    s = Spline(dp, bokeps=bokeps, boktests=boktests, bokwindow=bokwindow)

    if knotstep is None:
        s.uniknots(n, n=True)
    else:
        s.uniknots(knotstep, n=False)

    # And we add the SplineML to our curve :
    lc.addml(SplineML(s))

