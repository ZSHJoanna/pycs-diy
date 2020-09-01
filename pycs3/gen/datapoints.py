"""
Module to define Datapoints class which is a minimal version of a LightCurve made for fast computation
"""
import numpy as np
import logging
logger = logging.getLogger(__name__)

class DataPoints:
    """
    An ultralight version of a LightCurve, made for fast computations.
    Can be "merged" from a list of lightcurves, see factory function below.

    A Spline object has such a DataPoints object as attribute.

    ATTENTION
    Datapoints are expected to be ALWAYS SORTED BY JDS, and no two datapoints have the same jd !
    See the splitup option of the constructor.
    Note that this is not the case for lightcurves ! Hence the existence of datapoints.
    Should be enforced in every function that builds datapoints.

    ABOUT STAB POINTS
    With scipy splines, we always get the last knots at the extrema of data points.
    So to get knots "outside" of the real datapoints, we have to insert fake points.
    And while we are at it, these fake points can also be used to stabilize the spline in
    gaps.
    The mask is used to differentiate between actual data points and "stabilization points"
    that are inserted to make the spline behave well at the extrema and in season gaps.
    It is modified by the two addgappts and addextpts.

    The info about stabpoints is written into the object,
    so that they can be reconstructed from any new jds and mags.
    """

    def __init__(self, jds, mags, magerrs, splitup=True, deltat=0.000001, sort=True, stab=False,
                 stabext=300.0, stabgap=30.0, stabstep=5.0, stabmagerr=-2.0, stabrampsize=0, stabrampfact=1.0):
        """
        Constructor
        Always leave splitup and sort on True ! Only if you know that you are already
        sorted you can skip them.
        You cannot specify a mask, I do this myself. (could be done in principle).

        stab : do you want stabilization points ?
        Don't forget to run splitup, sort, and addstab again if you change the data !
        """

        self.jds = jds
        self.mags = mags
        self.magerrs = magerrs

        self.stab = stab
        self.stabext = stabext
        self.stabgap = stabgap
        self.stabstep = stabstep
        self.stabmagerr = stabmagerr

        self.stabrampsize = stabrampsize
        self.stabrampfact = stabrampfact

        self.mask = np.ones(len(self.jds), dtype=np.bool)  # an array of True

        self.deltat = deltat
        if splitup:
            self.splitup()
        elif sort:  # If we do the splitup, we sort anyway.
            self.sort()

        self.putstab()

    def splitup(self):
        """
        We avoid that two points get the same jds by introducing a small random shift of the points.
        Note that this might change the order of the jds,
        but only of very close ones, so one day it would be ok to leave the mags as they are.
        """
        self.jds += self.deltat * np.random.randn(len(self.jds))
        self.sort()

    def sort(self):
        """
        Absolutely mandatory, called in the constructor.
        """
        sortedindices = np.argsort(self.jds)
        self.jds = self.jds[sortedindices]
        self.mags = self.mags[sortedindices]
        self.magerrs = self.magerrs[sortedindices]
        self.mask = self.mask[sortedindices]
        self.validate()

    def validate(self):
        """
        We check that the datapoint jds are increasing strictly :
        """
        first = self.jds[:-1]
        second = self.jds[1:]
        if not np.alltrue(np.less(first, second)):  # Not less_equal ! Strictly increasing !
            raise RuntimeError("These datapoints don't have strictly increasing jds !")

    def rmstab(self):
        """
        Deletes all stabilization points
        """
        self.jds = self.jds[self.mask]
        self.mags = self.mags[self.mask]
        self.magerrs = self.magerrs[self.mask]
        self.mask = np.ones(len(self.jds), dtype=np.bool)

    def putstab(self):
        """
        Runs only if stab is True.
        I will :
        add datapoints (new jds, new mags, new magerrs)
        modify the mask = False for all those new datapoints.
        """
        if self.stab:

            # We start by deleting any previous stab stuff :

            self.rmstab()
            self.addgappts()
            self.addextpts()
        else:
            pass

    def calcstabmagerr(self):
        """
        Computes the mag err of the stabilisation points.
        """
        if self.stabmagerr >= 0.0:
            return self.stabmagerr
        else:
            return - self.stabmagerr * np.median(self.magerrs)

    def addgappts(self):
        """
        We add stabilization points with low weights into the season gaps
        to avoid those big excursions of the splines.
        This is done by a linear interpolation across the gaps.
        """

        absstabmagerr = self.calcstabmagerr()

        gaps = self.jds[1:] - self.jds[:-1]  # has a length of len(self.jds) - 1
        gapindices = np.arange(len(self.jds) - 1)[
            gaps > self.stabgap]  # indices of those gaps that are larger than stabgap

        for n in range(len(gapindices)):
            i = gapindices[n]
            a = self.jds[i]
            b = self.jds[i + 1]

            newgapjds = np.linspace(a, b, int(float(b - a) / float(self.stabstep)))[1:-1]
            newgapindices = i + 1 + np.zeros(len(newgapjds),dtype = int)
            newgapmags = np.interp(newgapjds, [a, b], [self.mags[i], self.mags[i + 1]])
            newgapmagerrs = absstabmagerr * np.ones(newgapmags.shape)
            newgapmask = np.zeros(len(newgapjds), dtype=np.bool)

            self.jds = np.insert(self.jds, newgapindices, newgapjds)
            self.mags = np.insert(self.mags, newgapindices, newgapmags)
            self.magerrs = np.insert(self.magerrs, newgapindices, newgapmagerrs)
            self.mask = np.insert(self.mask, newgapindices, newgapmask)

            gapindices += newgapjds.size  # yes, as we inserted some points the indices change.

        # If you change this structure, be sure to check SplineML.settargetmags as well !

        self.validate()

    def addextpts(self):
        """
        We add stabilization points at both extrema of the lightcurves
        This is done by "repeating" the extremal points, and a ramp in the magerrs
        """

        absstabmagerr = self.calcstabmagerr()

        extjds = np.arange(self.jds[0], self.jds[0] - self.stabext, -1 * self.stabstep)[::-1][:-1]
        extmags = self.mags[0] * np.ones(extjds.shape)
        extmagerrs = absstabmagerr * np.ones(extjds.shape)
        for i in range(1, int(self.stabrampsize) + 1):
            extmagerrs[-i] += (self.stabrampsize + 1 - i) * absstabmagerr * self.stabrampfact
        extindices = np.zeros(extjds.shape, dtype=int)
        mask = np.zeros(len(extjds), dtype=np.bool)
        self.jds = np.insert(self.jds, extindices, extjds)
        self.mags = np.insert(self.mags, extindices, extmags)
        self.magerrs = np.insert(self.magerrs, extindices, extmagerrs)
        self.mask = np.insert(self.mask, extindices, mask)

        # And the same at the other end :

        extjds = np.arange(self.jds[-1], self.jds[-1] + self.stabext, self.stabstep)[1:]
        extmags = self.mags[-1] * np.ones(extjds.shape)
        extmagerrs = absstabmagerr * np.ones(extjds.shape)
        for i in range(0, int(self.stabrampsize)):
            extmagerrs[i] += (self.stabrampsize - i) * absstabmagerr * self.stabrampfact
        extindices = len(self.jds) + np.zeros(extjds.shape, dtype=int)
        mask = np.zeros(len(extjds), dtype=np.bool)
        self.jds = np.insert(self.jds, extindices, extjds)
        self.mags = np.insert(self.mags, extindices, extmags)
        self.magerrs = np.insert(self.magerrs, extindices, extmagerrs)
        self.mask = np.insert(self.mask, extindices, mask)

        self.validate()

    def getmaskbounds(self):
        """
        Returns the upper and lower bounds of the regions containing stabilization points.
        This is used when placing knots, so to put fewer knots in these regions.
        Crazy stuff...
        """

        maskindices = np.where(self.mask == False)[0]

        if len(maskindices) < 3:
            logger.info("Hmm, not much masked here ...")
            return np.array([]), np.array([])
        else:
            lcuts = maskindices[np.where(maskindices[1:] - maskindices[:-1] > 1)[0] + 1]
            lcuts = np.insert(lcuts, 0, maskindices[0])
            ucuts = maskindices[np.where(maskindices[1:] - maskindices[:-1] > 1)[0]]
            ucuts = np.insert(ucuts, len(ucuts), maskindices[-1])
            return lcuts, ucuts

    def ntrue(self):
        """
        Returns the number of real datapoints (skipping stabilization points)
        """
        return np.sum(self.mask)
