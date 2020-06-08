"""
Module containing all we need to manipulate lightcurves. Could ideally be used by all our curve-shifting algorithms.
"""
import copy as pythoncopy
from functools import reduce

import numpy as np
from pycs3.gen.util import readidlist


class LightCurve:
    """
    A class to handle light curves of lensed QSO images.

    The actual information is stored into independent "vectors" (1D numpy arrays) : one for time, one for magnitude...

    An exception to this is the properties attribute : a list of dicts, one for each data point,
    usually all containing the same fields.

    Note that a lightcurve MUST always be chronologically ordered, this allows faster algorithms.
    See methods sort() and validate().
    Importation functions should automatically sort().

    .. note:: There is only one remaining philosophy about handling shifts of lightcurves:  use shifttime()
    and cie, this is fast and you do not change the actual data (recommended)
    """

    def __init__(self, telescopename="Telescope", object="Object", plotcolour="crimson"):
        """
        The plain constructor, binding all the instance variables.
        It gives a small default lightcurve with 5 points, labels, and a mask to play with.

        :param telescopename: string, name of the telescope that took this curve
        :param object: string, name of the astronomical object observed
        :param plotcolour: string, recognized matplotlib colour. Used when plotting the light curve
        """
        # Some various simple attributes :
        self.telescopename = telescopename
        self.object = object
        self.plotcolour = plotcolour

        # The data itself : here we give some default values for testing purposes :
        self.jds = np.array([1.1, 2.0, 3.2, 4.1, 4.9])  #
        """ 
        The "dates", as floats, in HJD, MHJD, ... these jds should always be in 
        chronological order please (and of course always correspond to the other arrays)."""

        self.mags = np.array([-10.0, -10.1, -10.2, -10.3, -10.2])
        self.magerrs = np.array([0.1, 0.2, 0.1, 0.1, 0.1])

        # Other attributes:
        self.mask = np.array([True, True, False, True, True])
        """
        A boolean mask, allows to "switch off" some points.
        True means "included", False means "skip this one".
        """

        self.properties = [{"fwhm": 1.0}, {"fwhm": 1.0}, {"fwhm": 1.0}, {"fwhm": 1.0}, {"fwhm": 1.0}]
        """
        This is a list of dicts (exactly one per data point) in which you can put whatever you want to.
        You can use them to carry image names, airmasses, sky levels, telescope names etc.
        Of course this slows down a bit things like merging etc.
        """

        # Other useful settings for plots
        self.ploterrorbars = True
        """A flag to show or hide errorbars in plots of this lightcurve."""

        # Values of the two obvious possible shifts.
        self.timeshift = 0.0
        """
        A float giving the shift in time (same unit as jds) that was applied to the lightcurve. 
        This is updated at every shift, so that a lightcurve is always "aware" of its eventual shift. 
        """

        self.magshift = 0.0
        """A float giving the shift in magnitude (similar to timeshift)."""

        self.fluxshift = 0.0
        """
        A float giving the shift in flux (!). As we work with magnitudes, a shift in flux is a bit special.
        Note by the way that a flux shift does not commute with a magnitude shift !
        """

        # And now the microlensing : by default there is none :
        self.ml = None
        """Optional, microlensing curve associated to the current light curve"""

        self.commentlist = []
        """Save all operation applied to the light curve in this list"""

        self.labels = [""] * len(self)
        """String if you want to associate a label to each data point"""

        self.showlabels = False
        """boolean to plot a label for each datapoint"""

    # I explicitly define how str(mylightcurve) will look. This allows a nice "print(mylightcurve)" for instance !
    def __str__(self):
        """
        We explicitly define how str(mylightcurve) will look. "[telescopename/objet]", or "[telescopename/object](timeshift,magshift)"
        in case the lightcurve is shifted.

        Microlensing "constrains" are shown like |12|, meaning in this case
        two seasons : first one with 1 parameter, second one with two.
        """
        retstr = "[%s/%s]" % (self.telescopename, self.object)

        if self.timeshift != 0.0 or self.magshift != 0.0 or self.fluxshift != 0.0:
            retstr += "(%.3f,%.3f,%.0f)" % (self.timeshift, self.magshift, self.fluxshift)

        if self.ml is not None:
            retstr += "%s" % self.ml

        return retstr

    def __len__(self):
        """
        We define what len(mylightcurve) should return : the number of points.
        I do not exclude masked points -- this is the full length.
        """
        return len(self.jds)

    def samplingstats(self, seasongap=60.0):
        """
        Calculates some statistics about the temporal sampling of the lightcurve object.
        I do not take into account the mask !

        :param seasongap: Minimal interval in days to consider a gap to be a season gap.

        I return a dict with the following keys :

        * nseas : number of seasons
        * meansg : mean length (in days, of course) of the season gaps
        * minsg : min season gap
        * maxsg : max season gap
        * stdsg : standard deviation of the season gaps
        * med : the median sampling interval inside seasons
        * mean : the mean sampling interval inside seasons
        * min : ...
        * max
        * std

        """

        self.validate()

        stats = {"len": len(self)}

        first = self.jds[:-1]
        second = self.jds[1:]
        gaps = second - first

        seasongaps = gaps[gaps >= seasongap]
        intervals = gaps[gaps < seasongap]  # The "normal" gaps

        if len(seasongaps) == 0:
            seasongaps = [0]
            stats["nseas"] = 1  # Number of seasons
        else:
            stats["nseas"] = int(len(seasongaps) + 1)  # Number of seasons
        stats["meansg"] = np.mean(seasongaps)
        stats["minsg"] = np.min(seasongaps)
        stats["maxsg"] = np.max(seasongaps)
        stats["stdsg"] = np.std(seasongaps)

        stats["med"] = np.median(intervals)
        stats["mean"] = np.mean(intervals)
        stats["max"] = np.max(intervals)
        stats["min"] = np.min(intervals)
        stats["std"] = np.std(intervals)

        return stats

    def commonproperties(self, notonlycommon=False):
        """
        Returns a list of strings containing the property field names that are common to
        all data points.
        :param notonlycommon : boolean. If True, then I return all fields, not only the ones common to
        all my data points.
        """
        propkeys = [set(props.keys()) for props in self.properties]
        if notonlycommon:
            return sorted(list(reduce(lambda x, y: x.union(y), propkeys)))
        else:
            return sorted(list(reduce(lambda x, y: x.intersection(y), propkeys)))

    def longinfo(self, title="None"):  # note that the name info() is reserved for python
        """
        Returns a multi-line string (small paragraph) with information
        about the current state of a lightcurve.
        """

        if title != "None":
            titlestr = "\n\t%s" % (title.upper())
        else:
            titlestr = ""

        samplingstats = self.samplingstats()
        string_list = ["- " * 30, titlestr, "\n\t", str(self),
                       "\n", "%i points (total), %i of which are masked" % (len(self), self.mask.tolist().count(False)),
                       "\n", "%i seasons (gap: >60), gap length : %.1f +/- %.1f days" % (
                           samplingstats["nseas"], samplingstats["meansg"], samplingstats["stdsg"]),
                       "\n", "Sampling : median %.1f, mean %.1f, max %.1f, min %.2f days" % (
                           samplingstats["med"], samplingstats["mean"], samplingstats["max"], samplingstats["min"]),

                       "\nShifts : (%.5f,%.5f,%.2f) [days, mag, flux]" % (
                           self.timeshift, self.magshift, self.fluxshift),
                       "\nColour : ", self.plotcolour,
                       "\nCommon properties : ", ", ".join(self.commonproperties()),
                       "\n   All properties : ", ", ".join(self.commonproperties(notonlycommon=True)),
                       "\nComments :"]
        for comment in self.commentlist:
            string_list.append("\n   %s" % comment)
        if self.ml is not None:
            mlstr = "\nMicrolensing : %s" % (str(self.ml))
            string_list.append(mlstr)
        string_list.extend(["\n", "- " * 30])
        return ''.join(string_list)

    def printinfo(self, title="None"):
        """ Prints the above longinfo about the lightcurve."""
        print(self.longinfo(title))

    # Elementary lightcurve methods related to shifts
    def shifttime(self, days):
        """
        Update the value of timeshift, relative to the "current" shift.
        The jds array is not shifted; we just keep track of a timeshift float. So this is fast.
        """
        self.timeshift += float(days)

    def shiftmag(self, deltamag):
        """
        Shifts the curve in mag, with respect to the "current" shift. Same remark as for shifttime().
        """
        self.magshift += float(deltamag)

    def shiftflux(self, flux):
        """
        Idem, but a shift in flux.
        """
        self.fluxshift += float(flux)

    def setfluxshift(self, flux, consmag=False):
        """
        An absolute setter for the fluxshift, that directly tries to correct
        for the equivalent magnitude shift with respect to the mean magnitude of the curve.
        So be careful : I change magshift as well !
        """
        if consmag:

            oldfsmags = self.calcfluxshiftmags()
            self.fluxshift = float(flux)
            newfsmags = self.calcfluxshiftmags()
            magcor = - np.mean(newfsmags - oldfsmags)
            self.shiftmag(magcor)  # to compensate for the fluxshift
        else:
            self.fluxshift = float(flux)

    def getjds(self):
        """
        A getter method returning the jds + timeshift. This one does the actual shift.
        Because of the addition, we return a copy.
        """
        return self.jds + self.timeshift

    def getminfluxshift(self):
        """
        Returns the minimal flux shift (i.e. largest negative fluxshift) so that the flux of all points in this curve
        remain positive.
        -> returns - the minimum flux of the curve.
        This does not depend on timeshifts, magshifts, or microlensing.
        When optimizing fluxshifts, you should check by yourself that you respect this minimum value.
        """
        return - np.min(self.getrawfluxes())

    def getrawfluxes(self):
        """
        Returns the raw fluxes corresponding to self.mags
        """

        return 10.0 ** (self.mags / -2.5)

    def calcfluxshiftmags(self, inverse=False):
        """
        Returns an array of mags that you have to add to the lc.mags if you want to take into account for the fluxshift.
        Be careful with the order of things ! Always take into account this fluxshift *before* the magshift / microlensing.

        Inverse = True is a special feature, it returns the mags to subtract from fluxshiftedmags to get back the original mags.
        Perfectly clear, no ? Just changes signs in the fomula.
        You could also use - calcfluxshiftmags(self, inverse=True) and reverse the sign of the fluxshift, but this is more convenient.
        """
        if self.fluxshift != 0.0:
            # We need to include a check :
            if inverse:
                shifts = 2.5 * np.log10((-self.fluxshift * np.ones(len(self)) / (10.0 ** (self.mags / -2.5))) + 1.0)
            else:
                shifts = -2.5 * np.log10((self.fluxshift * np.ones(len(self)) / (10.0 ** (self.mags / -2.5))) + 1.0)
            if np.all(np.isnan(shifts) is False) is False: # pragma: no cover  # If there is a nan in this...
                print("Ouch, negative flux !")
                return np.zeros(len(self))
            else:
                return shifts

        else:
            return np.zeros(len(self))

    def addfluxes(self, fluxes):
        """
        Give me fluxes as a numpy array as long as self.mags, and I'll add those to self.mags.
        I work directly on self.mags, thus I do not take into account magshifts or ml.
        As this changes the actual self.mags, it cannot be undone !
        """
        # We check that we won't go into negative fluxes :
        if not np.all(fluxes > self.getminfluxshift()): # pragma: no cover
            raise RuntimeError("That would give negative fluxes ...")

        newfluxes = self.getrawfluxes() + fluxes
        self.mags = -2.5 * np.log10(newfluxes)
        self.commentlist.append("Added some fluxes")

    def getmags(self, noml=False):
        """
        A getter method returning magnitudes. Now this is non trivial, as magnitudes are influenced by :
            - fluxshift
            - magshift
            - microlensing (= like a variable magshift)

        We "add" this stuff in this order, i.e. fluxshift always comes first.
        We always return a copy of the array, to prevent you from changing the actual lightcurve.


        lc.mags "+" fluxshift + magshift + microlensings if ml is present,
        and just lc.mags "+" fluxshift + magshift if not.
        You can overwrite this : if noml = True, I don't include the ml even if a microlensing is present !
        """
        if self.fluxshift != 0.0:
            if (self.ml is not None) and (noml is False):
                return self.mags + self.calcfluxshiftmags() + self.magshift + self.ml.calcmlmags(self)
            else:
                return self.mags + self.calcfluxshiftmags() + self.magshift
        else:
            if (self.ml is not None) and (noml is False):
                return self.mags + self.magshift + self.ml.calcmlmags(self)
            else:
                return self.mags + self.magshift

    def getmagerrs(self):
        """
        Return the magnitude errors
        :return: array of mag errors
        """
        return self.magerrs.copy()

        # Now, stuff for microlensing and manipulating the magnitudes :

    def addml(self, microlensing, verbose=False):
        """
        Adds a microlensing object (by add we mean it replaces one if it was already present) -- we are not just setting
        the parameters here ! Note that we COPY the microlensing object, so that the one added is independent from yours.
        """

        if self.ml is not None and verbose: # pragma: no cover
            print("I replace an existing microlensing.")

        self.ml = microlensing.copy()  # this copy is important if you append the "same" new ml object to different lcs.

        self.ml.checkcompatibility(self)
        self.commentlist.append('Setup of microlensing "%s"' % self.ml)

    def rmml(self):
        """Simply deletes the microlensing object"""
        if self.ml is not None:
            self.commentlist.append('Microlensing "%s" removed.' % self.ml)
        self.ml = None

    def resetml(self):
        """
        Resets the microlensing to "zero".
        * for polynomial ML the coefficients are put to zero
        * for spline ML we reset the coeffs to zero
        """
        if self.ml is not None:
            self.ml.reset()

    def resetshifts(self, keeptimeshift=False):
        """
        Removes all shifts and microlensing, getting back to the observed data.
        Does keep the mask.
        """

        self.rmml()
        self.fluxshift = 0.0
        self.magshift = 0.0

        if not keeptimeshift:
            self.timeshift = 0.0

    def applyfluxshift(self):
        """
        It adds the fluxshift-float to the present flux, then puts this fluxshift-float to 0. So that "nothing" changes as seen from
        the outside.
        This is used for instance when you want to merge different lightcurves with different fluxshifts.
        Needless to say, use this carefully, only if it really makes sense for what you want to do.
        Note that we do not touch microlensing here, it remains in place and does not change its meaning in any way.
        """

        self.mags += self.calcfluxshiftmags()
        self.commentlist.append("CAUTION : fluxshift of %f APPLIED" % self.fluxshift)
        self.fluxshift = 0.0  # as this is now applied.

    def applymagshift(self):
        """
        It adds the magshift-float to the present mags, then puts this magshift-float to 0. So that "nothing" changes as seen from
        the outside.
        This is used for instance when you want to merge different lightcurves with different magshifts.
        Needless to say, use this carefully, only if it really makes sense for what you want to do.

        Note that we do not touch microlensing here, it remains in place and does not change its meaning in any way.
        """

        if self.fluxshift != 0.0: # pragma: no cover
            raise RuntimeError("Apply the fluxshift before applying the magshift !")

        self.mags += self.magshift
        self.commentlist.append("CAUTION : magshift of %f APPLIED" % self.magshift)
        self.magshift = 0.0  # as this is now applied.

    def applyml(self):
        """
        We "add" the microlensing to the actual mags, then remove the microlensing object.
        The advantage is that getmags() gets faster, as it does not have to evaluate the microlensing anymore.
        It also allows you to do some other tricks like tweaking a curve, shifting seasons etc if you want.
        So as a conclusion : use this when you know what you are doing.

        We do not touch magshift, it remains perfectly valid afterwards.
        But, if there is a *fluxshift*, we will have to make sure that it was "applied" before !
        """

        if self.ml is None:# pragma: no cover
            raise RuntimeError("Hey, there is no ml associated to this lightcurve !")

        if self.fluxshift != 0.0:# pragma: no cover
            raise RuntimeError("Apply the fluxshift before applying the ML !")
        # This is really important. One possibility would be that this function applies the fluxshift ?

        self.mags += self.ml.calcmlmags(self)
        self.commentlist.append('CAUTION : microlensing %s APPLIED' % self.ml)
        self.rmml()  # very important, we get rid of the present stuff. This prevents you from applying it twice etc.

        # And various other features in random order.

    def setindexlabels(self):
        """First point gets "0", second point gets "1", etc; handy to identify troublemakers on a plot !"""

        self.labels = [str(i) for i in range(len(self.jds))]

    def setjdlabels(self):
        """Points are labeled by their mjd rounded to .1 days. Use these labels to write a skiplist."""

        self.labels = ["%.1f" % jd for jd in self.jds]

    def maskskiplist(self, filepath, searchrange=0.2, accept_multiple_matches=False, verbose=True):
        """
        I mask points according to a skiplist. I do not modify the mask for points that are not on the skiplist,
        i.e. I do not reset the mask in any way.
        The structure of the skiplist is one "point" per line, where the first element in the line
        is the MJD (for instance as identified on a plot using setjdlabels() !).
        For each point from the skiplist, I will search for the corresponding point in the lightcurve
        within searchrange days. I will warn you in case of anything fishy.


        :param filepath: file to read
        :type filepath: string or path
        :param searchrange: range in which I'll look for points (in days).
        :type searchrange: float
        :param accept_multiple_matches: control if many points can be masked at once
        :type accept_multiple_matches: boolean
        :param verbose :verbosity option
        :type verbose: boolean


        """
        skippoints = readidlist(filepath, verbose=verbose)

        if verbose:
            if self.hasmask():
                print("Note : %i epochs are already masked." % (np.sum(self.mask == False)))

        for skippoint in skippoints:
            skipjd = float(skippoint[0])
            indices = np.argwhere(np.fabs(self.jds - skipjd) <= searchrange)
            if len(indices) == 0:
                if verbose:
                    print("Warning, epoch %s from skiplist not found in %s !" % (skippoint[0], str(self)))
            elif len(indices) > 1:
                if verbose:
                    print("Warning, multiple matches for epoch %s from skiplist !" % (skippoint[0]))
                if accept_multiple_matches:
                    if verbose:
                        print("I mask all of them...")
                    for index in indices:
                        if not self.mask[index]:
                            if verbose:
                                print("Epoch %s is already masked." % (skippoint[0]))
                        else:
                            self.mask[index] = False
            elif len(indices) == 1:
                index = indices[0]
                if not self.mask[index]:
                    if verbose:
                        print("Epoch %s is already masked." % (skippoint[0]))
                else:
                    self.mask[index] = False

        if verbose:
            print("Done with maskskiplist, %i epochs are now masked." % (np.sum(self.mask == False)))

    def remove_epochs(self, index):
        """
        Delete epochs in your LightCurve

        :param integer or array of integer containing the position of the epoch to remove
        :return:
        """

        self.jds = np.delete(self.jds, index)
        self.mags = np.delete(self.mags, index)
        self.magerrs = np.delete(self.magerrs, index)
        self.mask = np.delete(self.mask, index)
        self.properties = np.delete(self.properties, index)
        self.labels = np.delete(self.labels, index)
        self.validate()

    def maskinfo(self):
        """
        Returns a description of the masked points and available properties of them.
        Note that the output format can be directly used as a skiplist.
        """

        cps = self.commonproperties()
        lines = []
        maskindices = np.argwhere(self.mask == False)
        for maskindex in maskindices:
            comment = ", ".join(["%s : %s" % (cp, self.properties[maskindex][cp]) for cp in cps])
            txt = "%.1f    %s" % (self.jds[maskindex], comment)
            lines.append(txt)

        txt = "\n".join(lines)
        txt = "# %i Masked points of %s :\n" % (np.sum(self.mask == False), str(self)) + txt
        return txt

    def clearlabels(self):
        """Sets label = "" for all points.
        You could use this before setting the labels of just some of the points "by hand", for instance."""

        self.labels = [""] * len(self)

    def clearproperties(self):
        """Removes all properties"""

        self.properties = [{} for i in range(len(self))]

    def copy(self):
        """
        Returns a "deep copy" of the lightcurve object. Try to avoid this within loops etc ... it is slow !
        Typically if you want to optmize time and mag shifts, think about using local backups of lc.getmags() and lc.getjds() etc ...

        We use the copy module, imported as "pythoncopy" to avoid confusion with this method.

        @rtype: lightcurve
        @return: A copy of the lightcurve.

        """
        return pythoncopy.deepcopy(self)

    def cutmask(self):
        """
        Erases all masked points from the lightcurve.

        This is one way of handling the mask, but perhaps not the best/fastest one,
        depending on what you want do !
        If you write a curve shifting algorithm, probably your code will be more elegant if you teach him to simply skip
        masked points, instead of making a copy and removing them using this method !

        .. warning:: Cutmask will change the meaning of seasons, that's why we are forced to delete any present microlensing. If you want to "keep" it, you'll have to apply it first.
        """

        self.jds = self.jds[self.mask]
        self.mags = self.mags[self.mask]
        self.magerrs = self.magerrs[self.mask]
        self.labels = [self.labels[i] for i in range(len(self.mask)) if self.mask[i]]
        # This is a bit harder, as labels is a plain python list and not a numpy array.
        # Be careful not to use self.jds in the range() at this point ...
        # Same is true for the properties :
        self.properties = [self.properties[i] for i in range(len(self.mask)) if self.mask[i]]
        # finally we change the mask itself :
        self.mask = self.magerrs >= 0.0  # This should be True for all !

        if self.ml is not None:
            print("WARNING : cutmask() just removed your microlensing !")
            self.rmml()

        # Note that removing microlensing is important, as the parameters change
        # their meanings, and seasons might change as well etc.

    def hasmask(self):
        """
        Returns True if some points are masked (i.e., if there is a False), False otherwise.
        """
        return not np.alltrue(self.mask)

    def clearmask(self):
        """
        Sets all the mask to True.
        """
        self.mask = np.ones(len(self), dtype=np.bool)

    def validate(self, verbose=False):
        """
        Checks the "health" and coherency of a lightcurve.
        Are there as many mags as jds, etc ? No return value, but raises RuntimeError in case
        of failure.

        .. note:: Here we also check that the curve is "ordered", i.e. the jds are monotonously increasing.
            At this stage it is OK to have several points with the *same* julian date.
            In fact such points are a problem for splines, but this is addressed once you use splines.

        """

        ndates = len(self.jds)
        if len(self.mags) != ndates or len(self.magerrs) != ndates or len(self.mask) != ndates or len(
                self.labels) != ndates or len(self.properties) != ndates:
            raise RuntimeError("Incoherence in the length of your lightcurve !")

        # I postulate that lightcurves shorter then 2 points make no sense (the next test, and seasons() etc would crash...)
        if ndates < 2:
            raise RuntimeError("Your lightcurve is too short... give me at least 2 points.")

        # some algorithms have been optimized so that they NEED ordered lightcurves, i.e. values in self.jds must be increasing.
        # here we check if this is the case for a given lightcurve.
        first = self.jds[:-1]
        second = self.jds[1:]
        if not np.alltrue(np.less_equal(first, second)):  # Note the less_equal : ok to have points with same JD.
            raise RuntimeError("Your lightcurve is not sorted !")

        # we check if the errors are positive:
        if not np.alltrue(self.magerrs > 0.0):
            raise RuntimeError("Your lightcurve has negative or null errors !")
        # note that we use this sometimes in the code to generate masks etc... so don't just
        # remove this check.

        # the associated microlensing object if present
        if self.ml is not None:
            self.ml.checkcompatibility(self)

        if verbose:
            print("%s : validation done !" % self)

    def sort(self):
        """
        We sort the lightcurve points according to jds.
        Of course all the arrays and lists have to be sorted.

        The meaning of seasons is changed, that's why we have to delete any microlensing.
        """
        # Now look how nice this is (so proud...) :
        sortedindices = np.argsort(self.jds)
        self.jds = self.jds[sortedindices]
        self.mags = self.mags[sortedindices]
        self.magerrs = self.magerrs[sortedindices]
        self.mask = self.mask[sortedindices]
        self.labels = [self.labels[i] for i in sortedindices]  # trick as labels is not a numpy array
        self.properties = [self.properties[i] for i in sortedindices]  # trick as properties is not a numpy array

        if self.ml is not None:
            print("WARNING : sort() just removed your microlensing !")
            self.rmml()

    def montecarlomags(self, f=1.0, seed=None):
        """
        We add gaussian noise to all mags according to their errors.
        We do not care about any microlensing of shifts here, but directly tweak the self.mags.
        The order of the points is not touched.

        I modify all points, even masked ones.

        :param seed: if None, the clock is used, if not, the given seed.
        :type seed: int or None

        :param f: a multiplier

        """

        self.commentlist.append("Monte Carlo on mags !")  # to avoid confusions.
        rs = np.random.RandomState(seed)  # we create a random state object, to control the seed.
        self.mags += rs.standard_normal(self.mags.shape) * f * self.magerrs  # and here we do the actual bootstrapping !

    def montecarlojds(self, amplitude=0.5, seed=None, keepml=True):
        """
        We add a UNIFORM random variable from -amplitude to +amplitude to each jd.
        This is not trivial, as we have to take care about the order.

        We need sorting, so normally the microlensing would be removed.
        But if you know that, using only a small amplitude, the seasons will remain of the same length after the bootstrapping,
        you can use keepml=True (default), and we will "keep" the microlensing. Note that the microlensing functions are defined
        with respect to the jds : so as the jds are bootstrapped, the mircolensing is slightly affected, but this is fine as typically
        you just want to keep a rough estimate of the microlensing, and run an ML optimization anyway !
        Hence this obscure option.

        I modify all points, even masked ones.
        """

        self.commentlist.append("Monte Carlo on jds !")  # to avoid confusions.
        rs = np.random.RandomState(seed)  # we create a random state object, to control the seed.
        self.jds += (rs.uniform(low=0.0, high=2.0 * amplitude, size=self.jds.shape) - amplitude)
        # uniform distribution. Yes, this is a bit strange, but low cannot be negative.

        # And now everything is fine but the curve might not be sorted, so :
        if keepml is False or self.ml is None:
            self.sort()  # bye bye microlensing
        else:
            ml_i_wanna_keep = self.ml.copy()
            self.rmml()
            self.sort()
            self.addml(ml_i_wanna_keep)
        # Isn't this wonderful ?!

    def pseudobootstrap(self):
        """
        We randomly mask some points, but without duplicating them.
        Real bootstap would be to draw N points from a lightcurve of N points with replacements.
        But we do not replace here, instead we do as if we would replace, but then skip any "duplicates".
        I respect mask : masked points will stay masked, I do as if they were not there.
        """

        indices = np.arange(len(self))[self.mask]  # the indices we want to bootstrap : only unmasked ones.

        if indices.size == 1:
            raise RuntimeError("Not enough points to bootstrap !")

        draws = np.random.randint(0, high=indices.size, size=indices.size)  # indexes of the indices that are drawn
        # We remove the duplicates from these draws:
        # uniquedraws = np.array(sorted(list(set(list(draws)))))

        # Faster and easier :
        newmask = np.zeros(len(self), dtype=np.bool)  # an array of False, as long as the full curve
        newmask[indices[draws]] = True  # drawn points are set True
        self.mask = newmask

        self.commentlist.append("Pseudobootstraped !")

    def merge(self, otherlc):
        """
        Merges a lightcurve into the current one. Warning : it takes the other lightcurve as it comes, with all shifts
        or eventual microlensings -- "applied" or just specified !
        In fact it behaves as if every shift or microlensing was applied before the merging.


        It's up to you to check that you don't merge lightcurves with timeshifts, which would probably be nonsense.
        We delete an eventual current microlensing object -> You will have to define new seasons and parameters if you want.

        :param otherlc: A lightcurve to merge into the current one.
        :type otherlc: lightcurve

        The origin of each point of the otherlc is appended to the labels.
        Settings and colour etc from the current curve are not changed.
        After the merging process, the lightcurve is sorted and validated.

        """
        # Let's warn the user if there are timeshifts in the curves :
        if self.timeshift != 0.0 or otherlc.timeshift != 0.0:
            print("WARNING : you ask me to merge time-shifted lightcurves !")

        # and microlensing :
        if self.ml is not None or otherlc.ml is not None:
            print("WARNING : I am merging lightcurves with possible microlensing !")

        # for the magnitude shift, for otherlc it is quite common, but not for self :
        if self.magshift != 0.0 or self.fluxshift != 0.0:
            print("WARNING : merging into a lightcurve with magshift or fluxshift : everything gets applied ! ")

        # Frist we just concatenate all the values into new numpy arrays :

        concjds = np.concatenate([self.getjds(), otherlc.getjds()])
        concmags = np.concatenate([self.getmags(), otherlc.getmags()])
        # To calculate the ML, we need the untouched input "self" curve.
        # Hence this new variable (and not directly using self.mags etc) !
        concmagerrs = np.concatenate([self.magerrs, otherlc.magerrs])
        concmask = np.concatenate([self.mask, otherlc.mask])

        self.jds = concjds
        self.mags = concmags
        self.magerrs = concmagerrs
        self.mask = concmask
        self.resetshifts()  # As we used getjds() and getmags() above ...

        # We change the new labels so that they tell us from which lightcurve the point was from.
        newlabels = [label + "(from %s)" % str(otherlc) for label in otherlc.labels]  # This way we make a copy
        self.labels.extend(newlabels)

        # And now the properties :
        self.properties.extend(otherlc.properties)

        self.commentlist.append("Merged with %s" % str(otherlc))
        # self.commentlist.extend(otherlc.commentlist)
        self.telescopename = self.telescopename + "+%s" % otherlc.telescopename

        # The very essential :
        self.sort()

        # Just to be sure that everything went fine :
        self.validate()

    def rdbexport(self, filename=None, separator="\t", writeheader=True, rdbunderline=True, properties=None):
        """
        Writes the lightcurve into an "rdb" file, that is tab-separeted columns and a header.
        Note that any shifts/ML are taken into account. So it's a bit like if you would apply the
        shifts before writing the file.

        Includes mask column only if required (if there is a mask)


        :param filename: where to write the file
        :type filename: string or path
        :param separator: how to separate the collumns
        :type separator: string
        :param writeheader: include rdb header ?
        :type writeheader: boolean
        :param properties: properties of the lightcurves to be include in the file.
        :type properties: list of strings, e.g. ["fwhm", "ellipticity"]
        :param rdbunderline : If you want to have a separation between column title and the data
        :type rdbunderline : boolean

        """
        import csv

        self.validate()  # Good idea to keep this here, as the code below is so ugly ...

        if filename is None:
            filename = "%s_%s.rdb" % (self.telescopename, self.object)

        # We include a "mask" column only if mask is not True for all points
        if False in self.mask:
            colnames = ["mhjd", "mag", "magerr", "mask"]
            data = [self.getjds(), self.getmags(), self.magerrs, self.mask]

        else:
            colnames = ["mhjd", "mag", "magerr"]
            data = [self.getjds(), self.getmags(), self.magerrs]

        # Now we do some special formatting for the cols mhjd, mag, and magerr
        data[0] = ["%.8f" % mhjd for mhjd in data[0]]  # formatting of mhjd
        data[1] = ["%.8f" % mhjd for mhjd in data[1]]  # formatting of mhjd
        data[2] = ["%.8f" % mhjd for mhjd in data[2]]  # formatting of mhjd

        data = list(map(list, list(zip(*data))))  # list to make it mutable

        # We add further columns
        if properties is None:
            properties = []
        colnames.extend(properties)
        for i in range(len(self.jds)):
            for property in properties:
                data[i].append(self.properties[i][property])

        underline = ["=" * n for n in map(len, colnames)]

        outfile = open(filename, "w")
        writer = csv.writer(outfile, delimiter=separator)

        if writeheader:
            writer.writerow(colnames)
            if rdbunderline:
                writer.writerow(underline)
        writer.writerows(data)

        outfile.close()
        print("Wrote %s into %s." % (str(self), filename))
