"""
One season -- one microlensing
This potentially allows to optimize one season after the other, thus reducing the number of simultaneous parameters,
but of course you can still choose to optimize evrything at once (when using dispersion).
When using a spline fit, optimizing a polynom corresponds to weighted to linear least squares -- fast !
"""

import copy as pythoncopy

import numpy as np
from pycs3.gen.sea import autofactory
import logging
logger = logging.getLogger(__name__)


def polyfit(jds, mags, magerrs, nparams):
    """
    I define my own polyfit, as numpy polyfit does not accept weights until numpy version 1.5
    This is linear least squares -- can only be used when optimizing polyml so to fit a "model" like a spline.

    What we want to do :
    fitp = np.polyfit(jds, mags, deg = nparams-1)

    :param jds: x values
    :param mags: y values
    :param magerrs: y errors
    :param nparams: number of parameters of polynom to fit.


    Code to test that the "heteroskedasticity correction" works :

    ::

        import matplotlib.pyplot as plt

        x = np.linspace(0, 10, 20)
        ystd = 1.0 * np.ones(len(x))
        yerrs = ystd * np.random.randn(len(x))
        yerrs[-3] = 40.0
        ystd[-3] = 40.0
        print yerrs

        y = 5.0 + 2.0*x - 1.0*x*x + yerrs

        finex = np.linspace(0, 10, 100)

        plt.errorbar(x, y, ystd)

        polyp = polyfit(x, y, ystd, 3)
        #polyp = np.polyfit(x, y, 3)

        finey = np.polyval(polyp, finex)

        plt.plot(finex, finey)
        plt.show()

    """

    pows = np.arange(nparams)

    # We do "generalized least squares" aka "heteroskedasticity correction"
    a = np.column_stack([(jds ** power) / magerrs for power in pows[::-1]])
    fitp = np.linalg.lstsq(a, mags / magerrs, rcond=None)[0]

    # Not using weights, this would be :
    # a = np.column_stack([(jds ** pow) for pow in pows[::-1]])
    # fitp = np.linalg.lstsq(a, mags)[0]

    # Those fitp are directly the polynom coeffs as returned by np.polyfit
    return fitp


class SeasonFct:
    """
    A SeasonFct object is one season + a certain polynom (simple poly or legendre for now -- from outside you do not see what it is) + some stuff
    to play with that.
    By itself, it makes no sense. It needs to be linked to a lightcurve : this is done by adding it to lc.ml, which is a
    microlensing object, i.e. essentially a list of seasonfct objects.

    The idea is that the number of params and the mask remain fixed for one given seasonfct object.
    This makes the code simpler and faster.
    So once the instance is made, only the values of the parameters can be updated.
    If you want to change something else, simply make a new object.

    One idea of the relatively simple get and set methods is that one day we might want to use something else
    than polynomials, and at that time all this might be handy.
    """

    def __init__(self, season, params=np.array([0.0]), mask=np.array([True])):
        """
        Also see the factory functions below ... they will be much easier to use for your task.
        Default values shown above : 1 parameter, i.e. a simple additive constant for the given season, which is free to be optimized.
        :param season: Season object
        :param params: A 1D numpy array of params (floats)
        :param mask : boolean array, True if the parameter is "free".
        """

        # Some basic checks :
        if len(params) != len(mask): # pragma: no cover
            raise RuntimeError("I want params and mask to be of the same length !")

        self.season = season
        self.mltype = "poly"

        self.params = np.asarray(params)

        self.mask = np.asarray(mask)

        self.nfree = np.bincount(self.mask)[-1]
        """@type: int
        @ivar: How many free parameters
        """

    def copy(self):
        """
        Return a copy of itself

        """
        return pythoncopy.deepcopy(self)

    def __str__(self):
        """
        This is just the number of params of the polynom (i.e. degree + 1)
        (Not the number of free params !)
        """
        return "%i" % (len(self.params))

    def longinfo(self):
        """
        Returns a longer description of the object.
        """
        if self.mltype == "poly":
            strlist = ["Plain poly with %s params on %s :" % (str(self), str(self.season))]
            for (i, p, m) in zip(list(range(len(self.params) - 1, -1, -1)), self.params, self.mask):
                if m:
                    paramstr = "\n deg %2i : %g (free)" % (i, p)
                else:
                    paramstr = "\n deg %2i : %g" % (i, p)
                strlist.append(paramstr)
            return "".join(strlist)

    def printinfo(self):
        """
        Prints the longer description of the object.
        """
        logger.info(self.longinfo())

    def setparams(self, p):
        """
        Remember that you are not allowed to change the number of parameters !
        """
        if len(p) == len(self.params):
            self.params = p  # no further testing ...
        else:# pragma: no cover
            raise RuntimeError("Wrong number of parameters !")

    def getparams(self):
        """
        Return all params

        """
        return self.params  # perhaps put a copy here ? Let's see how we use this.

    # it would be nice to return pointers one day...

    def setfreeparams(self, p):
        """
        Here we should really do some testing ... be careful when using this !!!
        """
        if len(p) == self.nfree:
            self.params[self.mask] = p  # no further testing ...
        else:# pragma: no cover
            raise RuntimeError("Wrong number of free parameters !")

    def getfreeparams(self):
        """
        Return the free parameters (those that are not masked)

        """
        return self.params[self.mask]  # put a copy here ?

    def validate(self):
        """
        Do nothing for the moment
        @todo: code this in case of ideas...
        """
        pass

    def checkcompatibility(self, lightcurve):
        """
        Check Season compatibility

        """
        self.season.checkcompatibility(lightcurve)

    def calcmlmags(self, lightcurve):
        """
        Returns a "lc.mags"-like array made using the ml-parameters.
        It has the same size as lc.mags, and contains the microlensing to be added to them.
        The lightcurve object is not changed !

        For normal use, call getmags() from the lightcurve.

        Idea : think about only returning the seasons mags to speed it up ? Not sure if reasonable, as no seasons defined outside ?
        """
        jds = lightcurve.jds[
            self.season.indices]  # Is this already a copy ? It seems so. So no need for an explicit copy().
        # We do not need to apply shifts (i.e. getjds()), as anyway we "center" the jds.

        # Polynomials :
        if self.mltype == "poly":
            refjd = np.mean(jds)
            jds -= refjd  # This is apparently safe, it does not shifts the lightcurves jds.

            allmags = np.zeros(len(lightcurve.jds))
            allmags[self.season.indices] = np.polyval(self.params, jds)  # probably faster then +=
            return allmags

    def smooth(self, lightcurve):
        """
        Only for plotting purposes : returns jds, mlmagshifts, and refmags with a tight and regular sampling,
        over the range given by the season.
        Note that this time we are interested in the acutal shifted jds, so that it looks right when plotted !
        We return arrays that can directly be plotted to illustrate the microlensing.
        """

        jds = lightcurve.getjds()[self.season.indices]

        # Old method :
        if self.mltype == "poly":
            refjd = np.mean(jds)
            jds -= refjd

            refmag = np.median(
                lightcurve.getmags())  # So the reference magnitude is evaluated from the entire lightcurve.
            # Independent on seasons.

            smoothtime = np.linspace(jds[0], jds[-1], 50)
            smoothml = np.polyval(self.params, smoothtime)
            smoothtime += refjd  # important, to get the time back at the right place.
            refmags = np.zeros(50) + refmag
            return {"jds": smoothtime, "ml": smoothml, "refmags": refmags}


class Microlensing:
    """
    Contains a list of seasonfct objects OF ONE SAME LIGHTCURVE OBJECT, and some methods to treat them.

    You probably do not want your seasonfct seasons to overlap. But no problem if they do.

    Again : do not change the contents of such a microlensing object otherwise then retrieving and updating the params
    through the provided methods !
    Do not change the list of seasonfct objects in any other way !
    1) build the seasonfct objects
    2) put them into a microlensing object
    3) do not touch the seasonfct objects anymore if you do not know what you are doing.


    Instead, make a brand new object if you need, using factory functions provided below or to be written.

    """

    def __init__(self, mllist):
        """

        :param mllist: the list of seasonfct objects to be applied.
        :type mllist: list
        :param nfree: the number of free parameters.
        :type nfree: float
        :param mltype: type of microlensing. I take the first Microlensing object of mllist
        :type mltype: str
        """
        self.mllist = mllist
        self.nfree = np.sum(np.array([sfct.nfree for sfct in self.mllist]))
        self.mltype = self.mllist[0].mltype

    def copy(self):
        """
        Return a copy of the Microlensing object

        :return: A copy of itself
        """
        return pythoncopy.deepcopy(self)

    def __str__(self):
        return "".join(["|%s/" % self.mltype] + ["%s" % m for m in self.mllist] + ["|"])

    def longinfo(self):
        """
        Return info about the object in the mllist

        :return: string containing the info

        """
        return "\n".join(["%s" % (m.longinfo()) for m in self.mllist])

    def printinfo(self):
        """
        Print info about the object in the mllist

        """
        logger.info(self.longinfo())

    def getfreeparams(self):
        """
        Straightforward.
        """
        return np.concatenate([sfct.getfreeparams() for sfct in self.mllist])

    # I tested this, the concatenate makes a copy, otherwise
    # we would be mutable, that would be nice to solve these unelegant setfreeparams problem.

    def setfreeparams(self, p):
        """
        This method distributes the params on the microlensing objects as fast as I could ... it's a bit delicate and unelegant.
        As we want to be fast -- do not mess with this and give a p with exactly the right size.
        """

        if len(p) == self.nfree:
            startindex = 0
            for sfct in self.mllist:
                stopindex = startindex + sfct.nfree
                if len(p) == 1:
                    sfct.setfreeparams(p)
                else:
                    sfct.setfreeparams(p[startindex: stopindex])
                startindex += sfct.nfree
        else: # pragma: no cover
            raise RuntimeError("Wrong number of free parameters !")

    def reset(self):
        """
        Puts all coefs back to 0.0
        Allows to start from scratch without having to rebuild a new ML.
        """
        self.setfreeparams(0.0 * self.getfreeparams())  # A bit sad, I agree :)

    def checkcompatibility(self, lightcurve):
        for sfct in self.mllist:
            sfct.checkcompatibility(lightcurve)

    def calcmlmags(self, lightcurve):
        """
        Returns one a "lc.mags"-like array made using the parameters of all seasonfct objects.
        This array has the same size as lc.mags, and contains the microlensing to be added to lc.mags.

        Idea : perhaps we can make this faster by not calling calcmags of the seasonfct objects ?
        """

        allmags = np.zeros(len(lightcurve.jds))
        for microlensing in self.mllist:
            allmags += microlensing.calcmlmags(lightcurve)
        return allmags

    def stats(self, lightcurve):
        """
        Calculates some statistics on the microlensing deformation evaluated at the same sampling
        than the lightcurve.
        The idea is to get the flux ratios, in case of calm microlensing.

        """

        mlmags = self.calcmlmags(lightcurve)

        mlmean = np.mean(mlmags)
        mlstd = np.std(mlmags)

        return {"mean": mlmean, "std": mlstd}


def multigetfreeparams(lclist):
    """
    Give me a list of lightcurves (with or without ml !), I give you a single flat array of parameters of all MLs concatenated.
    Note that the order of lightcurves in lclist is important ! You will have to give the same order for multisetfreeparams() !

    For now a bit simple ... but it's not that slow.
    """

    params = np.array([])  # we start with an empty array
    for curve in lclist:
        if curve.ml is not None:
            params = np.append(params, curve.ml.getfreeparams())

    if len(params) == 0:
        logger.warning("There are no free ml params !")
    return params


def multisetfreeparams(lclist, params):
    """
    Be careful to respect the order of lcs in lclist ... otherwise everything gets messed up.
    Again this seems a bit slower then it really is -- prefer simplicity.
    """

    startindex = 0
    for curve in lclist:
        if curve.ml is not None:
            stopindex = startindex + curve.ml.nfree
            if stopindex > len(params):
                raise RuntimeError("Something is fishy with your params...")

            if len(params) == 1:  # special solution needed if only 1 parameter :-(
                curve.ml.setfreeparams(params)
            else:
                curve.ml.setfreeparams(params[startindex: startindex + curve.ml.nfree])
            startindex += curve.ml.nfree

    if startindex != len(params): # pragma: no cover
        raise RuntimeError("You provided too many params !")


# What we still need is a method that collects and sets all the free params of a list of lightcurves.


def factory(seasons, nparams):
    """
    A factory function to create a microlensings object filled by seasonfct objects.
    seasons is a list of season objects
    nparams is an array or list of ints. "default" = one constant per season.
    All parameters will be set to 0.0, and free to be optimized.

    mltype = "poly" : simple polynomial microlensing, very stupid but fast, ok for degree <= 3
    default type.

    """

    if len(nparams) != len(seasons): # pragma: no cover
        raise RuntimeError("Give as many nparams as they are seasons !")

    mllist = []
    for (season, n) in zip(seasons, nparams):
        if n != 0:
            p = np.zeros(n, dtype="float")
            mask = p > -1.0
            sfct = SeasonFct(season, p, mask)
            mllist.append(sfct)

    return Microlensing(mllist)


def addtolc(l, seasons=None, nparams=1, autoseasonsgap=60.0):
    """
    Adds polynomial ML to the lightcurve l.
    Top level function, to make it really easy ! We just wrap the factory above.

    If seasons = None, I will make some autoseasons. You can specify autoseasonsgap.
    Else, seasons should be a list of seasons (see factory function above)

    If nparams is an int, each season will get a polynom of nparams parameters.
    1 = contant, 2 = slope, ...
    Else nparams should be a list of ints (see factory function above)

    If you want one single polynom for the full curve, just set autoseasonsgap to 10000.0 ...

    ::


        pycs.gen.polyml.addtolc(l, nparams=1) # One constant on each season.



    """

    if seasons is None:
        seasons = autofactory(l, seasongap=autoseasonsgap)
    if type(nparams) == int:
        nparams = [nparams] * len(seasons)

    m = factory(seasons, nparams)

    l.addml(m)
