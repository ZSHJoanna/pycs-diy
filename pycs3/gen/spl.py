"""
Module defining the Spline class, something easy to wrap around SciPy splines.
Includes BOK algorithms (Mollinari et al.)
Some rules of splrep (k = 3)
	- do not put more then 2 knots between data points.
	- splrep wants inner knots only, do not give extremal knots, even only "once".

"""

import copy as pythoncopy

import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as si
import scipy.optimize as spopt


class Spline:
    """
    A class to represent a spline, that is essentially a set of knots and coefficients.
    As finding knots and coefficients requires access to some data points, these are included
    in the form of a DataPoints object.

    Abount knots :
    Spline.t are all the knots, including extremas with multiplicity.
    But splrep wants internal knots only ! By internal we mean : not even the data extremas !
    Spline.getintt() returns only these internal knots.

    :param t: array of all the knots, not only internal ones !
    :param c: array of coefficients
    :param k: degree of the splines, default = cubic splines k=3, 3 means that you can differentiate twice at the knots.
    :type k: int
    :param bokeps: minimum allowed distance between knots
    :type bokeps: float
    :param boktests: spacing between knots during the test phase
    :type boktests: int
    :param bokwindow: window size to put the knots, by default (None), I'll take the full data lenght.
    :type bokwindow: float
    :param plotcolour: matplotlib color, when plotting the Spline

    """

    def __init__(self, datapoints, t=None, c=None, k=3, bokeps=2.0, boktests=5, bokwindow=None, plotcolour="black"):

        # self.origdatapoints = datapoints
        self.datapoints = datapoints

        # At this point we know that your datapoint jds are monotonously increasing. This is tested
        # by validate() of datapoints.

        self.t = t  # the array of knots
        self.c = c  # the coeffs
        self.k = k

        self.bokeps = bokeps
        self.boktests = boktests
        self.bokwindow = bokwindow

        self.knottype = "none"
        self.plotcolour = plotcolour
        self.showknots = True

        # Bounds, for BOK
        self.lims = None
        self.l = None
        self.u = None

        # We want to keep trace of the r2 of a spline.
        self.lastr2nostab = 0.0  # without stab points (the real thing)
        self.lastr2stab = 0.0  # with stab points (usually not so interesting)

        # If you did not give me a t&c, I'll make some default ones for you :
        try:
            if self.t is None:
                self.uniknots(2)  # This also puts self.c to 0s
        except: # pragma: no cover
            if len(self.t) == 0:
                self.uniknots(2)  # This also puts self.c to 0s

    def __str__(self):
        """
        Returns a string with:
        * degree
        * knot placement
        * number of intervals

        """
        # return "Spline of degree %i, %i knots (%i inner knots), and %i intervals." % (self.k, len(self.t), len(self.getintt()), self.getnint())

        if len(self.knottype) > 6:  # That's a string
            knottext = "%il%ib" % (self.knottype.count("l"), self.knottype.count("b"))
        else:
            knottext = self.knottype
        return "~%i/%s/%i~" % (self.k, knottext, self.getnint())

    def copy(self):
        """
        Returns a "deep copy" of the spline.
        """
        return pythoncopy.deepcopy(self)

    def shifttime(self, timeshift):
        """
        Hard-shifts your spline along the time axis.
        By "hard-shift", I mean that unlike for a lightcurve, the spline will not know that it was shifted !
        It's up to you to be sure that you want to move it.
        We shift both the datapoints and the knots.
        """

        self.t += timeshift
        self.datapoints.jds += timeshift

    def shiftmag(self, magshift):
        """
        Hard-shifts your spline along the mag axis.
        By "hard-shift", I mean that unlike for a lightcurve, the spline will not know that it was shifted !
        It's up to you to be sure that you want to move it.
        We shift both the datapoints and the knots.
        """

        self.c += magshift
        self.datapoints.mags += magshift

    def updatedp(self, newdatapoints, dpmethod="stretch"):
        """

        Replaces the datapoints of the spline, and makes sure that the knots
        stay compatible.

        If you tweaked your datapoints, I will have to tweak my knots to make sure
        that my external knots fit. Hence this method !
        Due to the splitup, this is needed even if you just tweaked the mags !
        And anyway in this case I have to rebuild the stab points.

        .. warning :: it's up to you to check that you don't replave datapoints with different stab settings.
            Anyway it would work, just look ugly !

        Replaces the datapoints (jds, mags, and magerrs) touching the knots and coeffs as less as possible.
        Note that we also have to deal with stab points here !

        This is made for instance for time shifts that only very slightly change the datapoints, and you don't want to
        optimize the knots all the time from scratch again.
        The current knots are "streched" (keeping their relative spacings) accross the new datapoints.

        Options for "dpmethod" :
        - "stretch" : changes all the knots
        - "extadj" : does not touch the internal knots, but adjusts the external ones only, to
        fit the new datapoints. Probably the method to use when optimizing time shifts.
        - "leave" : does not touch the knots -> ok to evaluate the spline,
        but you will not be able to fit it anymore, as the external knots don't correspond to datapoints.

        """

        if dpmethod == "stretch":

            oldmin = self.datapoints.jds[0]  # This includes potential stab points
            oldmax = self.datapoints.jds[-1]

            newmin = newdatapoints.jds[0]  # Idem
            newmax = newdatapoints.jds[-1]

            oldknots = self.getinttex()
            # print oldknots

            # we will stretch the oldknots by a factor a :
            a = (newmax - newmin) / (oldmax - oldmin)

            newknots = newmin + a * (oldknots - oldmin)

            # We set the new datapoints:
            self.datapoints = newdatapoints
            self.setinttex(newknots)

        elif dpmethod == "extadj":

            intknots = self.getintt()
            self.datapoints = newdatapoints

            # Ok, now the newdatapoints might be narrower or wider than the knots, we have to deal with this.
            # If they are wider, it's easy : setint will put move the external knot on the external datapoint.
            # If they are narrower, it's trickier : we have to remove some extra knots, so to really just keep the "internal" ones.
            # to feed into setintt.

            # if True: # works as well, but maybe faster to test first :
            if (self.datapoints.jds[0] >= intknots[0]) or (self.datapoints.jds[-1] <= intknots[-1]):

                keepmask = np.ones(intknots.shape, dtype=np.bool)
                for i in range(len(intknots)):  # Starting from the left ...
                    if intknots[i] <= self.datapoints.jds[0]:
                        keepmask[i] = False
                    else:
                        break
                for i in range(len(intknots))[::-1]:  # And now the right ...
                    if intknots[i] >= self.datapoints.jds[-1]:
                        keepmask[i] = False
                    else:
                        break

                # And finally, we apply the mask .
                intknots = intknots[keepmask]

            self.setintt(intknots)  # This automatically adjusts the extremal knots.

        elif dpmethod == "leave":

            knots = self.getinttex()
            self.datapoints = newdatapoints

            # We quickly check the boundaries
            if (knots[0] >= self.datapoints.jds[0]) or (knots[-1] <= self.datapoints.jds[-1]): # pragma: no cover
                raise RuntimeError("Your newdatapoints are to wide for the current knots !")

        else: # pragma: no cover
            raise RuntimeError("Don't know this updatedp method !")

        # We reset any bounds just to be sure.
        self.lims = None
        self.l = None
        self.u = None

    def uniknots(self, nint, n=True):
        """
        Uniform distribution of internal knots across the datapoints (including any stab points).
        We don't make a difference between stab and real points.

        :param nint: The number of intervals, or the step
        :param n:
            If True, nint is the number of intervals (== piecewise polynoms) you want.
            If False : nint is a step in days you want between the knots (approximately).
        :type n: boolean

        .. note:: I also put all coeffs back to 0.0 !

        """

        a = self.datapoints.jds[0]
        b = self.datapoints.jds[-1]
        if n:
            intt = np.linspace(a, b, int(nint + 1))[1:-1]
        else:
            intt = np.linspace(a, b, int(float(b - a) / float(nint)))[1:-1]

        if len(intt) == 0: # pragma: no cover
            raise RuntimeError("I am uniknots, and I have only 0 (zero) internal knots ! Increase this number !")

        self.setintt(intt)
        self.knottype = "u"

        # Important : we put some 0 coeffs to go with the new knots
        self.resetc()

    def resetc(self):
        """
        Sets all coeffs to 0.0 -- if you want to start again your fit, keeping the knot positions.
        """
        self.c = np.zeros(len(self.t))

    def reset(self):
        """
        Calls uniknots, i.e. resets both coeffs and knot positions, keeping the same number of knots.
        """
        self.uniknots(self.getnint(), n=True)

    def buildbounds(self, verbose=True):
        """
        Build bounds for bok.
        By default I will make those bounds as wide as possible, still respecting epsilon.
        The parameter epsilon is the minimum distance two knots can have.
        If you give me a window size, I will not make the bounds as wide as possible, but only put them
        0.5*window days around the current knots (still respecting all this epsilon stuff of course).

        I look where your current knots are, and for each knots I build the bounds so that
        epsilon distance is respected between adjacent upper and lower bounds.
        But, there might already be knots only epsilon apart.
        So I'm a bit tricky, not so straightforward as my predecessors.

        Knots at the extrema are not allowed to move.

        Requires existing knots, puts lims in between them, and builds the bounds.
        """

        if verbose:
            print("Building BOK bounds (bokeps = %.3f, bokwindow = %s) ..." % (self.bokeps, self.bokwindow))

        knots = self.getinttex()  # Including extremal knots (once).
        n = len(knots)

        # We start by checking the knot spacing
        knotspacings = knots[1:] - knots[:-1]
        if not np.alltrue(knotspacings > 0.0): # pragma: no cover
            raise RuntimeError("Ouch, your knots are not sorted !")
        minspace = np.min(knotspacings)
        if verbose:
            print("Minimal knot spacing : %.3f" % minspace)

        if minspace < self.bokeps - 0.00001: # pragma: no cover  # Rounding errors, we decrease epsilon a bit...
            # If this does still happens, then it was not just a rounding error ...
            # Yes it still happens, due to updatedp stretch ...
            raise RuntimeError("Knot spacing min = %f, epsilon = %f" % (minspace, self.bokeps))

        # Loop through the knots.
        lowers = [knots[0]]  # First knot is not allowed to move
        uppers = [knots[0]]
        for i in range(1, n - 1):  # Internal knots
            tk = knots[i]  # this knot
            pk = knots[i - 1]  # previous knot
            nk = knots[i + 1]  # next knot

            # First we build the wide bounds :
            guessl = 0.5 * (pk + tk) + 0.5 * self.bokeps
            if guessl >= tk:
                guessl = tk

            guessu = 0.5 * (nk + tk) - 0.5 * self.bokeps
            if guessu <= tk:
                guessu = tk

            # Now we see if the use wants a narrower window within those bounds :
            if self.bokwindow is not None:
                if tk - 0.5 * self.bokwindow >= guessl:
                    guessl = tk - 0.5 * self.bokwindow
                if tk + 0.5 * self.bokwindow <= guessu:
                    guessu = tk + 0.5 * self.bokwindow

            lowers.append(guessl)
            uppers.append(guessu)

        # And now this last knot, doesn't move, like the first one:
        lowers.append(knots[-1])
        uppers.append(knots[-1])
        self.l = np.array(lowers)
        self.u = np.array(uppers)
        self.knottype += "l"
        if verbose:
            print("Buildbounds done.")

    def bok(self, bokmethod="BF", verbose=True):
        """
        We optimize the positions of knots by some various techniques.
        We use fixed bounds for the exploration, run buildbounds (with low epsilon) first.
        This means that I will not move my bounds.

        For each knot, i will try ntestpos linearly spaced positions within its bounds.
        In this version, the bounds are included : I might put a knot on a bound !
        The way the bounds are placed by buildbounds ensures that in any case the minimal
        distance of epsilon is respected.

        Using this sheme, it is now possible to iteratively call mybok and buildbounds in a loop
        and still respect epsilon at any time.


        :param bokmethods: string
            - MCBF : Monte Carlo brute force with ntestpos trial positions for each knot
            - BF : brute force, deterministic. Call me twice
            - fminind : fminbound on one knot after the other.
            - fmin :global fminbound
        :param verbose: verbosity

        Exit is automatic, if result does not improve anymore...
        """
        intknots = self.getintt()  # only internal, the ones we will move
        nintknots = len(intknots)
        weights = 1.0 / self.datapoints.magerrs

        def score(intknts, index, value):
            modifknots = intknts.copy()
            modifknots[index] = value
            return \
                si.splrep(self.datapoints.jds, self.datapoints.mags, w=weights, xb=None, xe=None, k=self.k, task=-1,
                          s=None,
                          t=modifknots, full_output=True, per=False, quiet=True)[1]

        iniscore = score(intknots, 0, intknots[0])
        lastchange = 1
        lastscore = iniscore
        iterations = 0
        if verbose:
            print("Starting BOK-%s on %i intknots (boktests = %i)" % (bokmethod, nintknots, self.boktests))

        if bokmethod == "MCBF":

            while True:
                if lastchange >= 2 * nintknots:  # somewhat arbitrary, but why not.
                    break
                i = np.random.randint(0, nintknots)  # (inclusive, exclusive)

                testknots = np.linspace(self.l[i + 1], self.u[i + 1], int(self.boktests))
                # +1, as u and l include extremal knots...
                # So we include the extremas in our range to test.

                testscores = np.array([score(intknots, i, testknot) for testknot in testknots])
                bestone = np.argmin(testscores)

                bestscore = testscores[bestone]
                if bestscore < lastscore:
                    lastchange = 0
                intknots[i] = testknots[bestone]  # WE UPDATE the intknots array !
                lastscore = bestscore
                lastchange += 1
                iterations += 1

        if bokmethod == "BF":

            intknotindices = list(
                range(nintknots))  # We could potentially change the order, just to see if that makes sense.

            for i in intknotindices:
                testknots = np.linspace(self.l[i + 1], self.u[i + 1], int(self.boktests))
                # +1, as u and l include extremal knots...
                # So we include the extremas in our range to test.

                testscores = np.array([score(intknots, i, testknot) for testknot in testknots])
                bestone = np.argmin(testscores)

                bestscore = testscores[bestone]
                intknots[i] = testknots[bestone]  # WE UPDATE the intknots array !
                iterations += 1


        if bokmethod == "fminind":
            intknotindices = list(range(nintknots))
            for i in intknotindices:

                def target(value):
                    return score(intknots, i, value)

                out = spopt.fminbound(target, self.l[i + 1], self.u[i + 1], xtol=0.01, maxfun=100, full_output=True,
                                      disp=1)
                # print out
                optval = out[0]
                bestscore = out[1]

                intknots[i] = optval  # WE UPDATE the intknots array !
                iterations += 1

        if bokmethod == "fmin":
            def target(modifknots):
                return \
                    si.splrep(self.datapoints.jds, self.datapoints.mags, w=weights, xb=None, xe=None, k=self.k, task=-1,
                              s=None, t=modifknots, full_output=True, per=0, quiet=1)[1]

            bounds = [(a, b) for (a, b) in zip(self.l[1:-1], self.u[1:-1])]

            out = spopt.fmin_l_bfgs_b(target, intknots, approx_grad=True, bounds=bounds, m=10, factr=1e7, pgtol=1.e-05,
                                      epsilon=1e-04, iprint=-1, maxfun=15000)
            intknots = out[0]
            bestscore = out[1]

        # relative improvement :
        relimp = (iniscore - bestscore) / iniscore
        self.knottype += "b"
        self.setintt(intknots)

        self.optc()  # Yes, not yet done !
        finalr2 = self.r2(nostab=True)
        if verbose:
            print("r2 = %f (without stab poins)" % finalr2)
            print("Done in %i iterations, relative improvement = %f" % (iterations, relimp))
        # We count all datapoints here, as score returns the full chi2 including stab pts.

        return finalr2

    # Some stuff about knots :

    def getintt(self):
        """
        Returns the internal knots (i.e., not even the datapoints extrema)
        This is what you need to feed into splrep !
        There are nint - 1 such knots
        """
        return self.t[(self.k + 1):-(self.k + 1)].copy()  # We cut the outer knots.

    def getinttex(self):
        """
        Same as above, but we include the extremal points "once".
        """
        return self.t[self.k:-self.k].copy()

    def knotstats(self):
        """
        Returns a string describing the knot spacing
        """
        knots = self.getinttex()
        spacings = knots[1:] - knots[:-1]
        return " ".join(["%.1f" % spacing for spacing in sorted(spacings)])

    def setintt(self, intt):
        """
        Give me some internal knots (not even containing the datapoints extrema),
        and I build the correct total knot vector t for you.
        I add the extremas, with appropriate multiplicity.
        """

        # Ok a quick test for consisency :

        if len(intt) == 0: # pragma: no cover
            raise RuntimeError("Your list of internal knots is empty !")

        if not self.datapoints.jds[0] < intt[0]: # pragma: no cover
            raise RuntimeError("Ouch.")
        if not self.datapoints.jds[-1] > intt[-1]: # pragma: no cover
            raise RuntimeError("Ouch.")
        # assert self.datapoints.jds[0] < intt[0] # should we put <= here ?
        # assert self.datapoints.jds[-1] > intt[-1]

        pro = self.datapoints.jds[0] * np.ones(self.k + 1)
        post = self.datapoints.jds[-1] * np.ones(self.k + 1)

        self.t = np.concatenate((pro, intt, post))

    def setinttex(self, inttex):
        """
        Including extremal knots
        """
        # pro = self.datapoints.jds[0] * np.ones(self.k)
        # post = self.datapoints.jds[-1] * np.ones(self.k)
        pro = inttex[0] * np.ones(self.k)
        post = inttex[-1] * np.ones(self.k)

        self.t = np.concatenate((pro, inttex, post))

    def getnint(self):
        """
        Returns the number of intervals
        """
        return len(self.t) - 2 * (self.k + 1) + 1

    # Similar stuff about coeffs :

    def getc(self, m=0):
        """
        Returns all active coefficients of the spline, the ones it makes sense to play with.
        The length of this guy is number of intervals - 2 !
        """
        return self.c[m:-(self.k + 1 + m)].copy()

    def setc(self, c, m=0):
        """
        Puts the coeffs from getc back into place.
        """
        self.c[m:-(self.k + 1 + m)] = c

    def getco(self, m=0):
        """
        Same as getc, but reorders the coeffs in a way more suited for nonlinear optimization
        """
        c = self.getc(m=m)
        mid = int(len(c) / 2.0)
        return np.concatenate([c[mid:], c[:mid][::-1]])

    def setco(self, c, m=0):
        """
        The inverse of getco.
        """
        mid = int(len(c) / 2.0)
        self.setc(np.concatenate([c[mid + 1:][::-1], c[:mid + 1]]), m=m)

    def setcflat(self, c):
        """
        Give me coeffs like those from getc(m=1), I will set the coeffs so that the spline extremas
        are flat (i.e. slope = 0).
        """

        self.setc(c, m=1)
        self.c[0] = self.c[1]
        self.c[-(self.k + 2)] = self.c[-(self.k + 3)]

    def setcoflat(self, c):
        """
        idem, but for reordered coeffs.
        """
        mid = int(len(c) / 2.0)
        self.setcflat(np.concatenate([c[mid:][::-1], c[:mid]]))

    def r2(self, nostab=True, nosquare=False):
        """
        Evaluates the spline, compares it with the data points and returns a weighted sum of residuals r2.

        If nostab = False, stab points are included
        This is precisely the same r2 as is used by splrep for the fit, and thus the same value as
        returned by optc !

        This method can set lastr2nostab, so be sure to end any optimization with it.

        If nostab = True, we don't count the stab points
        """

        if nostab:
            splinemags = self.eval(nostab=True, jds=None)
            errs = self.datapoints.mags[self.datapoints.mask] - splinemags
            werrs = errs / self.datapoints.magerrs[self.datapoints.mask]
            if nosquare:
                r2 = np.sum(np.fabs(werrs))
            else:
                r2 = np.sum(werrs * werrs)
            self.lastr2nostab = r2
        else:
            splinemags = self.eval(nostab=False, jds=None)
            errs = self.datapoints.mags - splinemags
            werrs = errs / self.datapoints.magerrs
            if nosquare:
                r2 = np.sum(np.fabs(werrs))
            else:
                r2 = np.sum(werrs * werrs)
            self.lastr2stab = r2

        return r2

    def tv(self):
        """
        Returns the total variation of the spline. Simple !
        http://en.wikipedia.org/wiki/Total_variation

        """

        # Method 1 : linear approximation

        ptd = 5  # point density in days ... this is enough !

        a = self.t[0]
        b = self.t[-1]
        x = np.linspace(a, b, int((b - a) * ptd))
        y = self.eval(jds=x)
        tv1 = np.sum(np.fabs(y[1:] - y[:-1]))

        return tv1

    def optc(self):
        """
        Optimize the coeffs, don't touch the knots
        This is the fast guy, one reason to use splines :-)
        Returns the chi2 in case you want it (including stabilization points) !

        Sets lastr2stab, but not lastr2nostab !

        """

        out = si.splrep(self.datapoints.jds, self.datapoints.mags, w=1.0 / self.datapoints.magerrs, xb=None, xe=None,
                        k=self.k, task=-1, s=None, t=self.getintt(), full_output=1, per=0, quiet=1)
        # We check if it worked :
        if not out[2] <= 0: # pragma: no cover
            raise RuntimeError("Problem with spline representation, message = %s" % (out[3]))

        self.c = out[0][1]  # save the coeffs

        # import matplotlib.pyplot as plt
        # plt.plot(self.datapoints.jds, self.datapoints.magerrs)
        # plt.show()

        self.lastr2stab = out[1]
        return out[1]

    def optcflat(self, verbose=False):
        """
        Optimizes only the "border coeffs" so to get zero slope at the extrema
        Run optc() first ...
        This has to be done with an iterative optimizer
        """

        full = self.getc(m=1)
        inip = self.getc(m=1)[[0, 1, -2, -1]]  # 4 coeffs

        def setp(p):
            full[[0, 1, -2, -1]] = p
            self.setcflat(full)

        if verbose:
            print("Starting flat coeff optimization ...")
            print("Initial pars : ", inip)

        def errorfct(p):
            setp(p)
            return self.r2(nostab=False)  # To get the same as optc would return !

        minout = spopt.fmin_powell(errorfct, inip, full_output=1, disp=verbose)
        popt = minout[0]
        if popt.shape == ():
            popt = np.array([popt])

        if verbose:
            print("Optimal pars : ", popt)
        setp(popt)
        return self.r2(nostab=False)  # We include the stab points, like optc does.

    # This last line also updates self.lastr2 ...

    def eval(self, jds=None, nostab=True):
        """
        Evaluates the spline at jds, and returns the corresponding mags-like vector.
        By default, we exclude the stabilization points !
        If jds is not None, we use them instead of our own jds (in this case excludestab makes no sense)
        """
        if jds is None:
            if nostab:
                jds = self.datapoints.jds[self.datapoints.mask]
            else:
                jds = self.datapoints.jds
        else:
            # A minimal check for non-extrapolation condition should go here !
            pass

        fitmags = si.splev(jds, (self.t, self.c, self.k))
        # By default ext=0 : we do return extrapolated values
        return fitmags

    def display(self, showbounds=True, showdatapoints=True, showerrorbars=True, figsize=(16, 8), filename=None):
        """
        A display of the spline object, with knots, jds, stab points, etc.
        For debugging and checks.
        """

        plt.figure(figsize=figsize)

        if showdatapoints:
            if showerrorbars:
                mask = self.datapoints.mask
                plt.errorbar(self.datapoints.jds[mask], self.datapoints.mags[mask], yerr=self.datapoints.magerrs[mask],
                             linestyle="None", color="blue")
                if not np.alltrue(mask):
                    mask = mask == False
                    plt.errorbar(self.datapoints.jds[mask], self.datapoints.mags[mask],
                                 yerr=self.datapoints.magerrs[mask], linestyle="None", color="gray")

            else:
                plt.plot(self.datapoints.jds, self.datapoints.mags, "b,")

        if np.any(self.t) is not None:

            if getattr(self, "showknots", True):
                for knot in self.t:
                    plt.axvline(knot, color="gray")

            # We draw the spline :
            xs = np.linspace(self.datapoints.jds[0], self.datapoints.jds[-1], 1000)
            ys = self.eval(jds=xs)
            plt.plot(xs, ys, "b-")

        if showbounds:
            if (np.any(self.l) is not None) and (np.any(self.u) is not None):
                for l in self.l:
                    plt.axvline(l, color="blue", dashes=(4, 4))
                for u in self.u:
                    plt.axvline(u, color="red", dashes=(5, 5))

        axes = plt.gca()
        axes.set_ylim(axes.get_ylim()[::-1])

        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)
