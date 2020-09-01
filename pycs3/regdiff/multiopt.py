"""
Optimization using regdiff

Would be great to have an opt_flux in here !

"""
import pycs3.gen.lc
import pycs3.gen.lc_func
import pycs3.regdiff.rslc
import logging
logger = logging.getLogger(__name__)


def opt_ts(lcs, method="weights", pd=2., covkernel="matern", pow=1.5, amp=1.0, scale=200.0, errscale=1.0, verbose=True):
    """
    Give me lightcurves (with more or less good initial time shifts)
    I run a regression on them, optimize regdiff, and set their delays to the optimal values.

    :param method: optimisation method. Choose between "weights" and "simple" (default : "weights")
    :type method: str
    :param pd: the point density, in points per days.
    :type pd: float
    :param covkernel: Choose between "matern","RatQuad" and "RBF". See scikit GP documentation for details
    :type covkernel: str
    :param pow: float, exponent coefficient of the covariance function
    :type pow: float
    :param amp: initial amplitude coefficient of the covariance function
    :type amp: float
    :param scale: float, initial characteristic time scale
    :type scale: amp
    :param errscale: additional scaling of the photometric error
    :type errscale: float

    The parameters pow, amp, scale, errscale are passed to the GPR, see its doc (or explore their effect on the GPR before runnign this...)
    """

    if verbose:
        logger.info("Starting regdiff opt_ts, initial time delays :")
        logger.info("%s" % (pycs3.gen.lc_func.getnicetimedelays(lcs, separator=" | ")))

    rss = [pycs3.regdiff.rslc.factory(l, pd=pd, covkernel=covkernel, pow=pow, amp=amp, scale=scale, errscale=errscale)
           for l in lcs]
    # The time shifts are transfered to these rss, any microlensing is disregarded

    if verbose:
        logger.info("Regressions done.")

    minwtv = pycs3.regdiff.rslc.opt_rslcs(rss, method=method, verbose=verbose)

    for (l, r) in zip(lcs, rss):
        l.timeshift = r.timeshift
        l.commentlist.append("Timeshift optimized with regdiff.")

    if verbose:
        logger.info("Optimization done ! Optimal time delays :")
        logger.info("%s" % (pycs3.gen.lc_func.getnicetimedelays(lcs, separator=" | ")))

    return rss, minwtv
