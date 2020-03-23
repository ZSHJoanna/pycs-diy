"""
Optimization using regdiff

Would be great to have an opt_flux in here !

"""
import pycs3.regdiff.rslc
import pycs3.gen.lc
import pycs3.gen.lc_func


def opt_ts(lcs, method="weights", pd=5, covkernel="matern", pow=1.5, amp=2.0, scale=200.0, errscale=5.0, verbose=True):
    """
    Give me lightcurves (with more or less good initial time shifts)
    I run a regression on them, optimize regdiff, and set their delays to the optimal values.

    :param pd: the point density, in points per days.

    The parameters pow, amp, scale, errscale are passed to the GPR, see its doc (or explore their effect on the GPR before runnign this...)
    """

    if verbose:
        print("Starting regdiff opt_ts, initial time delays :")
        print("%s" % (pycs3.gen.lc_func.getnicetimedelays(lcs, separator=" | ")))

    rss = [pycs3.regdiff.rslc.factory(l, pd=pd, covkernel=covkernel, pow=pow, amp=amp, scale=scale, errscale=errscale)
           for l in lcs]
    # The time shifts are transfered to these rss, any microlensing is disregarded

    if verbose:
        print("Regressions done.")

    minwtv = pycs3.regdiff.rslc.opt_rslcs(rss, method=method, verbose=True)

    for (l, r) in zip(lcs, rss):
        l.timeshift = r.timeshift
        l.commentlist.append("Timeshift optimized with regdiff.")

    if verbose:
        print("Optimization done ! Optimal time delays :")
        print("%s" % (pycs3.gen.lc_func.getnicetimedelays(lcs, separator=" | ")))

    return rss, minwtv