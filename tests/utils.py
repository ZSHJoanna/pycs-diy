import pycs3.spl.topopt
import pycs3.regdiff.multiopt as multiopt
import pycs3.sim.twk


def spl(lcs):
    spline = pycs3.spl.topopt.opt_rough(lcs, nit=5, knotstep=30)
    spline = pycs3.spl.topopt.opt_fine(lcs, nit=10, knotstep=20)
    return spline


def regdiff(lcs, **kwargs):
    return multiopt.opt_ts(lcs, pd=kwargs['pd'], covkernel=kwargs['covkernel'], pow=kwargs['pow'],
                           amp=kwargs['amp'], scale=kwargs['scale'], errscale=kwargs['errscale'], verbose=True,
                           method="weights")


# The small scale extrinsic variability, used to generated the synthetic curves:
def Atweakml(lcs, spline):
    return pycs3.sim.twk.tweakml(lcs,spline, beta=-1.5, sigma=0.25, fmin=1 / 500.0, fmax=None, psplot=False)


def Btweakml(lcs, spline):
    return pycs3.sim.twk.tweakml(lcs,spline, beta=-1.0, sigma=0.9, fmin=1 / 500.0, fmax=None, psplot=False)


def Ctweakml(lcs,spline):
    return pycs3.sim.twk.tweakml(lcs,spline, beta=-1.0, sigma=1.5, fmin=1 / 500.0, fmax=None, psplot=False)


def Dtweakml(lcs,spline):
    return pycs3.sim.twk.tweakml(lcs,spline, beta=-0.0, sigma=4.5, fmin=1 / 500.0, fmax=None, psplot=False)
