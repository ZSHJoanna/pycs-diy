import pycs3.spl.topopt as toopt
import pycs3.regdiff.multiopt as multiopt

def spl(lcs):
	spline = toopt.opt_rough(lcs, nit=5, knotstep=30)
	spline = toopt.opt_fine(lcs, nit=10, knotstep=20)
	return spline

def regdiff(lcs, **kwargs):
	return multiopt.opt_ts(lcs, pd=kwargs['pd'], covkernel=kwargs['covkernel'], pow=kwargs['pow'],
										amp=kwargs['amp'], scale=kwargs['scale'], errscale=kwargs['errscale'], verbose=True, method="weights")