import pycs3.spl.topopt
import pycs3.regdiff.multiopt as multiopt
import pycs3.disp.disps
import pycs3.disp.topopt

def spl(lcs):
	spline = pycs3.spl.topopt.opt_rough(lcs, nit=5, knotstep=30)
	spline = pycs3.spl.topopt.opt_fine(lcs, nit=10, knotstep=20)
	return spline

def regdiff(lcs, **kwargs):
	return multiopt.opt_ts(lcs, pd=kwargs['pd'], covkernel=kwargs['covkernel'], pow=kwargs['pow'],
										amp=kwargs['amp'], scale=kwargs['scale'], errscale=kwargs['errscale'], verbose=True, method="weights")


rawdispersionmethod = lambda lc1, lc2: pycs3.disp.disps.linintnp(lc1, lc2, interpdist=30.0)
dispersionmethod = lambda lc1, lc2: pycs3.disp.disps.symmetrize(lc1, lc2, rawdispersionmethod)

def disp(lcs):
	return pycs3.disp.topopt.opt_full(lcs, rawdispersionmethod, nit=5, verbose=True)