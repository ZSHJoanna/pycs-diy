"""
Testing script, light curves, microlensing
"""

import os
import sys

# add the root directory to sys.path, so pycs3 can be imported without installation
pycs3_path = os.path.join(os.path.dirname(os.path.realpath(sys.argv[0])), '../')
path = os.path.dirname(os.path.realpath(sys.argv[0]))
sys.path.append(pycs3_path)

rdbfile = os.path.join(path, "data", "trialcurves.txt")

from pycs3.gen.util import readpickle
from pycs3.gen.lc_func import display, getnicetimedelays, getdelays
from pycs3.gen.polyml import addtolc
from pycs3.spl.topopt import opt_fine, opt_rough
from numpy.testing import assert_allclose

# A simple attempt to get a multi-purpose free-knot spline method :
def spl(lcs):
	spline = opt_rough(lcs, nit=5, knotstep=50)
	for l in lcs:
		l.resetml()
	spline = opt_rough(lcs, nit=5, knotstep=30)
	spline = opt_fine(lcs, nit=10, knotstep=20)
	return spline

lcs = readpickle("data/trialcurves.pkl")

# Let's try some curve shifting, without correcting for extrinsic variability...
# With the freek-knot spline technique :
spline = spl(lcs)

# Show the result
display(lcs, [spline], filename=os.path.join(path,'output','spline_wo_ml.png'))
# (It doesn't look good, these curves don't overlap without a microlensing model)

# For humans, at any time we can print out the time delays between the curves,
# computed from the current time shifts of each curve :
print(getnicetimedelays(lcs, separator="\n", to_be_sorted=True))

# To get better results, we need to add microlensing models, e.g., polynomials
# to our light curves.
addtolc(lcs[1], nparams=2, autoseasonsgap=60.0) #add affine microlensing to each season
addtolc(lcs[2], nparams=3, autoseasonsgap=600.0) # add polynomial of degree 2 on the entire light curve
addtolc(lcs[3], nparams=3, autoseasonsgap=600.0)
# (this choice is just an illustration)

# Let's try the free-knot spline optimization again :
spline = spl(lcs)
delays = getdelays(lcs, to_be_sorted=True)
print(delays)
display(lcs, [spline], filename=os.path.join(path, 'output','spline_wi_ml.png'))
delays_th = [-6.44376707514413, -26.199323016152675, -70.92455333399347, -19.755555941008545, -64.48078625884935, -44.7252303178408]

assert_allclose(delays, delays_th, atol=0.2)


