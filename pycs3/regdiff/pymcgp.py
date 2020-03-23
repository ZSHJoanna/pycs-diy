"""
Wrapper around pymc's GP module

"""
import os

os.environ['OMP_NUM_THREADS'] = "1"

import pymc.gp
import pymc.gp.GPutils
from pymc.gp.cov_funs import matern
from pymc.gp.cov_funs import gaussian
from pymc.gp.cov_funs import pow_exp

import numpy as np


def regression(x, y, yerr, priormeanfct, covkernel='matern', pow=1.5, amp=2.0, scale=200.0, errscale=5.0, verbose=True):
    """
    Give me data points

    yerr is the 1sigma error of each y

    I return a function : you pass an array of new x, the func returns (newy, newyerr)

    pow, amp and scale are params for the covariance function.

    """
    obs_mesh = x
    obs_vals = y
    obs_v = yerr * yerr  # Converting std to variance

    if verbose:
        print("Computing GPR with params pow=%.1f, amp=%.1f, scale=%.1f, errscale=%.1f" % (pow, amp, scale, errscale))
    # pymc GP objects :
    mean = pymc.gp.Mean(priormeanfct)

    # v4, allow you to chose your kernel.
    if covkernel == "matern":
        C = pymc.gp.Covariance(eval_fun=matern.euclidean, diff_degree=pow, amp=amp, scale=scale)
    elif covkernel == "pow_exp":
        C = pymc.gp.Covariance(eval_fun=pow_exp.euclidean, pow=pow, amp=amp, scale=scale)
    elif covkernel == "gaussian":
        C = pymc.gp.Covariance(eval_fun=gaussian.euclidean, amp=amp, scale=scale)
    else:
        raise RuntimeError("I do not know the covariance kernel you gave me ! %s" % covkernel)
    obs_v *= errscale

    # Impose observations on the GP
    pymc.gp.GPutils.observe(mean, C, obs_mesh=obs_mesh, obs_V=obs_v, obs_vals=obs_vals)

    def outfct(jds):
        # Ok, so here is the issue. That fucker GPutils.point_eval does not want to be called through a process from multiprocess.Pool. It just stops its execution, and stares blankly at you.
        (m_out, v_out) = pymc.gp.GPutils.point_eval(mean, C, jds)
        newy = m_out
        newyerr = np.sqrt(v_out)

        return newy, newyerr

    return outfct
