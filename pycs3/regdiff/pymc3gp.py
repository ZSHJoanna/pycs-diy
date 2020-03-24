"""
Wrapper around pymc's GP module

"""
import os

os.environ['OMP_NUM_THREADS'] = "1"

from pymc3 import find_MAP
import pymc3.gp.mean
from pymc3 import Model
from pymc3.gp import Marginal, Latent
import pymc3.gp.cov as cov
import numpy as np


def regression(x, obs_vals, yerr, mean, jds_new, covkernel='matern', pow=1.5, amp=2.0, scale=200.0, errscale=5.0, verbose=True):
    """
    Give me data points

    yerr is the 1sigma error of each y

    I return a function : you pass an array of new x, the func returns (newy, newyerr)

    pow, amp and scale are params for the covariance function.

    """
    obs_mesh = x[:,None]
    # obs_vals = y
    jds_new = jds_new[:,None]
    obs_v = yerr * yerr  # Converting std to variance

    if verbose:
        print("Computing GPR with params pow=%.1f, amp=%.1f, scale=%.1f, errscale=%.1f" % (pow, amp, scale, errscale))
    # pymc GP objects :
    mean = pymc3.gp.mean.Constant(mean)
    with Model() as model:
        # v4, allow you to chose your kernel.
        if covkernel == "matern32":
            cov_func = amp * cov.Matern32(1, ls = scale)
        elif covkernel == "matern52":
            cov_func = amp * cov.Matern52(1, ls = scale)
        elif covkernel == "exponential":
            cov_func = amp * cov.Exponential(1, ls = scale)
        else:
            raise RuntimeError("I do not know the covariance kernel you gave me ! %s" % covkernel)
        obs_v *= errscale

        # Impose observations on the GP
        gp = Latent(mean_func = mean, cov_func=cov_func)
        f = gp.prior("f", X=obs_mesh)

    # pymc.gp.GPutils.observe(mean, cov_func, obs_mesh=obs_mesh, obs_V=obs_v, obs_vals=obs_vals)
    with model :
        f_pred = gp.conditional("f_pred", jds_new)
        # Ok, so here is the issue. That fucker GPutils.point_eval does not want to be called through a process from multiprocess.Pool. It just stops its execution, and stares blankly at you.
        m_out, v_out = gp.predict(jds_new)
        newy = m_out
        newyerr = np.sqrt(v_out)

    return newy, newyerr
