"""
Wrapper around pymc's GP module

"""
import os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, RationalQuadratic
from sklearn.gaussian_process.kernels import ConstantKernel
import numpy as np


def regression(x, y, yerr, mean, covkernel='matern', pow=1.5, amp=2.0, scale=200.0, errscale=5.0, verbose=True):
    """
    Give me data points

    yerr is the 1sigma error of each y

    I return a function : you pass an array of new x, the func returns (newy, newyerr)

    pow, amp and scale are params for the covariance function.

    """
    obs_mesh = x.reshape(-1, 1)
    obs_vals = y
    # jds_new = np.atleast_2d(x).T
    obs_v = (yerr) **2  # Converting std to variance

    if verbose:
        print("Computing GPR with params pow=%.1f, amp=%.1f, scale=%.1f, errscale=%.1f" % (pow, amp, scale, errscale))

    # v4, allow you to chose your kernel.
    if covkernel == "matern":
        kernel = ConstantKernel() + amp*Matern(length_scale=scale, nu=pow)+ WhiteKernel(noise_level=errscale)
    elif covkernel =="RBF": # RBF is mattern when nu --> inf
        kernel =  ConstantKernel() + amp*RBF(length_scale=scale) + WhiteKernel(noise_level=errscale)
    elif covkernel == "RatQuad": #alpha is the scale mixture parameter
        kernel = ConstantKernel() + amp*RationalQuadratic(length_scale=scale, alpha=pow) + WhiteKernel(noise_level=errscale)
    else:
        raise RuntimeError("I do not know the covariance kernel you gave me ! %s" % covkernel)
    obs_v *= errscale

    # Impose observations on the GP
    gp = GaussianProcessRegressor(kernel=kernel, alpha=obs_v)
    gp.fit(obs_mesh, obs_vals)

    def outfct(jds):  # this is made to speed up the code, no need to refit the GP
        m_out, v_out = gp.predict(jds.reshape(-1, 1), return_std=True) #this retunr std not variance
        newy = m_out
        newyerr = v_out
        return newy, newyerr

    return outfct
