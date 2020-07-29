"""
Wrapper around Sklearn GP module

"""
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, RationalQuadratic


def regression(x, y, yerr, covkernel='matern', pow=1.5, amp=1.0, scale=200.0, errscale=1.0, verbose=False):
    """
    Give me data points and error. ``yerr`` is the 1sigma error of each ``y``
    I return a function : you pass an array of new x, the func returns (newy, newyerr)

    .. warning:: ``pow`` is not used for RBF, nor for Ratquad. ``amp`` and ``scale`` are now fitted to the data, you now provide the starting point, it is better leave them to default value.

    :param x: array containing the time data
    :param y: array containing the magnitude data
    :param yerr: array containing the magnitude errors
    :param covkernel: Choose between "matern", "RatQuad" and "RBF". See scikit GP documentation for details
    :type covkernel: str
    :param pow: exponent coefficient of the covariance function
    :type pow: float
    :param amp: amplitude coefficient of the covariance function
    :type amp: float
    :param scale: characteristic time scale
    :type scale: float
    :param errscale: additional scaling of the photometric errors
    :type errscale: float
    :param verbose: verbosity
    :type verbose: bool

    :return: A python function to make the prediction

    """
    obs_mesh = x.reshape(-1, 1)
    obs_vals = y
    # jds_new = np.atleast_2d(x).T
    obs_v = (yerr) **2  # Converting std to variance
    mean_err = np.mean(obs_v)

    if verbose:
        print("Computing GPR with params covkernel=%s, pow=%.1f, errscale=%.1f" % (covkernel, pow, errscale))

    # v4, allow you to chose your kernel.
    if covkernel == "matern":
        kernel = ConstantKernel() + amp*Matern(length_scale=scale, nu=pow)+ WhiteKernel()
    elif covkernel =="RBF": # RBF is mattern when nu --> inf
        kernel =  ConstantKernel() + amp*RBF(length_scale=scale) + WhiteKernel()
    elif covkernel == "RatQuad": #alpha is the scale mixture parameter
        kernel = ConstantKernel() + amp*RationalQuadratic(length_scale=scale) + WhiteKernel()
    else: # pragma: no cover
        raise RuntimeError("I do not know the covariance kernel you gave me ! %s" % covkernel)
    obs_v *= errscale

    # Impose observations on the GP
    gp = GaussianProcessRegressor(kernel=kernel, alpha=obs_v, normalize_y =True)
    gp.fit(obs_mesh, obs_vals)
    if verbose :
        print("Kernel after optimisation :", gp.kernel_)

    def outfct(jds):  # this is made to speed up the code, no need to refit the GP
        m_out, v_out = gp.predict(jds.reshape(-1, 1), return_std=True) #this retunr std not variance
        newy = m_out
        newyerr = v_out
        return newy, newyerr

    return outfct
