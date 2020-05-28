"""
Global optimizers that make use of the building blocks from multiopt.py
They return a float, the dispersion value.
They can be used with :py:func:`pycs.sim.run.multirun`
We do not redefine the ML, but we keep it.
"""
import pycs3.disp.multiopt
import pycs3.gen.lc


def opt_full(lcs, rawdispersionmethod, nit=5, verbose=True):
    """
    A first optimization of ml and ts, alternatively doing one after the other.
    Works great, seems to be a keeper.
    Note that I do **keep** the ML.

    You can put a rawdispersionmethod (i.e. non symmetric) into these guys, as they will anyway apply it on AB and BA for every pair AB.
    """

    if verbose:
        print("Starting dispersion optimization of :\n%s" % ("\n".join([str(lc) for lc in lcs])))
        print("Initial delays :")
        print("%s" % (pycs3.gen.lc_func.getnicetimedelays(lcs, separator=" | ")))

    # We start with some magnitude shifts :
    pycs3.disp.multiopt.opt_magshift(lcs, rawdispersionmethod, verbose=False)

    for i in range(nit):

        d2 = pycs3.disp.multiopt.opt_ml(lcs, rawdispersionmethod, maxit=1, verbose=False)
        d2 = pycs3.disp.multiopt.opt_ts_mix(lcs, rawdispersionmethod, movefirst=False, verbose=False)

        if verbose:
            print("Iteration %i done, d2 = %8.3f" % (i + 1, d2))
            print("%s" % (pycs3.gen.lc_func.getnicetimedelays(lcs, separator=" | ")))

    if verbose:
        print("Done with optimization of :\n%s" % ("\n".join([str(lc) for lc in lcs])))
    return d2
