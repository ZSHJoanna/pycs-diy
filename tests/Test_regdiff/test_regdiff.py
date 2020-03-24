import os
import pytest
import unittest
from tests import TEST_PATH
import pycs3.gen.mrg as mrg
import pycs3.gen.lc_func as lc_func
import pycs3.regdiff.multiopt as multiopt
from numpy.testing import assert_allclose

def regdiff(lcs, **kwargs):
	return multiopt.opt_ts(lcs, pd=kwargs['pointdensity'], covkernel=kwargs['covkernel'], pow=kwargs['pow'],
										amp=kwargs['amp'], scale=kwargs['scale'], errscale=kwargs['errscale'], verbose=True, method="weights")

class TestRgediff(unittest.TestCase):
    def setUp(self):
        self.path = TEST_PATH
        self.outpath = os.path.join(self.path, "output")
        self.rdbfile = os.path.join(self.path, "data", "trialcurves.txt")
        self.lcs = [
            lc_func.rdbimport(self.rdbfile, object='A', magcolname='mag_A', magerrcolname='magerr_A',
                      telescopename="Trial"),
            lc_func.rdbimport(self.rdbfile, object='B', magcolname='mag_B', magerrcolname='magerr_B',
                      telescopename="Trial"),
            lc_func.rdbimport(self.rdbfile, object='C', magcolname='mag_C', magerrcolname='magerr_C',
                      telescopename="Trial"),
            lc_func.rdbimport(self.rdbfile, object='D', magcolname='mag_D', magerrcolname='magerr_D',
                      telescopename="Trial")
        ]
        mrg.colourise(self.lcs)

    def test_regdiff_mattern(self):
        #test several kernel :
        #Mattern kernel
        lc_copy = [lc.copy() for lc in self.lcs]
        lc_func.settimeshifts(lc_copy, shifts=[0, -5, -20, -60], includefirst=True) # intial guess
        myrslcs, error_fct = multiopt.opt_ts(lc_copy, pd=2, covkernel='matern',
                                             pow=1.5, amp=1., scale=200., errscale=1., verbose=True,
                                             method="weights")  # good set for mattern
        lc_func.display(lc_copy, myrslcs, filename=os.path.join(self.outpath, 'regdiff_mattern1.5.png'))
        delays = lc_func.getdelays(lc_copy, to_be_sorted=True)
        delays_th = [-4.39,-20.79,-70.52,-16.40,-66.13,-49.72]
        assert_allclose(delays, delays_th, atol=0.5)
        assert error_fct <= 0.01

        # RBF kernel
        lc_func.resetlcs(lc_copy)
        lc_func.settimeshifts(lc_copy, shifts=[0, -5, -20, -60], includefirst=True)  # intial guess
        myrslcs, error_fct = multiopt.opt_ts(lc_copy, pd=2, covkernel='RBF',
                pow=1.5, amp=0.5, scale=200., errscale=1., verbose=True, method="weights") # good set for Radial-Basis Function
        delays = lc_func.getdelays(lc_copy, to_be_sorted=True)
        delays_th = [-6.33,-21.68,-70.45,-15.35,-64.13,-48.77]
        assert_allclose(delays, delays_th, atol=0.5)
        assert error_fct <= 0.015

        # RatQuad kernal
        lc_func.resetlcs(lc_copy)
        lc_func.settimeshifts(lc_copy, shifts=[0, -5, -20, -60], includefirst=True)  # intial guess
        myrslcs, error_fct = multiopt.opt_ts(lc_copy, pd=2, covkernel='RatQuad',
                pow=1., amp=0.5, scale=200., errscale=1., verbose=True, method="weights") # good set for mattern
        delays = lc_func.getdelays(lc_copy, to_be_sorted=True)
        delays_th = [-4.37,-20.23,-69.98,-15.86,-65.62,-49.75]
        assert_allclose(delays, delays_th, atol=0.5)
        assert error_fct <= 0.015


if __name__ == '__main__':
    pytest.main()