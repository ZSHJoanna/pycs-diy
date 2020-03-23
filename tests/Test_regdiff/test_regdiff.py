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

class TestLightCurve(unittest.TestCase):
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

    def test_regdiff(self):
        lc_copy = [lc.copy() for lc in self.lcs]
        myrslcs, error_fct = regdiff(lc_copy, pointdensity=2, covkernel='matern',
                                     pow=1.7, amp=0.5, scale=200.0, errscale=20, verbose=True, method="weights")
        lc_func.display(lc_copy, myrslcs, filename=os.path.join(self.outpath, 'regdiff.png'))
        delays = lc_func.getdelays(lc_copy, to_be_sorted=True)
        delays_th = [-6.31,-22.51,-67.44,-16.19,-61.13,-44.94]
        assert_allclose(delays, delays_th, atol=0.2)
        assert error_fct <= 0.006



if __name__ == '__main__':
    pytest.main()