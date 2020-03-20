import os
import sys
import pytest
import unittest

from tests import TEST_PATH
from pycs3.gen.polyml import addtolc
import pycs3.spl.topopt as toopt
from pycs3.gen.util import writepickle
import pycs3.gen.lc_func as lc_func
from numpy.testing import assert_allclose

def spl(lcs):
	spline = toopt.opt_rough(lcs, nit=5, knotstep=30)
	spline = toopt.opt_fine(lcs, nit=10, knotstep=20)
	return spline

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

    def test_lc_infos(self):
        print(self.lcs[0])
        print(self.lcs[0].printinfo()) #call lc.longinfo()
        stats = {'len': 192, 'nseas': 4, 'meansg': 163.99396333333425, 'minsg': 130.38524000000325, 'maxsg': 210.23221999999805, 'stdsg': 33.7985590219782, 'med': 3.989030000000639, 'mean': 4.38115308510637, 'max': 25.124900000002526, 'min': 0.8326200000010431, 'std': 3.460329376531179}
        test_stat = self.lcs[0].samplingstats()
        for key in test_stat.keys():
            self.assertAlmostEqual(stats[key],test_stat[key], places=3)
        commonproperties = self.lcs[0].commonproperties()
        commonproperties2 = self.lcs[0].commonproperties(notonlycommon=True)

    def test_opt_spline(self):
        addtolc(self.lcs[1], nparams=2, autoseasonsgap=60.0)  # add affine microlensing to each season
        addtolc(self.lcs[2], nparams=3, autoseasonsgap=600.0)  # add polynomial of degree 2 on the entire light curve
        addtolc(self.lcs[3], nparams=3, autoseasonsgap=600.0)
        spline = spl(self.lcs)
        delays = lc_func.getdelays(self.lcs, to_be_sorted=True)
        lc_func.getnicetimedelays(self.lcs)
        lc_func.display(self.lcs, [spline], filename=os.path.join(self.outpath, 'spline_wi_ml.png'))
        delays_th = [-6.44376707514413, -26.199323016152675, -70.92455333399347, -19.755555941008545,
                     -64.48078625884935, -44.7252303178408]

        assert_allclose(delays, delays_th, atol=0.2)



if __name__ == '__main__':
    pytest.main()
