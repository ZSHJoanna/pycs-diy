import os
import numpy as np
import pytest
import unittest

from tests import TEST_PATH
from pycs3.gen.polyml import addtolc
import pycs3.spl.topopt as toopt
import pycs3.gen.mrg as mrg
import pycs3.gen.lc_func as lc_func
from numpy.testing import assert_allclose, assert_almost_equal, assert_array_equal

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
        mrg.colourise(self.lcs)

    def test_lc_infos(self):
        print(self.lcs[0])
        print(self.lcs[0].printinfo()) #call lc.longinfo()
        stats = {'len': 192, 'nseas': 4, 'meansg': 163.99396333333425, 'minsg': 130.38524000000325, 'maxsg': 210.23221999999805, 'stdsg': 33.7985590219782, 'med': 3.989030000000639, 'mean': 4.38115308510637, 'max': 25.124900000002526, 'min': 0.8326200000010431, 'std': 3.460329376531179}
        test_stat = self.lcs[0].samplingstats()
        for key in test_stat.keys():
            self.assertAlmostEqual(stats[key],test_stat[key], places=3)
        commonproperties = self.lcs[0].commonproperties()
        commonproperties2 = self.lcs[0].commonproperties(notonlycommon=True)
        jds_ranges = (54018.545346499996, 55465.7478835)
        mag_ranges = (-9.51947, -13.70035)
        test_jds_ranges, test_mag_ranges = lc_func.displayrange(self.lcs)
        assert_almost_equal(test_jds_ranges,jds_ranges, decimal=3)
        assert_almost_equal(test_mag_ranges,mag_ranges, decimal=3)


    def test_opt_spline(self):
        addtolc(self.lcs[1], nparams=2, autoseasonsgap=60.0)  # add affine microlensing to each season
        addtolc(self.lcs[2], nparams=3, autoseasonsgap=600.0)  # add polynomial of degree 2 on the entire light curve
        addtolc(self.lcs[3], nparams=3, autoseasonsgap=600.0)
        spline = spl(self.lcs)
        delays = lc_func.getdelays(self.lcs, to_be_sorted=True)
        lc_func.getnicetimedelays(self.lcs)
        lc_func.display(self.lcs, [spline],style="homepagepdf", filename=os.path.join(self.outpath, 'spline_wi_ml1.png'))
        lc_func.display(self.lcs, [spline],style="homepagepdfnologo", filename=os.path.join(self.outpath, 'spline_wi_ml2.png'))
        lc_func.display(self.lcs, [spline],style="2m2", filename=os.path.join(self.outpath, 'spline_wi_ml3.png'))
        lc_func.display(self.lcs, [spline],style="posterpdf", filename=os.path.join(self.outpath, 'spline_wi_ml4.png'))
        lc_func.display(self.lcs, [spline],style="internal", filename=os.path.join(self.outpath, 'spline_wi_ml5.png'))
        lc_func.display(self.lcs, [spline],style="cosmograil_dr1", filename=os.path.join(self.outpath, 'spline_wi_ml6.png'))
        lc_func.display(self.lcs, [spline],style="cosmograil_dr1_microlensing", filename=os.path.join(self.outpath, 'spline_wi_ml7.png'))
        delays_th = [-6.44376707514413, -26.199323016152675, -70.92455333399347, -19.755555941008545,
                     -64.48078625884935, -44.7252303178408]

        assert_allclose(delays, delays_th, atol=0.2)

    def test_fluxshift(self):
        print("Fluxshift test")
        lc_copy = [lc.copy() for lc in self.lcs]
        shifts = [10, 20, 30, 40]
        min_flux = lc_copy[0].getminfluxshift()
        for i,lc in enumerate(lc_copy) :
            lc.resetml()
            lc.resetshifts()
            lc.setfluxshift(shifts[i], consmag=True)
            print(lc.magshift)

        lc_func.shuffle(lc_copy)
        lcs_sorted = lc_func.objsort(lc_copy, ret = True)
        lc_func.objsort(lc_copy, ret=False)



    def test_timeshifts(self):
        lc_copy = [lc.copy() for lc in self.lcs]
        lc_copy2 = [lc.copy() for lc in self.lcs]
        shifts = np.asarray([0., 10., 20., 30.])
        lc_func.settimeshifts(lc_copy, shifts, includefirst=True)
        lc_func.settimeshifts(lc_copy2, shifts[1:4], includefirst=False)
        test_shift1 = lc_func.gettimeshifts(lc_copy,  includefirst=True)
        test_shift2 = lc_func.gettimeshifts(lc_copy2, includefirst=False)
        assert_array_equal(test_shift1, shifts)
        assert_array_equal(test_shift2, shifts[1:4])


if __name__ == '__main__':
    pytest.main()
