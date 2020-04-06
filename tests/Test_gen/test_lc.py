import os
import numpy as np
import pytest
import unittest
import glob

from tests import TEST_PATH
import pycs3.gen.polyml
import pycs3.gen.splml
from tests import utils
import pycs3.gen.mrg as mrg
import pycs3.gen.lc_func as lc_func
from numpy.testing import assert_allclose, assert_almost_equal, assert_array_equal

class TestLightCurve(unittest.TestCase):
    def setUp(self):
        self.path = TEST_PATH
        self.outpath = os.path.join(self.path, "output")
        self.rdbfile = os.path.join(self.path, "data", "trialcurves.txt")
        self.skiplist = os.path.join(self.path, "data", "skiplist.txt")
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


    def test_opt_spline_polyml(self):
        lc_copy =[lc.copy() for lc in self.lcs]
        pycs3.gen.polyml.addtolc(lc_copy[1], nparams=2, autoseasonsgap=60.0)  # add affine microlensing to each season
        pycs3.gen.polyml.addtolc(lc_copy[2], nparams=3, autoseasonsgap=600.0)  # add polynomial of degree 2 on the entire light curve
        pycs3.gen.polyml.addtolc(lc_copy[3], nparams=3, autoseasonsgap=600.0)
        spline = utils.spl(lc_copy)
        delays = lc_func.getdelays(lc_copy, to_be_sorted=True)
        lc_func.getnicetimedelays(lc_copy)
        lc_func.display(lc_copy, [spline],style="homepagepdf", filename=os.path.join(self.outpath, 'spline_wi_ml1.png'))
        lc_func.display(lc_copy, [spline],style="homepagepdfnologo", filename=os.path.join(self.outpath, 'spline_wi_ml2.png'))
        lc_func.display(lc_copy, [spline],style="2m2", filename=os.path.join(self.outpath, 'spline_wi_ml3.png'))
        lc_func.display(lc_copy, [spline],style="posterpdf", filename=os.path.join(self.outpath, 'spline_wi_ml4.png'))
        lc_func.display(lc_copy, [spline],style="internal", filename=os.path.join(self.outpath, 'spline_wi_ml5.png'))
        lc_func.display(lc_copy, [spline],style="cosmograil_dr1", filename=os.path.join(self.outpath, 'spline_wi_ml6.png'))
        lc_func.display(lc_copy, [spline],style="cosmograil_dr1_microlensing", filename=os.path.join(self.outpath, 'spline_wi_ml7.png'))
        delays_th = [-6.44376707514413, -26.199323016152675, -70.92455333399347, -19.755555941008545,
                     -64.48078625884935, -44.7252303178408]

        assert_allclose(delays, delays_th, atol=0.2)

        #play a bit with the microlensing object:
        microlensing = lc_copy[2].ml
        microlensing.printinfo()
        stat = microlensing.stats(lc_copy[2])
        stat_th = {'mean': -0.17782270209855014, 'std': 0.2211515237812574}
        for key in stat.keys():
            assert_almost_equal(stat[key], stat_th[key], decimal=2)

        #play a bit with the params.
        params = pycs3.gen.polyml.multigetfreeparams(lc_copy)
        pycs3.gen.polyml.multisetfreeparams(lc_copy, params)
        microlensing.reset()

    def test_opt_spline_splml(self):
        lc_copy = [lc.copy() for lc in self.lcs]
        mlknotstep = 200
        mlbokeps_ad = mlknotstep / 3.0  # maybe change this
        pycs3.gen.splml.addtolc(lc_copy[1], knotstep=mlknotstep, bokeps=mlbokeps_ad)
        pycs3.gen.splml.addtolc(lc_copy[2], knotstep=mlknotstep, bokeps=mlbokeps_ad)
        pycs3.gen.splml.addtolc(lc_copy[3], knotstep=mlknotstep, bokeps=mlbokeps_ad)
        spline = utils.spl(lc_copy)
        tv, dist = pycs3.gen.spl_func.mltv(lc_copy, spline) # some metric of the fit
        assert tv < 900
        assert dist<5710

        delays = lc_func.getdelays(lc_copy, to_be_sorted=True)
        print(delays)
        delays_th = [-5.93916576886539, -20.7304134782579, -31.155687183119934, -14.791247709392511, -25.216521414254544, -10.425273704862033]
        assert_allclose(delays, delays_th, atol=0.2)
        lc_func.display(lc_copy, [spline], style="homepagepdf",filename=os.path.join(self.outpath, 'spline_wi_splml.png'))

        #trace function :
        self.clean_trace()
        pycs3.gen.util.trace(lc_copy, spline, tracedir=self.outpath)
        pycs3.gen.util.plottrace(tracedir=self.outpath)

    def test_fluxshift(self):
        shifts = [-0.1, -0.2, -0.3, -0.4]
        lc_copy = [lc.copy() for lc in self.lcs]
        lc0 = lc_copy[0].copy()
        lc0.getrawfluxes()
        min_flux = lc_copy[0].getminfluxshift()
        assert_almost_equal(min_flux, -64108.55676508218)

        for i,lc in enumerate(lc_copy) :
            lc.resetml()
            lc.resetshifts()
            lc.setfluxshift(shifts[i], consmag=True)

        magshifts = [lc.magshift for lc in lc_copy]
        assert_allclose(magshifts, [-9.467925255575702e-07, -3.6197204037074295e-06,-1.3700547842369769e-05,-3.143986495236058e-05], rtol=1e-3)

        new_mags = lc_copy[0].getmags()
        assert_almost_equal(new_mags[0], -12.38173974)

        fluxvector = shifts[0] * np.ones(len(lc0.mags))
        lc0.addfluxes(fluxvector)
        assert_allclose(lc0.getmags(),lc_copy[0].getmags(), rtol=0.0000001)
        lc_copy[0].calcfluxshiftmags(inverse = True)

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


    def test_mask(self):
        lc_copy = [lc.copy() for lc in self.lcs]
        for i,lc in enumerate(lc_copy) :
            lc.maskskiplist(self.skiplist,searchrange=4, accept_multiple_matches=True, verbose=True)

        print(lc_copy[0].maskinfo())
        assert np.sum(lc_copy[0].mask == False) == 14
        assert lc_copy[0].hasmask()
        lc_copy[0].cutmask()
        assert not lc_copy[0].hasmask()

        lc_copy[1].clearmask()
        assert not lc_copy[1].hasmask()

        lc_copy[1].pseudobootstrap()
        print("Bootstrap masked %i datapoints" % np.sum(lc_copy[1].mask == False))


    def test_montecarlo(self):
        lc0 = self.lcs[0].copy()
        lc0_copy = self.lcs[0].copy()
        pycs3.gen.polyml.addtolc(lc0, nparams=2, autoseasonsgap=60.0)
        lc0.montecarlomags()
        lc0.montecarlojds(amplitude=0.5, seed=1, keepml=True)

        lc0_copy.merge(lc0)
        lc0.rdbexport(filename=os.path.join(self.outpath, "merged_A+A.txt"))

    def clean_trace(self):
        pkls = glob.glob(os.path.join(self.outpath,  "??????.pkl"))
        for pkl in pkls :
            os.remove(pkl)


if __name__ == '__main__':
    pytest.main()
