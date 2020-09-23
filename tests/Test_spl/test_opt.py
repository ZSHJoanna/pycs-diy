import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import glob
from tests import TEST_PATH
import pycs3.gen.lc_func as lc_func
import pycs3.gen.mrg as mrg
import pycs3.spl.multiopt
import pycs3.spl.topopt
import pycs3.gen.splml
from pycs3.sim.draw import shareflux
import pytest
import unittest
import numpy as np
from numpy.testing import assert_allclose
from tests import utils


class TestOpt(unittest.TestCase):
    def setUp(self):
        self.path = TEST_PATH
        self.outpath = os.path.join(self.path, "output")
        self.rdbfile_ECAM = os.path.join(self.path, "data", "DES0408_ECAM.rdb")
        self.rdbfile_WFI = os.path.join(self.path, "data", "DES0408_WFI.rdb")
        self.rdbfile = os.path.join(self.path, "data", "trialcurves.txt")
        self.true_delays = [-5.0, -20.0, -70., -15., -65., -50.]
        self.guess_timeshifts = [0., 0., -15., -65.]

        self.lcs_ECAM = [
            lc_func.rdbimport(self.rdbfile_ECAM, object='A', magcolname='mag_A', magerrcolname='magerr_A_5',
                              telescopename="ECAM"),
            lc_func.rdbimport(self.rdbfile_ECAM, object='B', magcolname='mag_B', magerrcolname='magerr_B_5',
                              telescopename="ECAM"),
            lc_func.rdbimport(self.rdbfile_ECAM, object='D', magcolname='mag_D', magerrcolname='magerr_D_5',
                              telescopename="ECAM")
        ]
        self.lcs_WFI = [
            lc_func.rdbimport(self.rdbfile_WFI, object='A', magcolname='mag_A', magerrcolname='magerr_A_5',
                              telescopename="WFI"),
            lc_func.rdbimport(self.rdbfile_WFI, object='B', magcolname='mag_B', magerrcolname='magerr_B_5',
                              telescopename="WFI"),
            lc_func.rdbimport(self.rdbfile_WFI, object='D', magcolname='mag_D', magerrcolname='magerr_D_5',
                              telescopename="WFI")
        ]

        self.lcs, self.spline = pycs3.gen.util.readpickle(os.path.join(self.path, 'data', "optcurves.pkl"))

        mrg.colourise(self.lcs_ECAM)
        mrg.colourise(self.lcs)
        for (l, c) in zip(self.lcs_WFI, ['orange','cyan','lightgreen']):
            l.plotcolour = c

    def tearDown(self):
        plt.close('all')

    def test_opt_magshift(self):

        lc_copy_ECAM = [lc.copy() for lc in self.lcs_ECAM]
        lc_copy_WFI = [lc.copy() for lc in self.lcs_WFI]
        self.clean_trace()
        pycs3.spl.multiopt.opt_magshift([lc_copy_ECAM[0],lc_copy_WFI[0]], sourcespline=None, verbose=True, trace=True,tracedir=self.outpath)
        pycs3.spl.multiopt.opt_magshift([lc_copy_ECAM[1],lc_copy_WFI[1]], sourcespline=None, verbose=True, trace=False)
        pycs3.spl.multiopt.opt_magshift([lc_copy_ECAM[2],lc_copy_WFI[2]], sourcespline=None, verbose=True, trace=False)
        magshift_ECAM = [lc.magshift for lc in lc_copy_ECAM]
        magshift_WFI = [lc.magshift for lc in lc_copy_WFI]
        mag_shift_th = [1.5689610000000016, 1.5397070000000017, 1.5588184999999992]

        lc_func.display(lc_copy_ECAM + lc_copy_WFI, [], showlegend=False, filename=os.path.join(self.outpath, 'ECAM+WFI_magshift.png'))
        assert_allclose(magshift_WFI, mag_shift_th, atol=0.01)

    def test_opt_fluxshift(self):
        lc_copy_ECAM = [lc.copy() for lc in self.lcs_ECAM]
        lc_copy_WFI = [lc.copy() for lc in self.lcs_WFI]
        pycs3.gen.lc_func.applyshifts(lc_copy_ECAM, [0 for lc in lc_copy_ECAM], [-np.median(lc.getmags()) for lc in lc_copy_ECAM])
        pycs3.gen.lc_func.applyshifts(lc_copy_WFI, [0 for lc in lc_copy_WFI], [-np.median(lc.getmags()) for lc in lc_copy_WFI])
        spline0 = utils.spl([lc_copy_WFI[0]])
        spline1 = utils.spl([lc_copy_WFI[1]])
        spline2 = utils.spl([lc_copy_WFI[2]])
        pycs3.spl.multiopt.opt_fluxshift([lc_copy_ECAM[0],lc_copy_WFI[0]], sourcespline=spline0, verbose=True)
        pycs3.spl.multiopt.opt_fluxshift([lc_copy_ECAM[1],lc_copy_WFI[1]], sourcespline=spline1, verbose=True)
        pycs3.spl.multiopt.opt_fluxshift([lc_copy_ECAM[2],lc_copy_WFI[2]], sourcespline=spline2, verbose=True)
        fluxshift_ECAM = [lc.fluxshift for lc in lc_copy_ECAM]
        fluxshift_WFI = [lc.fluxshift for lc in lc_copy_WFI]
        print(fluxshift_WFI)
        flux_shift_th = [237.2567336258173, 737.9918915027724, 3046.6033420598515]

        lc_func.display(lc_copy_ECAM + lc_copy_WFI, [], showlegend=False, filename=os.path.join(self.outpath, 'ECAM+WFI_fluxshift.png'))
        assert_allclose(fluxshift_WFI, flux_shift_th, rtol=0.01)

    def test_opt_ts_indi(self):
        lc_copy = [lc.copy() for lc in self.lcs]
        spline = self.spline.copy()
        pycs3.spl.multiopt.opt_ts_indi(lc_copy, spline, method='fmin', crit="r2", verbose=True, optml=True)
        delays = lc_func.getdelays(lc_copy, to_be_sorted=True)

        lc_copy2 = [lc.copy() for lc in self.lcs]
        spline2 = self.spline.copy()
        pycs3.spl.multiopt.opt_ts_indi(lc_copy2, spline2, method='fmin', crit="tv", verbose=True, optml=True)
        delays2 = lc_func.getdelays(lc_copy2, to_be_sorted=True)
        assert_allclose(delays, self.true_delays, atol=3)
        assert_allclose(delays2, self.true_delays, atol=3)

    def test_opt_source_magshift(self):
        lc_copy = [lc.copy() for lc in self.lcs]
        self.clean_trace()
        pycs3.spl.multiopt.opt_magshift(lc_copy, sourcespline=self.spline, verbose=True, trace=True, tracedir=self.outpath)
        magshift = [lc.magshift for lc in lc_copy]
        magshift_th = [12.625243648475646, 11.979070915466306, 11.020983992368699, 10.532716687860487]
        lc_func.display(lc_copy, [self.spline], showlegend=False, filename=os.path.join(self.outpath, 'trial_opt_magshift_source.png'))
        assert_allclose(magshift, magshift_th, atol=0.01)

    def test_opt_source(self):
        lc_copy = [lc.copy() for lc in self.lcs]
        self.clean_trace()
        r2 = pycs3.spl.multiopt.opt_source(lc_copy, sourcespline=self.spline, dpmethod="extadj", bokmethod="BF", verbose=True,trace=True,tracedir=self.outpath)
        assert r2 < 3000

    def test_distrib_flux(self):
        lc_copy = [lc.copy() for lc in self.lcs[:2]] #we take only 2 curves to test flux sharing
        shareflux(lc_copy[0], lc_copy[1], frac=0.1)
        lc_func.display(lc_copy, [], showlegend=False,
                        filename=os.path.join(self.outpath, 'trial_opt_distrib_flux_before_optim.png'))
        spline = self.spline.copy()
        pycs3.spl.topopt.opt_fine(lc_copy, spline, distribflux=True)
        delays = lc_func.getdelays(lc_copy, to_be_sorted=True)
        print(delays)
        lc_func.display(lc_copy, [spline], showlegend=False,
                        filename=os.path.join(self.outpath, 'trial_opt_distrib_flux_after_optim.png'))
        assert_allclose(delays, self.true_delays[0], atol=4)

    def clean_trace(self):
        pkls = glob.glob(os.path.join(self.outpath,  "??????.pkl"))
        for pkl in pkls :
            os.remove(pkl)


if __name__ == '__main__':
    pytest.main()
