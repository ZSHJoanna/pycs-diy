import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import pytest
import unittest

from tests import TEST_PATH
import pycs3.gen.mrg as mrg
import pycs3.gen.stat as stat
import pycs3.gen.lc_func as lc_func
import pycs3.gen.polyml
import pycs3.gen.splml
from tests import utils
from numpy.testing import assert_almost_equal

class TestStat(unittest.TestCase):
    def setUp(self):
        self.path = TEST_PATH
        self.outpath = os.path.join(self.path, "output")
        self.rdbfile_ECAM = os.path.join(self.path, "data", "DES0408_ECAM.rdb")
        self.rdbfile_WFI = os.path.join(self.path, "data", "DES0408_WFI.rdb")

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
        mrg.colourise(self.lcs_ECAM)
        mrg.colourise(self.lcs_WFI)
        self.lcs, self.spline = pycs3.gen.util.readpickle(os.path.join(self.path, "data", "optcurves.pkl"))

    def tearDown(self):
        plt.close('all')

    def test_structure_function(self):
        stat.sf(self.lcs_WFI[0])
        stat.sf(self.lcs_WFI[0],ssf=True)

    def test_mad(self):
        mags = self.lcs_WFI[0].mags
        mad1 = stat.mad(mags)
        assert_almost_equal(mad1,0.11140650000000107)


    def test_residuals(self):
        wfi_copy = [lc.copy() for lc in self.lcs_WFI]
        lc_func.settimeshifts(wfi_copy, shifts=[0, -112, -155,], includefirst=True)  # intial guess
        for lc in wfi_copy :
            pycs3.gen.polyml.addtolc(lc,  nparams=3, autoseasonsgap=60.0)

        spline = utils.spl(wfi_copy)
        rls = pycs3.gen.stat.subtract(wfi_copy, spline)
        pycs3.gen.stat.plotresiduals([rls],filename=os.path.join(self.outpath, "residuals_0408_WFI.png"))

        resistat = stat.mapresistats(rls)
        resistat_th = [{'mean': 0.0008286510218475188, 'std': 0.007764122554957176, 'zruns': -0.5794322853835916, 'pruns': 0.5622974446166266, 'nruns': 46}, {'mean': 0.0003049159858952155, 'std': 0.004410927637093387, 'zruns': -1.6274111536712763, 'pruns': 0.10364978682168813, 'nruns': 41}, {'mean': -0.00378797569187167, 'std': 0.010562779490206786, 'zruns': -1.0222084377857539, 'pruns': 0.30668229473751496, 'nruns': 44}]

        print(resistat)
        for i,r in enumerate(resistat) :
            for key in r.keys():
                assert_almost_equal(r[key], resistat_th[i][key], decimal=4)

        stats = pycs3.gen.stat.runstest(rls[0].mags, autolevel=True, verbose=True)

    def test_dof(self):
        rls = pycs3.gen.stat.subtract(self.lcs, self.spline)
        chi2_ml_dof = pycs3.gen.stat.compute_chi2(rls, 20, 200)
        chi2_noml_dof = pycs3.gen.stat.compute_chi2(rls, 20, 0)
        print(chi2_ml_dof, chi2_noml_dof)


if __name__ == '__main__':
    pytest.main()