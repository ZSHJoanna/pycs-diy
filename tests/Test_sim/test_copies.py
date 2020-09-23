import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import shutil
import unittest

import numpy as np
import pytest
from numpy.testing import assert_allclose

import pycs3.gen.lc_func as lc_func
import pycs3.gen.mrg as mrg
import pycs3.gen.splml
import pycs3.gen.util
import pycs3.sim.draw
import pycs3.sim.plot
import pycs3.sim.run
from tests import TEST_PATH
from tests import utils


class TestCopies(unittest.TestCase):
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

    def tearDown(self):
        plt.close('all')

    def test_draw_run_copies(self):
        self.clear_copies()
        lc_copy = [lc.copy() for lc in self.lcs]
        lc_copy_disp = [lc.copy() for lc in self.lcs]
        lc_copy_regdiff = [lc.copy() for lc in self.lcs]
        # draw the copy :
        pycs3.sim.draw.multidraw(lc_copy, onlycopy=True, n=5, npkl=1, simset="copies", destpath=self.outpath)

        # Set the initial shift and microlensing model
        lc_func.settimeshifts(lc_copy, shifts=[0, -5, -20, -60], includefirst=True)  # intial guess
        lc_func.settimeshifts(lc_copy_disp, shifts=[0, -5, -20, -60], includefirst=True)  # intial guess
        lc_func.settimeshifts(lc_copy_regdiff, shifts=[0, -5, -20, -60], includefirst=True)  # intial guess
        for lc in lc_copy:
            pycs3.gen.splml.addtolc(lc, knotstep=150)
        for lc in lc_copy_disp:
            pycs3.gen.polyml.addtolc(lc, nparams=2, autoseasonsgap = 60.0)

        kwargs_optim = {}
        kwargs_optim_regdiff = {'pd': 2, 'covkernel': 'matern', 'pow': 1.5, 'amp': 1., 'scale': 200., 'errscale': 1.,
                                'verbose': True, 'method': "weights"}

        success_dic_spline = pycs3.sim.run.multirun("copies", lc_copy, utils.spl, kwargs_optim, optset="spl",
                                                    tsrand=10.0, keepopt=True, destpath=self.outpath,
                                                    use_test_seed=True)
        success_dic_regdiff = pycs3.sim.run.multirun("copies", lc_copy_regdiff, utils.regdiff, kwargs_optim_regdiff,
                                                     optset="regdiff",
                                                     tsrand=10.0, keepopt=True, destpath=self.outpath,
                                                     use_test_seed=True)
        assert success_dic_spline['success'] is True
        assert success_dic_regdiff['success'] is True


        dataresults = [
            pycs3.sim.run.collect(directory=os.path.join(self.outpath, "sims_copies_opt_regdiff"), plotcolour="red",
                                  name="Regression difference technique"),
            pycs3.sim.run.collect(directory=os.path.join(self.outpath, "sims_copies_opt_spl"), plotcolour="blue",
                                  name="Free-knot spline technique"),
        ]
        result_dic_regdiff = dataresults[0].get_delays_from_ts()
        result_dic_spline = dataresults[1].get_delays_from_ts()
        print("Spline : ", result_dic_spline)
        print("Regdiff : ", result_dic_regdiff)

        result_th = np.asarray([-5., -20., -70., -15., -65., -50. ])

        assert_allclose(result_dic_spline['center'], result_th, atol=3)
        assert_allclose(result_dic_regdiff['center'], result_th, atol=3)
        pycs3.sim.plot.hists(dataresults, r=5.0, nbins=100, showqs=False,
                             filename=os.path.join(self.outpath, "fig_intrinsicvariance.png"), dataout=True,outdir=self.outpath)
        pycs3.sim.plot.hists(dataresults, r=5.0, nbins=100, showqs=True, showallqs=True, niceplot= True, blindness=True,
                             dataout=False,outdir=self.outpath)


    def clear_copies(self):
        if os.path.exists(os.path.join(self.outpath, "sims_copies_opt_spl")):
            shutil.rmtree(os.path.join(self.outpath, "sims_copies_opt_spl"))
        if os.path.exists(os.path.join(self.outpath, "sims_copies_opt_regdiff")):
            shutil.rmtree(os.path.join(self.outpath, "sims_copies_opt_regdiff"))
        if os.path.exists(os.path.join(self.outpath, "sims_copies_opt_disp")):
            shutil.rmtree(os.path.join(self.outpath, "sims_copies_opt_disp"))
        if os.path.exists(os.path.join(self.outpath, "sims_copies")):
            shutil.rmtree(os.path.join(self.outpath, "sims_copies"))


if __name__ == '__main__':
    pytest.main()
