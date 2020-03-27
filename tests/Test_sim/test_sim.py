import os
import pytest
import unittest
from tests import TEST_PATH
import pycs3.gen.lc_func as lc_func
import pycs3.gen.mrg as mrg
import pycs3.sim.draw
import pycs3.gen.util
import pycs3.gen.splml
import pycs3.sim.run
import pycs3.sim.plot
from tests import utils
import numpy as np
import shutil
from numpy.testing import assert_allclose

class TestSimuf(unittest.TestCase):
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

    def test_draw_run_copies(self):
        self.clear_sim()
        lc_copy = [lc.copy() for lc in self.lcs]
        #draw the copy :
        pycs3.sim.draw.multidraw(lc_copy, onlycopy=True, n=5, npkl=1, simset="copies", destpath=self.outpath)

        #Set the initial shift and microlensing model
        lc_func.settimeshifts(lc_copy, shifts=[0, -5, -20, -60], includefirst=True)  #intial guess
        for lc in lc_copy:
            pycs3.gen.splml.addtolc(lc, knotstep=150)
        kwargs_optim = {}
        success_dic = pycs3.sim.run.multirun("copies", lc_copy, utils.spl, kwargs_optim, optset="spl", tsrand=10.0, keepopt=True,destpath=self.outpath, use_test_seed=True)

        dataresults = [
            # pycs3.sim.run.collect("sims_copies_opt_disp", "red", "Dispersion-like technique"),
            # pycs3.sim.run.collect("sims_copies_opt_regdiff", "green", "Regression difference technique"),
            pycs3.sim.run.collect(directory=os.path.join(self.outpath, "sims_copies_opt_spl"), plotcolour="blue", name="Free-knot spline technique")
        ]
        result_dic = dataresults[0].getts()
        print(result_dic)
        print(success_dic)
        result_th_center =np.asarray([  -5.07276501,  -9.78323028, -25.77944057, -73.49879065])
        assert_allclose(result_dic['center'], result_th_center, atol=2.)
        pycs3.sim.plot.hists(dataresults, r=5.0, nbins=100, showqs=False,
                            filename=os.path.join(self.outpath,"fig_intrinsicvariance.png"), dataout=True)
    def clear_sim(self):
        if os.path.exists(os.path.join(self.outpath, "sims_copies_opt_spl")):
            shutil.rmtree(os.path.join(self.outpath, "sims_copies_opt_spl"))
        if os.path.exists(os.path.join(self.outpath, "sims_copies")):
            shutil.rmtree(os.path.join(self.outpath, "sims_copies"))

if __name__ == '__main__':
    pytest.main()