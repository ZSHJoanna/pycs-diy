import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import pytest
import unittest
from tests import TEST_PATH
import pycs3.gen.lc_func as lc_func
import pycs3.gen.mrg as mrg
import pycs3.sim.draw
import pycs3.gen.splml
import pycs3.sim.run
import pycs3.sim.plot
import pycs3.tdcomb.plot
import pycs3.tdcomb.comb
import pycs3.gen.stat

from tests import utils
import shutil


class TestSims(unittest.TestCase):
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
        for lc in self.lcs:
            pycs3.gen.splml.addtolc(lc, knotstep=150)
        lc_func.settimeshifts(self.lcs, shifts=[0, -5, -20, -60], includefirst=True)  # intial guess
        self.spline = utils.spl(self.lcs)

    def tearDown(self):
        plt.close('all')

    def test_draw_run_sims(self):
        self.clear_sims()
        pycs3.sim.draw.saveresiduals(self.lcs, self.spline)
        pycs3.sim.draw.multidraw(self.lcs, self.spline, n=10, npkl=1, simset="mocks", destpath=self.outpath,
                                 truetsr=8.0, tweakml=[utils.Atweakml, utils.Btweakml, utils.Ctweakml, utils.Dtweakml])

        kwargs_optim = {}
        kwargs_optim_regdiff = {'pd': 2, 'covkernel': 'matern', 'pow': 1.5, 'amp': 1., 'scale': 200., 'errscale': 1.,
                                'verbose': True, 'method': "weights"}
        success_dic_spline = pycs3.sim.run.multirun("mocks", self.lcs, utils.spl, kwargs_optim, optset="spl",destpath=self.outpath,
                                                    tsrand=10.0, keepopt=True, use_test_seed=True)
        success_dic_regdiff = pycs3.sim.run.multirun("mocks", self.lcs, utils.regdiff, kwargs_optim_regdiff,destpath=self.outpath,
                                                     optset="regdiff", tsrand=10.0, keepopt=True, use_test_seed=True)
        assert success_dic_spline['success'] is True
        assert success_dic_regdiff['success'] is True

        simresults = [
            pycs3.sim.run.collect(directory=os.path.join(self.outpath, "sims_mocks_opt_spl"), plotcolour="blue",
                                  name="Free-knot spline technique"),
            pycs3.sim.run.collect(directory=os.path.join(self.outpath, "sims_mocks_opt_regdiff"),
                                  plotcolour="red",
                                  name="Regression difference technique")
            ]

        pycs3.sim.plot.measvstrue(simresults, errorrange=3.5, r=5.0, nbins=10, binclip=True, binclipr=20.0,
                                  plotpoints=True, filename=os.path.join(self.outpath, "fig_measvstrue.png"),
                                  dataout=True, outdir=self.outpath)
        pycs3.sim.plot.measvstrue(simresults, errorrange=3.5, r=5.0, nbins=10, binclip=True, binclipr=20.0,
                                  plotpoints=True, filename=os.path.join(self.outpath, "fig_measvstrue2.png"),
                                  dataout=True, outdir=self.outpath, tweakeddisplay=True, blindness=True)
        pycs3.sim.plot.newcovplot(simresults, filepath=self.outpath, showplots=False, printcovmat=True)
        pycs3.sim.plot.newcovplot(simresults, filepath=self.outpath, showplots=False, printcovmat=True, detailplots=True, method='depbin',
                                  printdetails=False)

        #test anaoptdrawn
        stat = pycs3.gen.stat.anaoptdrawn(self.lcs, self.spline, simset="mocks", optset="spl", npkl=1, plots=True, nplots=1, r=0.11,
                plotjdrange=None, plotcurveindexes=None, showplot=False, directory=self.outpath, plotpath=self.outpath, resihist_figsize=None)

    def test_sample(self):
        lc_copy = self.lcs[0]
        lc_copy.shiftflux(0.2)
        pycs3.sim.draw.sample(lc_copy, self.spline)

    def test_single_draw(self):
        pycs3.sim.draw.saveresiduals(self.lcs, self.spline)
        fakelcs = pycs3.sim.draw.draw(self.lcs, self.spline, shotnoise='magerrs')
        lc_func.display(fakelcs, [], filename=os.path.join(self.outpath, 'fakelcs1.png'))
        fakelcs = pycs3.sim.draw.draw(self.lcs, self.spline, shotnoise='res', keepshifts= True, keeporiginalml=False, keeptweakedml=True)
        lc_func.display(fakelcs, [], filename=os.path.join(self.outpath, 'fakelcs2.png'))
        fakelcs = pycs3.sim.draw.draw(self.lcs, self.spline, shotnoise='sigma', keepshifts= True, keeporiginalml=True, keeptweakedml=False)
        lc_func.display(fakelcs, [], filename=os.path.join(self.outpath, 'fakelcs3.png'))
        fakelcs = pycs3.sim.draw.draw(self.lcs, self.spline, shotnoise='mcres')
        lc_func.display(fakelcs, [], filename=os.path.join(self.outpath, 'fakelcs4.png'))

    def clear_sims(self):
        if os.path.exists(os.path.join(self.outpath, "sims_mocks_opt_spl")):
            shutil.rmtree(os.path.join(self.outpath, "sims_mocks_opt_spl"))
        if os.path.exists(os.path.join(self.outpath, "sims_mocks_opt_regdiff")):
            shutil.rmtree(os.path.join(self.outpath, "sims_mocks_opt_regdiff"))
        if os.path.exists(os.path.join(self.outpath, "sims_mocks")):
            shutil.rmtree(os.path.join(self.outpath, "sims_mocks"))


if __name__ == '__main__':
    pytest.main()
