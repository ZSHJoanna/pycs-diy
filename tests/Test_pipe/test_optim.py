import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import pytest
import unittest

from tests import TEST_PATH
import pycs3.gen.polyml
import pycs3.gen.splml
import pycs3.sim.draw
import pycs3.pipe.optimiser


def attachml(lcs, mlknotstep):
    for lc in lcs:
        mlbokeps_ad = mlknotstep / 3.0  # maybe change this
        pycs3.gen.splml.addtolc(lc, knotstep=mlknotstep, bokeps=mlbokeps_ad)


class TestOptim(unittest.TestCase):
    def setUp(self):
        self.path = TEST_PATH
        self.outpath = os.path.join(self.path, "output")
        self.datapath = os.path.join(self.path, "data")
        self.lcs, self.spline = pycs3.gen.util.readpickle(os.path.join(self.datapath, "optcurves.pkl"))

    def tearDown(self):
        plt.close('all')

    def testoptim_PS_residuals(self):
        pycs3.sim.draw.saveresiduals(self.lcs, self.spline)
        fit_vector = pycs3.pipe.optimiser.get_fit_vector(self.lcs, self.spline)  # we get the target parameter now
        mlknotstep = 200
        theta_init = [[0.1625], [0.1625], [0.2000], [0.2750]]

        # check if the microlensing exists :
        for k, l in enumerate(self.lcs):
            if l.ml == None:
                print(
                    'I dont have ml, I have to introduce minimal extrinsic variation to generate the mocks. Otherwise I have nothing to modulate.')
                pycs3.gen.splml.addtolc(l, n=2)

        dic_opt = pycs3.pipe.optimiser.DicOptimiser(self.lcs, fit_vector, self.spline, attachml, mlknotstep,
                                                    knotstep=20,
                                                    savedirectory=self.outpath,
                                                    recompute_spline=True, max_core=None,
                                                    n_curve_stat=4,
                                                    shotnoise=None, tweakml_type='PS_from_residuals',
                                                    tweakml_name='PS', display=False,
                                                    verbose=True,
                                                    correction_PS_residuals=True, max_iter=5,
                                                    tolerance=1,
                                                    theta_init=theta_init, debug=True)

        chain = dic_opt.optimise()
        dic_opt.analyse_plot_results()
        chi2, B_best = dic_opt.get_best_param()
        A = dic_opt.A_correction
        dic_opt.reset_report()
        dic_opt.report()

        print('Best fit', B_best)
        print('A correction', A)

        assert chi2 < 13


if __name__ == '__main__':
    pytest.main()
