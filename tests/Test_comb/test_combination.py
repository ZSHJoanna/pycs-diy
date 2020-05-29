import os
import pytest
import unittest
import copy
from tests import TEST_PATH
import numpy as np
import pycs3.mltd.comb
import pycs3.mltd.plot
from numpy.testing import assert_allclose
import pickle as pkl


class TestComb(unittest.TestCase):
    def setUp(self):
        self.path = TEST_PATH
        self.outpath = os.path.join(self.path, "output")
        self.datapath = os.path.join(self.path, "data")
        self.CS_spline = pycs3.mltd.comb.CScontainer("Test Spline", knots="20", ml="spl-150", colour="blue",
                                                     result_file_delays=os.path.join(self.path, 'data',
                                                                                     "sims_copies_opt_spl_delays.pkl"),
                                                     result_file_errorbars=os.path.join(self.path, "data",
                                                                                        "sims_mocks_opt_spl_errorbars.pkl"))
        self.CS_regdiff = pycs3.mltd.comb.CScontainer("Test Regdiff", colour="red",
                                                      result_file_delays=os.path.join(self.path, 'data',
                                                                                      "sims_copies_opt_regdiff_delays.pkl"),
                                                      result_file_errorbars=os.path.join(self.path, "data",
                                                                                         "sims_mocks_opt_regdiff_errorbars.pkl"))
        self.text = [
            (0.85, 0.90, r"Test" + "\n" + r"$\mathrm{PyCS\ estimates}$",
             {"fontsize": 26, "horizontalalignment": "center"})]

    def test_groups(self):
        groups = [pycs3.mltd.comb.getresults(self.CS_spline, useintrinsic=False),
                  pycs3.mltd.comb.getresults(self.CS_regdiff, useintrinsic=False)]

        pycs3.mltd.plot.delayplot(groups, rplot=6.0, displaytext=True, text=self.text,
                                  filename=os.path.join(self.outpath, "fig_delays.png"), figsize=(15, 10),
                                  hidedetails=False, showbias=False, showran=False, showlegend=True,
                                  auto_radius=True, horizontaldisplay=False, legendfromrefgroup=False,
                                  tick_step_auto=True)

    def test_combine(self):
        groups = [pycs3.mltd.comb.getresults(self.CS_spline, useintrinsic=False),
                  pycs3.mltd.comb.getresults(self.CS_regdiff, useintrinsic=False)]

        medians_list = np.asarray([gr.medians for gr in groups])
        errors_down_list = np.asarray([gr.errors_down for gr in groups])
        errors_up_list = np.asarray([gr.errors_up for gr in groups])
        binslist = []

        for i, lab in enumerate(groups[0].labels):
            bins = np.linspace(min(medians_list[:, i]) - 10 * min(errors_down_list[:, i]),
                               max(medians_list[:, i]) + 10 * max(errors_up_list[:, i]), 500)
            binslist.append(bins)

        for g, group in enumerate(groups):
            group.binslist = binslist
            group.linearize(testmode=True)

        combined = copy.deepcopy(pycs3.mltd.comb.combine_estimates(groups, sigmathresh=0., testmode=True))
        combined.linearize(testmode=True)
        combined.name = 'PyCS-Sum'
        combined.nicename = 'PyCS-Sum'
        combined.plotcolor = 'black'

        mult = pycs3.mltd.comb.mult_estimates(groups)
        mult.name = "PyCS-Mult"
        mult.nicename = "PyCS-Mult"
        mult.plotcolor = "gray"

        print("Final combination for marginalisation :")
        combined.niceprint()

        medians_th = [-4.786234278675292, -21.205331381025204, -16.5855087451734, -70.11602724895087,
                      -65.13316024716289, -48.84628785720808]
        errors_down_th = [1.8080595799043957, 2.281292027345117, 3.100649051325565, 2.142508184263548,
                          2.259956445125397, 2.430396705443627]
        errors_up_th = [1.6689780737579039, 2.3725437084389185, 3.224675013378583, 2.2282085116340937,
                        1.7923792495822184, 2.340382012649421]
        assert_allclose(combined.medians, medians_th, atol=0.2)
        assert_allclose(combined.errors_up, errors_up_th, atol=0.2)
        assert_allclose(combined.errors_down, errors_down_th, atol=0.2)

        pycs3.mltd.plot.delayplot(groups + [combined] + [mult], rplot=6.0, refgroup=combined, displaytext=True,
                                  text=self.text,
                                  filename=os.path.join(self.outpath, "fig_delays_comb.png"), figsize=(15, 10),
                                  hidedetails=False, showbias=False, showran=False, showlegend=True,
                                  auto_radius=True, horizontaldisplay=False, legendfromrefgroup=False,
                                  tick_step_auto=True)

        pycs3.mltd.plot.delayplot(groups, rplot=6.0, refgroup=combined, displaytext=True, text=self.text,
                                  filename=os.path.join(self.outpath, "fig_delays_comb_blind.png"), figsize=(15, 10),
                                  hidedetails=False, showbias=True, showran=False, showlegend=True,
                                  auto_radius=True, horizontaldisplay=True, legendfromrefgroup=False,
                                  tick_step_auto=True, blindness=True)

        pycs3.mltd.plot.write_delays(combined, write_dir=self.outpath)

    def test_combine_from_pkl(self):
        path_list = [os.path.join(self.datapath, 'marginalisation_regdiff.pkl'),
                     os.path.join(self.datapath, 'marginalisation_spline.pkl')]

        group_list, combined = pycs3.mltd.comb.group_estimate(path_list, name_list=['regdiff', 'spline'], colors=['red', 'blue'],
                                       sigma_thresh=0, new_name_marg='Spline + regdiff', testmode=True)

        print("Final combination for marginalisation :")
        combined.niceprint()
        medians_th = [-11.5]
        error_up_th = [0.9]
        error_down_th = [0.9]

        assert_allclose(combined.medians, medians_th, atol=0.2)
        assert_allclose(combined.errors_up, error_up_th, atol=0.2)
        assert_allclose(combined.errors_down, error_down_th, atol=0.2)

if __name__ == '__main__':
    pytest.main()
