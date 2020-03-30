import os
import pytest
import unittest
from tests import TEST_PATH
import pycs3.gen.lc_func as lc_func
import pycs3.gen.mrg as mrg
import pycs3.mltd.comb
import pycs3.mltd.plot


class TestComb(unittest.TestCase):
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

    def test_groups(self):
        CS_spline = pycs3.mltd.comb.CScontainer("Test Spline", knots="20", ml="spl-150", colour="blue",
                                                result_file_delays=os.path.join(self.path, 'data',
                                                                                "sims_copies_opt_spl_delays.pkl"),
                                                result_file_errorbars=os.path.join(self.path, "data",
                                                                                   "sims_mocks_opt_spl_errorbars.pkl"))
        CS_regdiff = pycs3.mltd.comb.CScontainer("Test Regdiff", colour="red",
                                                 result_file_delays=os.path.join(self.path, 'data',
                                                                                 "sims_copies_opt_regdiff_delays.pkl"),
                                                 result_file_errorbars=os.path.join(self.path, "data",
                                                                                    "sims_mocks_opt_regdiff_errorbars.pkl"))

        groups = [pycs3.mltd.comb.getresults(CS_spline, useintrinsic=False),
                  pycs3.mltd.comb.getresults(CS_regdiff, useintrinsic=False)]

        pycs3.mltd.plot.delayplot(groups, rplot=6.0, displaytext=True,
                                  filename=os.path.join(self.outpath, "fig_delays_comb.png"), figsize=(15, 10),
                                  hidedetails=False, showbias=False, showran=False, showlegend=True,
                                  auto_radius=True, horizontaldisplay=False, legendfromrefgroup=False,
                                  tick_step_auto=True )

        if __name__ == '__main__':
            pytest.main()
