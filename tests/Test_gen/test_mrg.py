import os
import numpy as np
import pytest
import unittest

from tests import TEST_PATH
import pycs3.gen.polyml
import pycs3.gen.splml
import pycs3.disp.disps
import pycs3.gen.mrg as mrg
import pycs3.gen.lc_func as lc_func
from numpy.testing import assert_allclose, assert_almost_equal, assert_array_equal


class TestMrg(unittest.TestCase):
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

    def test_merge(self):
        n_ecam =  len(self.lcs_ECAM[0].jds)
        n_wfi =  len(self.lcs_WFI[0].jds)
        print("Datapoints ECAM : ",n_ecam)
        print("Datapoints WFI: ", n_wfi)
        lc_func.display(self.lcs_ECAM, style="homepagepdf",
                        filename=os.path.join(self.outpath, 'merged_0408_ECAM.png'), jdrange=[57550, 57900])
        lc_func.display(self.lcs_WFI, style="homepagepdf",
                        filename=os.path.join(self.outpath, 'merged_0408_WFI.png'), jdrange=[57550, 57900])
        pycs3.gen.mrg.matchtels(self.lcs_WFI, self.lcs_ECAM, pycs3.disp.disps.linintnp, fluxshifts=True)
        merged_lcs = pycs3.gen.mrg.merge([self.lcs_WFI, self.lcs_ECAM])
        n_merged = len(merged_lcs[0].jds)
        print("Datapoints WFI+ECAM: ", n_merged)
        assert n_merged == n_wfi + n_ecam
        lc_func.display(merged_lcs, style="homepagepdf",filename=os.path.join(self.outpath, 'merged_0408_ECAM-WFI.png'), jdrange=[57550, 57900])

if __name__ == '__main__':
    pytest.main()