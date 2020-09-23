import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import pytest
import unittest

from tests import TEST_PATH
import pycs3.gen.polyml
import pycs3.gen.splml
import pycs3.gen.mrg as mrg
import pycs3.gen.lc_func as lc_func


class TestMrg(unittest.TestCase):
    def setUp(self):
        self.path = TEST_PATH
        self.outpath = os.path.join(self.path, "output")
        self.rdbfile_ECAM = os.path.join(self.path, "data", "DES0408_ECAM.rdb")
        self.rdbfile_WFI = os.path.join(self.path, "data", "DES0408_WFI.rdb")

        self.lcs_ECAM = [
            lc_func.rdbimport(self.rdbfile_ECAM, object='A', magcolname='mag_A', magerrcolname='magerr_A_5',
                              telescopename="ECAM",propertycolnames="lcmanip"),
            lc_func.rdbimport(self.rdbfile_ECAM, object='B', magcolname='mag_B', magerrcolname='magerr_B_5',
                              telescopename="ECAM",propertycolnames="lcmanip"),
            lc_func.rdbimport(self.rdbfile_ECAM, object='D', magcolname='mag_D', magerrcolname='magerr_D_5',
                              telescopename="ECAM",propertycolnames="lcmanip")
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

    def tearDown(self):
        plt.close('all')

    def test_merge(self):
        lcs_ECAM_copy = [lc.copy() for lc in self.lcs_ECAM]
        lcs_WFI_copy = [lc.copy() for lc in self.lcs_WFI]
        n_ecam =  len(lcs_ECAM_copy[0].jds)
        n_wfi =  len(lcs_WFI_copy[0].jds)
        print("Datapoints ECAM : ",n_ecam)
        print("Datapoints WFI: ", n_wfi)
        test_stat_WFI = lcs_WFI_copy[0].samplingstats()

        lc_func.display(lcs_ECAM_copy, style="homepagepdf",
                        filename=os.path.join(self.outpath, 'merged_0408_ECAM.png'), jdrange=[57550, 57900])
        lc_func.display(lcs_WFI_copy, style="homepagepdf",
                        filename=os.path.join(self.outpath, 'merged_0408_WFI.png'), jdrange=[57550, 57900])
        pycs3.gen.mrg.matchtels(lcs_WFI_copy, lcs_ECAM_copy, pycs3.gen.lc_func.linintnp, fluxshifts=True)
        merged_lcs = pycs3.gen.mrg.merge([lcs_WFI_copy, lcs_ECAM_copy])
        n_merged = len(merged_lcs[0].jds)
        print("Datapoints WFI+ECAM: ", n_merged)
        print(merged_lcs[0].longinfo())
        assert n_merged == n_wfi + n_ecam
        lc_func.display(merged_lcs, style="homepagepdf",filename=os.path.join(self.outpath, 'merged_0408_ECAM-WFI.png'), jdrange=[57550, 57900])

        lcs_ECAM_copy2 = [lc.copy() for lc in self.lcs_ECAM]
        lcs_WFI_copy2 = [lc.copy() for lc in self.lcs_WFI]
        pycs3.gen.mrg.matchtels(lcs_ECAM_copy2, lcs_WFI_copy2, pycs3.gen.lc_func.linintnp, fluxshifts=False)

    def test_export(self):
        pycs3.gen.util.multilcsexport(self.lcs_ECAM, os.path.join(self.outpath,"lcs_export.rdb"), properties=["fwhm", "ellipticity", "airmass"])

if __name__ == '__main__':
    pytest.main()
