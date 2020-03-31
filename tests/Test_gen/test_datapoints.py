import os
import pytest
import unittest
from tests import TEST_PATH
import pycs3.gen.lc_func as lc_func
import pycs3.gen.mrg as mrg
from pycs3.gen.datapoints import DataPoints


class TestDatapoints(unittest.TestCase):
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

    def test_datapoints(self):
        datapts = DataPoints(self.lcs[0].jds, self.lcs[0].mags, self.lcs[0].magerrs, splitup=False, sort=True,
                             stab=True, stabext=300.0, stabgap=30.0, stabstep=5.0, stabmagerr=-2.0, stabrampsize=5.0,
                             stabrampfact=1.0)
        datapts.putstab()
        bounds = datapts.getmaskbounds()
        print(bounds)
        assert datapts.ntrue() == 192


if __name__ == '__main__':
    pytest.main()
