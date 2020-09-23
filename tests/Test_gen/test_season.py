import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import pytest
import unittest

from tests import TEST_PATH
import pycs3.gen.polyml
import pycs3.gen.splml
import pycs3.gen.spl_func
import pycs3.gen.sea
import pycs3.gen.mrg as mrg
import pycs3.gen.lc_func as lc_func

class TestSeason(unittest.TestCase):
    def setUp(self):
        self.path = TEST_PATH
        self.outpath = os.path.join(self.path, "output")
        self.rdbfile = os.path.join(self.path, "data", "trialcurves.txt")
        self.skiplist = os.path.join(self.path, "data", "skiplist.txt")
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

    def test_season(self):
        lc_copy = [lc.copy() for lc in self.lcs]
        lc_copy2 = [lc.copy() for lc in self.lcs]
        ndata = len(lc_copy[0].jds)
        knots = pycs3.gen.spl_func.seasonknots(lc_copy, knotstep=10, ingap=1)
        print("Knots : ", knots)

        # create the Season object:
        season = pycs3.gen.sea.autofactory(lc_copy[0], tpe='seas')
        pycs3.gen.sea.printinfo(season)
        season_copy = season.copy()
        interseason = pycs3.gen.sea.autofactory(lc_copy[0], tpe='interseasons')
        pycs3.gen.sea.printinfo(season)

        #remove some seasons:
        pycs3.gen.sea.easycut(lc_copy, keep=[1,2,3], mask=False, verbose=True)
        pycs3.gen.sea.easycut(lc_copy2, keep=[1,2,3], mask=True, verbose=True)
        ndata_cut = len(lc_copy[0].jds)
        ndata_mask = len(lc_copy2[0].jds)
        assert ndata_cut == 133
        assert ndata_mask == ndata

        #manually create season object
        seas1 = pycs3.gen.sea.manfactory(lc_copy[0], [[54000, 54300], [54400, 54600]])


if __name__ == '__main__':
    pytest.main()
