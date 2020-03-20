import os
import sys
import pytest
import unittest

from pycs3.gen.lc_func import rdbimport
from pycs3.gen.mrg import colourise
from pycs3.gen.util import writepickle


class TestLightCurve(unittest.TestCase):
    def setUp(self):
        self.path = os.path.dirname(os.path.realpath(sys.argv[0]))
        self.rdbfile = os.path.join(self.path, "tests", "data", "trialcurves.txt")
        self.lcs = [
            rdbimport(self.rdbfile, object='A', magcolname='mag_A', magerrcolname='magerr_A',
                      telescopename="Trial"),
            rdbimport(self.rdbfile, object='B', magcolname='mag_B', magerrcolname='magerr_B',
                      telescopename="Trial"),
            rdbimport(self.rdbfile, object='C', magcolname='mag_C', magerrcolname='magerr_C',
                      telescopename="Trial"),
            rdbimport(self.rdbfile, object='D', magcolname='mag_D', magerrcolname='magerr_D',
                      telescopename="Trial")
        ]

    def test_lc_infos(self):
        print(self.lcs[0])
        print(self.lcs[0].printinfo())
        stats = {'len': 192, 'nseas': 4, 'meansg': 163.99396333333425, 'minsg': 130.38524000000325, 'maxsg': 210.23221999999805, 'stdsg': 33.7985590219782, 'med': 3.989030000000639, 'mean': 4.38115308510637, 'max': 25.124900000002526, 'min': 0.8326200000010431, 'std': 3.460329376531179}
        test_stat = self.lcs[0].samplingstats()
        for key in test_stat.keys():
            self.assertAlmostEqual(stats[key],test_stat[key], places=3)



if __name__ == '__main__':
    pytest.main()
