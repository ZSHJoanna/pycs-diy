import os
import glob
from tests import TEST_PATH
import pycs3.gen.lc_func as lc_func
import pycs3.gen.mrg as mrg
import pytest
import unittest
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal, assert_array_equal
from tests import utils

class TestLightCurve(unittest.TestCase):
    def setUp(self):
        self.path = TEST_PATH
        self.outpath = os.path.join(self.path, "output")
        self.rdbfile_ECAM = os.path.join(self.path, "data", "DES0408_ECAM.rdb")
        self.rdbfile_WFI = os.path.join(self.path, "data", "DES0408_WFI.rdb")
        self.rdbfile = os.path.join(self.path, "data", "trialcurves.txt")


if __name__ == '__main__':
    pytest.main()