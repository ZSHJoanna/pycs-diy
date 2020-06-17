import os
import glob
from tests import TEST_PATH

import pycs3.gen.util
import pycs3.sim.twk
import pytest
import unittest
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal, assert_array_equal
from tests import utils


class TestSource(unittest.TestCase):
    def setUp(self):
        self.path = TEST_PATH
        self.outpath = os.path.join(self.path, "output")
        self.rdbfile = os.path.join(self.path, "data", "trialcurves.txt")
        self.lcs, self.spline = pycs3.gen.util.readpickle(os.path.join(self.path, 'data', "optcurves.pkl"))

    def test_tweakml(self):
        lc_copy = [lc.copy() for lc in self.lcs]
        spline_copy = self.spline.copy()
        pycs3.sim.twk.tweakml(lc_copy, spline_copy, psplot=True)

    def test_tweakspl(self):
        lc_copy = [lc.copy() for lc in self.lcs]
        spline_copy = self.spline.copy()
        pycs3.sim.twk.tweakspl(spline_copy, psplot=True)