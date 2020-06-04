import os
import glob
from tests import TEST_PATH

import pycs3.gen.util
import pycs3.sim.power_spec
from pycs3.sim.src import Source
import pytest
import unittest
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal, assert_array_equal
from tests import utils


class TestPowerSpectrum(unittest.TestCase):
    def setUp(self):
        self.path = TEST_PATH
        self.outpath = os.path.join(self.path, "output")
        self.rdbfile = os.path.join(self.path, "data", "trialcurves.txt")
        self.lcs, self.spline = pycs3.gen.util.readpickle(os.path.join(self.path, 'data', "optcurves.pkl"))

    def test_PS(self):
        spline =self.spline.copy()
        source = Source(spline)
        power_spec = pycs3.sim.power_spec.PowerSpectrum(source)
        power_spec_copy = power_spec.copy()
        power_spec.calcslope()
        print(power_spec)
        print(power_spec.slope['slope'])
        assert_almost_equal(power_spec.slope['slope'], -7.2061329070514155, 7)

        pycs3.sim.power_spec.psplot([power_spec_copy],filename=os.path.join(self.outpath,'power_spec_plot.png'))
        pycs3.sim.power_spec.psplot([power_spec],nbins=30,filename=os.path.join(self.outpath,'power_spec_plot2.png'))


if __name__ == '__main__':
    pytest.main()