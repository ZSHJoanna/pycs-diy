import matplotlib
matplotlib.use('Agg')
import os
from tests import TEST_PATH
import pycs3.gen.util
import pycs3.sim.twk
import unittest
import pytest


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
        spline_copy = self.spline.copy()
        pycs3.sim.twk.tweakspl(spline_copy, psplot=True)

    def test_tweakmlPS(self):
        lc_copy = [lc.copy() for lc in self.lcs]
        for k, l in enumerate(lc_copy):
            if l.ml == None:
                print('I dont have ml, I have to introduce minimal extrinsic variation to generate the mocks. Otherwise I have nothing to modulate.')
                pycs3.gen.splml.addtolc(l, n=2)

        spline_copy = self.spline.copy()
        pycs3.sim.twk.tweakml_PS(lc_copy,spline_copy, 1, psplot=True, verbose=True)

    def test_bandnoise(self):
        noise = pycs3.sim.twk.band_limited_noise(1./1000., 1.)
        print(noise)


if __name__ == '__main__':
    pytest.main()