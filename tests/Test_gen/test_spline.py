import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import pytest
import unittest
from tests import TEST_PATH
import pycs3.gen.lc_func as lc_func
import pycs3.gen.mrg as mrg
import pycs3.gen.spl_func as spl_func
from pycs3.gen.datapoints import DataPoints
from numpy.testing import assert_almost_equal
import numpy as np
import copy

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
        self.datapts =DataPoints(self.lcs[0].jds, self.lcs[0].mags, self.lcs[0].magerrs, splitup=False, sort=True,
                             stab=True, stabext=300.0, stabgap=30.0, stabstep=5.0, stabmagerr=-2.0, stabrampsize=5.0,
                             stabrampfact=1.0)

    def tearDown(self):
        plt.close('all')

    def test_datapoints(self):
        datapts = copy.deepcopy(self.datapts)
        datapts.putstab()
        bounds = datapts.getmaskbounds()
        print(bounds)
        assert datapts.ntrue() == 192

    def test_spline(self):
        spline = spl_func.fit(self.lcs, bokmethod = 'BF')
        spline.display(filename=os.path.join(self.outpath, "spline_plot_BF.png"))
        r2 = spline.r2()
        print("R2 BF:", r2)
        spline.reset()

        #try other technique
        r2 = spline.bok(bokmethod="MCBF")
        spline.display(filename=os.path.join(self.outpath, "spline_plot_MCBF.png"))
        print("R2 MCBF:", r2)
        spline.reset()

        r2 = spline.bok(bokmethod="fminind")
        spline.display(filename=os.path.join(self.outpath, "spline_plot_fminind.png"))
        print("R2 fminind :", r2)
        spline.reset()

        r2 = spline.bok(bokmethod="fmin")
        spline.display(filename=os.path.join(self.outpath, "spline_plot_fmin.png"))
        print("R2 fmin:", r2)
        print("TV :", spline.tv())

        #test shift
        meant = np.mean(spline.t)
        spline.shifttime(-10)
        spline.shiftmag(-0.2)
        assert_almost_equal(np.mean(spline.t), meant - 10)
        print(spline.knotstats())
        coef = spline.getco()
        spline.setco(coef)

    def test_updatedp(self):
        myspline = spl_func.fit(self.lcs, bokmethod = 'BF')
        newdp = spl_func.merge(self.lcs, stab=False)  # Indeed we do not care about stabilization points here.
        myspline.updatedp(newdp, dpmethod="leave")
        r2 = myspline.r2(nostab=True)
        print(r2)

        myspline.updatedp(newdp, dpmethod="stretch")
        r2 = myspline.r2(nostab=True)
        print(r2)

        myspline.updatedp(newdp, dpmethod="extadj")
        r2 = myspline.r2(nostab=True)
        print(r2)

if __name__ == '__main__':
    pytest.main()
