import matplotlib
matplotlib.use('Agg')
import os
import numpy as np
import pytest
import unittest
import glob
import matplotlib.pyplot as plt

from tests import TEST_PATH
import pycs3.gen.polyml
import pycs3.gen.splml
from tests import utils
import pycs3.gen.mrg as mrg
import pycs3.gen.lc_func as lc_func
from numpy.testing import assert_allclose, assert_almost_equal, assert_array_equal


class TestLightCurve(unittest.TestCase):
    def setUp(self):
        self.path = TEST_PATH
        self.outpath = os.path.join(self.path, "output")
        self.rdbfile = os.path.join(self.path, "data", "trialcurves.txt")
        self.rdbfile_WFI = os.path.join(self.path, "data", "DES0408_WFI.rdb")
        self.rdbfile_ECAM = os.path.join(self.path, "data", "DES0408_ECAM.rdb")
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
        self.lcs_WFI = [
            lc_func.rdbimport(self.rdbfile_WFI, object='A', magcolname='mag_A', magerrcolname='magerr_A_5',
                              telescopename="WFI"),
            lc_func.rdbimport(self.rdbfile_WFI, object='B', magcolname='mag_B', magerrcolname='magerr_B_5',
                              telescopename="WFI"),
            lc_func.rdbimport(self.rdbfile_WFI, object='D', magcolname='mag_D', magerrcolname='magerr_D_5',
                              telescopename="WFI")
        ]
        self.lcs_ECAM = [
            lc_func.rdbimport(self.rdbfile_ECAM, object='A', magcolname='mag_A', magerrcolname='magerr_A_5',
                              telescopename="ECAM",propertycolnames="lcmanip"),
            lc_func.rdbimport(self.rdbfile_ECAM, object='B', magcolname='mag_B', magerrcolname='magerr_B_5',
                              telescopename="ECAM",propertycolnames="lcmanip"),
            lc_func.rdbimport(self.rdbfile_ECAM, object='D', magcolname='mag_D', magerrcolname='magerr_D_5',
                              telescopename="ECAM",propertycolnames="lcmanip")
        ]
        self.guess_timeshifts = [0., 0., -15., -65.]
        self.true_delays = [-5.0, -20.0, -70., -15., -65., -50.]
        mrg.colourise(self.lcs)

    def tearDown(self):
        plt.close('all')

    def test_lc_infos(self):
        print(self.lcs[0])
        print(self.lcs[0].printinfo())  # call lc.longinfo()
        stats = {'len': 192, 'nseas': 4, 'meansg': 163.99396333333425, 'minsg': 130.38524000000325,
                 'maxsg': 210.23221999999805, 'stdsg': 33.7985590219782, 'med': 3.989030000000639,
                 'mean': 4.38115308510637, 'max': 25.124900000002526, 'min': 0.8326200000010431,
                 'std': 3.460329376531179}
        test_stat = self.lcs[0].samplingstats()
        for key in test_stat.keys():
            self.assertAlmostEqual(stats[key], test_stat[key], places=3)
        commonproperties = self.lcs[0].commonproperties()
        commonproperties2 = self.lcs[0].commonproperties(notonlycommon=True)
        jds_ranges = (54018.545346499996, 55465.7478835)
        mag_ranges = (-9.51947, -13.70035)
        test_jds_ranges, test_mag_ranges = lc_func.displayrange(self.lcs)
        assert_almost_equal(test_jds_ranges, jds_ranges, decimal=3)
        assert_almost_equal(test_mag_ranges, mag_ranges, decimal=3)

    def test_opt_spline_polyml(self):
        lc_copy = [lc.copy() for lc in self.lcs]
        pycs3.gen.lc_func.applyshifts(lc_copy, self.guess_timeshifts, [-np.median(lc.getmags()) for lc in lc_copy])
        pycs3.gen.polyml.addtolc(lc_copy[1], nparams=2, autoseasonsgap=60.0)  # add affine microlensing to each season
        pycs3.gen.polyml.addtolc(lc_copy[2], nparams=3,
                                 autoseasonsgap=600.0)  # add polynomial of degree 2 on the entire light curve
        pycs3.gen.polyml.addtolc(lc_copy[3], nparams=3, autoseasonsgap=600.0)
        spline = utils.spl(lc_copy)
        delays = lc_func.getdelays(lc_copy)
        delays = lc_func.getdelays(lc_copy, to_be_sorted=True)
        lc_func.getnicetimedelays(lc_copy)
        lc_func.getnicetimedelays(lc_copy, to_be_sorted=True)
        lc_func.display(lc_copy, [spline], style="homepagepdf",
                        filename=os.path.join(self.outpath, 'spline_wi_ml1.png'))
        lc_func.display(lc_copy, [spline], style="homepagepdfnologo",
                        filename=os.path.join(self.outpath, 'spline_wi_ml2.png'))
        lc_func.display(lc_copy, [spline], style="2m2_largeticks", filename=os.path.join(self.outpath, 'spline_wi_ml3.png'))
        lc_func.display(lc_copy, [spline], style="posterpdf", filename=os.path.join(self.outpath, 'spline_wi_ml4.png'))
        lc_func.display(lc_copy, [spline], style="internal", filename=os.path.join(self.outpath, 'spline_wi_ml5.png'))
        lc_func.display(lc_copy, [spline], style="cosmograil_dr1",
                        filename=os.path.join(self.outpath, 'spline_wi_ml6.png'))
        figure = plt.figure()
        ax = plt.subplot(111)
        ax = lc_func.display(lc_copy, [spline], style="cosmograil_microlensing", ax=ax, showspldp= True, verbose=True)
        figure.savefig('spline_wi_ml7.png')

        delay_th = [-6.380928, -26.039074, -70.74382, -19.658146, -64.362892,
                    -44.704746]  # delay not accurately recover but this is because the poor ml model
        assert_allclose(delays, delay_th, atol=1.)

        # play a bit with the microlensing object:
        microlensing = lc_copy[2].ml
        microlensing.printinfo()
        stat = microlensing.stats(lc_copy[2])
        stat_th = {'mean': -0.17782270209855014, 'std': 0.2211515237812574}
        for key in stat.keys():
            assert_almost_equal(stat[key], stat_th[key], decimal=2)

        # play a bit with the params.
        params = pycs3.gen.polyml.multigetfreeparams(lc_copy)
        pycs3.gen.polyml.multisetfreeparams(lc_copy, params)
        microlensing.reset()

    def test_opt_spline_splml(self):
        lc_copy = [lc.copy() for lc in self.lcs]
        pycs3.gen.lc_func.applyshifts(lc_copy, self.guess_timeshifts, [-np.median(lc.getmags()) for lc in lc_copy])
        mlknotstep = 200
        mlbokeps_ad = mlknotstep / 3.0  # maybe change this
        pycs3.gen.splml.addtolc(lc_copy[1], knotstep=mlknotstep, bokeps=mlbokeps_ad)
        pycs3.gen.splml.addtolc(lc_copy[2], knotstep=mlknotstep, bokeps=mlbokeps_ad)
        pycs3.gen.splml.addtolc(lc_copy[3], knotstep=mlknotstep, bokeps=mlbokeps_ad)
        spline = utils.spl(lc_copy)
        tv, dist = pycs3.gen.spl_func.mltv(lc_copy, spline)  # some metric of the fit
        assert tv < 900
        assert dist < 5950

        delays = lc_func.getdelays(lc_copy, to_be_sorted=True)
        print(delays)
        print(lc_copy[0].longinfo())
        assert_allclose(delays, self.true_delays, atol=2.)
        lc_func.display(lc_copy, [spline], style="homepagepdf",
                        filename=os.path.join(self.outpath, 'spline_wi_splml.png'))
        pycs3.gen.util.writepickle((lc_copy, spline), os.path.join(self.outpath, 'optcurves.pkl'))
        mags = lc_copy[0].getmags(noml=True)

        # trace function :
        self.clean_trace()
        pycs3.gen.util.trace(lc_copy, spline, tracedir=self.outpath)
        pycs3.gen.util.plottrace(tracedir=self.outpath)

    def test_fluxshift(self):
        shifts = [-0.1, -0.2, -0.3, -0.4]
        lc_copy = [lc.copy() for lc in self.lcs]
        lc0 = lc_copy[0].copy()
        lc0.getrawfluxes()
        min_flux = lc_copy[0].getminfluxshift()
        assert_almost_equal(min_flux, -64108.55676508218)

        for i, lc in enumerate(lc_copy):
            lc.resetml()
            lc.resetshifts()
            lc.setfluxshift(shifts[i], consmag=True)

        magshifts = [lc.magshift for lc in lc_copy]
        assert_allclose(magshifts, [-9.467925255575702e-07, -3.6197204037074295e-06, -1.3700547842369769e-05,
                                    -3.143986495236058e-05], rtol=1e-3)

        new_mags = lc_copy[0].getmags()
        assert_almost_equal(new_mags[0], -12.38173974)

        fluxvector = shifts[0] * np.ones(len(lc0.mags))
        lc0.addfluxes(fluxvector)
        assert_allclose(lc0.getmags(), lc_copy[0].getmags(), rtol=0.0000001)
        lc_copy[0].calcfluxshiftmags(inverse=True)

        lc_func.shuffle(lc_copy)
        lcs_sorted = lc_func.objsort(lc_copy, ret=True)
        lc_func.objsort(lc_copy, ret=False)
        lc_copy[0].shiftflux(0.1)

    def test_timeshifts(self):
        lc_copy = [lc.copy() for lc in self.lcs]
        lc_copy2 = [lc.copy() for lc in self.lcs]
        shifts = np.asarray([0., 10., 20., 30.])
        lc_func.settimeshifts(lc_copy, shifts, includefirst=True)
        lc_func.settimeshifts(lc_copy2, shifts[1:4], includefirst=False)
        test_shift1 = lc_func.gettimeshifts(lc_copy, includefirst=True)
        test_shift2 = lc_func.gettimeshifts(lc_copy2, includefirst=False)
        assert_array_equal(test_shift1, shifts)
        assert_array_equal(test_shift2, shifts[1:4])

    def test_mask(self):
        lc_copy = [lc.copy() for lc in self.lcs]
        for i, lc in enumerate(lc_copy):
            lc.maskskiplist(self.skiplist, searchrange=4, accept_multiple_matches=True, verbose=True)

        print(lc_copy[0].maskinfo())
        assert np.sum(lc_copy[0].mask == False) == 14
        assert lc_copy[0].hasmask()
        lc_copy[0].cutmask()
        assert not lc_copy[0].hasmask()

        lc_copy[1].clearmask()
        assert not lc_copy[1].hasmask()

        lc_copy[1].pseudobootstrap()
        print("Bootstrap masked %i datapoints" % np.sum(lc_copy[1].mask == False))

    def test_montecarlo(self):
        lc0 = self.lcs[0].copy()
        lc1 = self.lcs[1].copy()
        lc0_copy = self.lcs[0].copy()
        lc1_copy = self.lcs[1].copy()
        pycs3.gen.polyml.addtolc(lc0, nparams=2, autoseasonsgap=60.0)
        lc0.montecarlomags()
        lc0.montecarlojds(amplitude=0.5, seed=1, keepml=True)
        lc1.montecarlojds(amplitude=0.5, seed=1, keepml=False)

        lc0_copy.merge(lc0)
        lc1_copy.merge(lc1)
        lc0.rdbexport(filename=os.path.join(self.outpath, "merged_A+A.txt"))
        lc1.rdbexport(filename=os.path.join(self.outpath, "merged_noml_A+A.txt"))

    def test_add_rm_ml(self):
        lc_copy = [lc.copy() for lc in self.lcs]
        print("0", lc_copy[1].ml)
        pycs3.gen.splml.addtolc(lc_copy[1], knotstep=200, bokeps=200 / 3.)
        print("1", lc_copy[1].ml)
        pycs3.gen.polyml.addtolc(lc_copy[1], nparams=2, autoseasonsgap=60.0)  # add affine microlensing to each season
        print("2", lc_copy[1].ml)
        lc_copy[1].resetml()  # reset not remove
        print("3", lc_copy[1].ml)
        lc_copy[1].rmml()  # reset not remove
        print("4", lc_copy[1].ml)

    def test_jdlabels(self):
        lc_copy = [lc.copy() for lc in self.lcs]
        lc_copy[1].setjdlabels()
        lc_copy[1].showlabels = True
        lc_copy[1].remove_epochs(np.arange(50,len(lc_copy[1].jds)))
        lc_func.display([lc_copy[1]], [], style="cosmograil_dr1_microlensing", collapseref=True, hidecollapseref=True,
                        showlogo=True, logopos='center', showinsert=True,
                        insertname=os.path.join(self.path, "data", "EPFL_Logo.png"), magrange=[-11., -12.4], jdrange=[54000, 54600],
                        filename=os.path.join(self.outpath, 'test_jd_labels.png'), verbose=True)
        lc_copy[1].clearlabels()
        lc_copy[1].clearproperties()
        lc_copy[1].validate(verbose=True)

    def test_factory(self):
        mags = 0.1*np.random.randn(10) + 1.
        jds = np.arange(1,11)
        lc = lc_func.factory(jds, mags, verbose=True)

    def test_import(self):
        lc_ECAM = lc_func.flexibleimport(os.path.join(self.path, "data", "DES0408_ECAM.rdb"), jdcol=1,startline=3,
                                         magcol=12, errcol=13, flagcol=11, propertycols={"fwhm":6}, absmagerrs=True)


    def test_display(self):
        txt = 'Trial curves'
        colour = 'black'
        kwargs = {"fontsize": 22, "color": 'black'}
        disptext = [(0.8, 0.8, txt, kwargs)]
        lc_func.display(self.lcs, [], showlogo=True, logopos='center', showgrid=True, showdates=False, magrange=3, showdelays=True,
                        figsize=(15,10), text=disptext,
                        filename=os.path.join(self.outpath, 'display_test.png'))
        lc_func.display(self.lcs_WFI, [], colourprop=('fwhm', 'Seeing', 0,3), hidecolourbar=True,
                        filename=os.path.join(self.outpath, 'display_test2.png'), titlexpos=0.3)
        lc_func.display(self.lcs_WFI, [], colourprop=('fwhm', 'Seeing', 0,3), hidecolourbar=False,  figsize=(12,9),
                        filename=os.path.join(self.outpath, 'display_test2.png'))

    def test_linintp(self):
        lc_func.linintnp(self.lcs_WFI[0].copy(), self.lcs_ECAM[0].copy(), usemask=False, weights=False, plot=True,
                         filename=os.path.join(self.outpath, 'test_linintp.png'))

    def test_interpolate(self):
        lc0 = self.lcs[0].copy()
        lc1 = self.lcs[1].copy()
        lc_interp = lc_func.interpolate(lc0, lc1, interpolate='nearest')
        lc_interp2 = lc_func.interpolate(lc0, lc1, interpolate='linear')
        lc_interp.plotcolour = 'black'
        lc_interp2.plotcolour = 'purple'
        lc_func.display([lc0,lc1,lc_interp,lc_interp2], [], filename=os.path.join(self.outpath, 'test_interp_nearest.png'))

    def clean_trace(self):
        pkls = glob.glob(os.path.join(self.outpath, "??????.pkl"))
        for pkl in pkls:
            os.remove(pkl)


if __name__ == '__main__':
    pytest.main()
