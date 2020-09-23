import matplotlib
matplotlib.use('Agg')
import os
from tests import TEST_PATH

import pycs3.gen.util
import pycs3.pipe.pipe_utils as piut
import pytest
import unittest
import numpy as np

NUM = np.nan # variable to be replaced when writing the function
def function_trial(test_number=NUM):
    print('Hello world !',test_number)

class TestPipeUtils(unittest.TestCase):
    def setUp(self):
        self.path = TEST_PATH
        self.outpath = os.path.join(self.path, "output")
        self.rdbfile = os.path.join(self.path, "data", "trialcurves.txt")
        self.lcs, self.spline = pycs3.gen.util.readpickle(os.path.join(self.path, 'data', "optcurves.pkl"))
        self.outstream = open(os.path.join(self.outpath, 'test_func.py'),'w')
        self.regdiff_file = os.path.join(self.path, "data", "preset_regdiff.txt")
        self.delays = [-5.0, -20.0, -70.]

    def test_get_delays(self):
        delay_pair, delay_name = piut.getdelays(self.lcs)
        print(delay_name, delay_pair)

    def test_write_func(self):
        piut.write_func_append(function_trial,self.outstream, NUM=str(6))

    def test_generate_regdiff_kw(self):
        kw_list = piut.generate_regdiffparamskw(pointdensity=[2,3],covkernel=['Matern'], pow=[1.5,2.5], errscale=[1.] )
        print(kw_list)

    def test_read_preselected_regdiff_kw(self):
        kw_list = piut.read_preselected_regdiffparamskw(self.regdiff_file)
        print(kw_list)

    def test_read_preselected_regdiff(self):
        dic_list = piut.get_keyword_regdiff_from_file(self.regdiff_file)
        print(dic_list)

    def test_generate_regdiff_dic(self):
        dic_list = piut.get_keyword_regdiff(pointdensity=[2,3],covkernel=['Matern'], pow=[1.5,2.5], errscale=[1.] )
        print(dic_list)

    def test_generate_spline_dic(self):
        dic = piut.get_keyword_spline(30)
        print(dic)

    def test_convert_delay(self):
        timeshift = piut.convert_delays2timeshifts(self.delays)
        print(timeshift)


if __name__ == '__main__':
    pytest.main()