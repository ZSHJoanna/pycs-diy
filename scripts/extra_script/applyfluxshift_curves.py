import argparse as ap
import sys
import os
import importlib
import pycs3.gen.lc_func
import pycs3.gen.stat
import pycs3.gen.util
import pycs3.gen.mrg
import pycs3.spl.multiopt
import numpy as np


def main(lensname, dataname, work_dir='../'):
    os.chdir('..')
    main_path = os.getcwd()
    sys.path.append(work_dir + "config/")
    sys.path.append(work_dir)
    config = importlib.import_module("config_" + lensname + "_" + dataname)

    figure_directory = config.figure_directory + "spline_and_residuals_plots/"
    if not os.path.isdir(figure_directory):
        os.mkdir(figure_directory)

    # fluxshift_vec = [0.000,0.199723, -0.002115,0.006784] #2M1134
    # fluxshift_vec = [0.000,0.113165,  -0.031351,-0.402337] #2WG0214
    fluxshift_vec = [0.000,0.437, 0.0,0.0] #2WG0214___2
    # fluxshift_vec = [0.000,0.199723, -0.002115,0.006784] #2M1134
    magshift_vec = [11.084, 11.130, 11.110, 10.355] #WG0214
    lcs = pycs3.gen.util.readpickle(config.data)
    pycs3.gen.lc_func.applyshifts(lcs, [0 for i in lcs], magshift_vec)  # remove median and set the time shift to the initial guess
    for lc,fs in zip(lcs,fluxshift_vec) :
        lc.shiftflux(fs)
        lc.applyfluxshift()

    pycs3.gen.util.multilcsexport(lcs, '/Users/martin/Desktop/DR2/extra_tests/%s_%sfs.rdb'%(lensname, dataname))






if __name__ == '__main__':
    parser = ap.ArgumentParser(prog="python {}".format(os.path.basename(__file__)),
                               description="Plot the final results.",
                               formatter_class=ap.RawTextHelpFormatter)
    help_lensname = "name of the lens to process"
    help_dataname = "name of the data set to process (Euler, SMARTS, ... )"
    help_work_dir = "name of the working directory"
    parser.add_argument(dest='lensname', type=str,
                        metavar='lens_name', action='store',
                        help=help_lensname)
    parser.add_argument(dest='dataname', type=str,
                        metavar='dataname', action='store',
                        help=help_dataname)
    parser.add_argument('--dir', dest='work_dir', type=str,
                        metavar='', action='store', default='./',
                        help=help_work_dir)
    args = parser.parse_args()
    main(args.lensname, args.dataname, work_dir=args.work_dir)