"""
Script to display the optimised light curves.
"""
import sys
import importlib
import os, glob
import argparse as ap
import pycs3.gen.stat
import pycs3.gen.util
import pycs3.gen.lc_func

show_res = True

def main(lensname,dataname, work_dir='./'):
    os.chdir('..')
    main_path = os.getcwd()
    sys.path.append(work_dir + "config/")
    sys.path.append(work_dir)
    config = importlib.import_module("config_" + lensname + "_" + dataname)

    if config.mltype == "splml":
        if config.forcen:
            ml_param = config.nmlspl
            string_ML = "nmlspl"
        else:
            ml_param = config.mlknotsteps
            string_ML = "knml"
    elif config.mltype == "polyml":
        ml_param = config.degree
        string_ML = "deg"
    else:
        raise RuntimeError('I dont know your microlensing type. Choose "polyml" or "spml".')

    for a, kn in enumerate(config.knotstep):
        for b, ml in enumerate(ml_param):
            for o, opt in enumerate(config.optset):
                pickle_files = glob.glob(config.lens_directory + config.combkw[a, b] +'/sims_%s_opt_%s/' % (config.simset_mock, opt) + '*_opt.pkl')
                for f in pickle_files :
                    opttweak = pycs3.gen.util.readpickle(f)
                    for lcs,spline in zip(opttweak["optlcslist"],opttweak["optfctoutlist"]):
                        print("Attached ML : ", lcs[0].ml)
                        pycs3.gen.lc_func.display(lcs, [spline], showlegend=True, showdelays=True, filename="screen")
                        if show_res :
                            rls = pycs3.gen.stat.subtract(lcs, spline)
                            pycs3.gen.stat.plotresiduals([rls])


if __name__ == '__main__':
    parser = ap.ArgumentParser(prog="python {}".format(os.path.basename(__file__)),
                               description="Visualize the results of the optimisaiton of the mock curves",
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
    main(args.lensname,args.dataname, work_dir=args.work_dir)