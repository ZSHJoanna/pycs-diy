import argparse as ap
import os

import pycs3.gen.lc_func
import pycs3.gen.util


def main(lensname, dataname, magerr, data_dir='../data/'):
    rdbfile = data_dir + lensname + '_' + dataname + '.rdb'
    print(rdbfile)

    with open(rdbfile, 'r') as f:
        header = f.readline()

    header = header.split('\t')
    lcs = []

    try :
        if "mag_A" in header:
            lcs.append(pycs3.gen.lc_func.rdbimport(rdbfile, 'A', 'mag_A', 'magerr_A_%i' % magerr, dataname))
        if "mag_B" in header:
            lcs.append(pycs3.gen.lc_func.rdbimport(rdbfile, 'B', 'mag_B', 'magerr_B_%i' % magerr, dataname))
        if "mag_C" in header:
            lcs.append(pycs3.gen.lc_func.rdbimport(rdbfile, 'C', 'mag_C', 'magerr_C_%i' % magerr, dataname))
        if "mag_D" in header:
            lcs.append(pycs3.gen.lc_func.rdbimport(rdbfile, 'D', 'mag_D', 'magerr_D_%i' % magerr, dataname))
    except Exception as e :
        if "mag_A" in header:
            lcs.append(pycs3.gen.lc_func.rdbimport(rdbfile, 'A', 'mag_A', 'magerr_A', dataname))
        if "mag_B" in header:
            lcs.append(pycs3.gen.lc_func.rdbimport(rdbfile, 'B', 'mag_B', 'magerr_B', dataname))
        if "mag_C" in header:
            lcs.append(pycs3.gen.lc_func.rdbimport(rdbfile, 'C', 'mag_C', 'magerr_C', dataname))
        if "mag_D" in header:
            lcs.append(pycs3.gen.lc_func.rdbimport(rdbfile, 'D', 'mag_D', 'magerr_D', dataname))
    pycs3.gen.util.multilcsexport(lcs, data_dir + lensname + '_' + dataname + '_reformated.rdb')


if __name__ == '__main__':
    parser = ap.ArgumentParser(prog="python {}".format(os.path.basename(__file__)),
                               description="Reformat the rdb file from COSMOULINE output to something readable by this pipeline. File must have the a name like OBJECT_DATANAME.rdb.",
                               formatter_class=ap.RawTextHelpFormatter)
    help_lensname = "name of the lens to process"
    help_dataname = "name of the data set to process (Euler, SMARTS, ... )"
    help_magerr = "Type of COSMOULINE error, take 5 by default"
    help_data_dir = "name of the data directory (default '../data/')"
    parser.add_argument(dest='lensname', type=str,
                        metavar='lens_name', action='store',
                        help=help_lensname)
    parser.add_argument(dest='dataname', type=str,
                        metavar='dataname', action='store',
                        help=help_dataname)
    parser.add_argument(dest='magerr', type=float,
                        metavar='magerr', action='store',
                        help=help_magerr)
    parser.add_argument('--dir', dest='data_dir', type=str,
                        metavar='', action='store', default='../data/',
                        help=help_data_dir)
    args = parser.parse_args()

    main(args.lensname, args.dataname, args.magerr, data_dir=args.data_dir)
