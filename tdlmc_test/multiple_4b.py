"""
This script will execute the 4b_marginalise_spline.py from the main pipeline for each of the light curve with sigma = 0, 0.5, 1000
"""
import sys 
import os
import argparse as ap
import subprocess
import py_compile

def update_and_execute(sigma, name, name_type, number_pair, work_dir, skip_pair = 0):
	config_directory = work_dir + "config/"
	dataname = "ECAM"
	lens_name = []	
	for i in range(1,number_pair+1):
		lens_name.append(name + '_' + name_type + '_' + 'pair%i'%i)
		if i<=skip_pair : continue
		line_count = 0
		tmp = ''
		with open(config_directory + 'config_' + lens_name[i-1] + '_ECAM.py', 'r') as f :
			Lines=f.readlines()
			for line in Lines :
				line_count += 1
				if (line_count == 122) :
					tmp += 'sigmathresh = %f   #sigma threshold for sigma clipping, 0 is a true marginalisation, choose 1000 to take the most precise.\n'%sigma
				else :
					tmp += line
		with open(config_directory + 'config_' + lens_name[i-1] + '_ECAM.py', 'w') as f :
			f.write(tmp)
			f.close()
		print(('Done updating the file : ' + config_directory + 'config_' + lens_name[i-1] + '_ECAM.py'))
		#update the pyc file aswell
		py_compile.compile(config_directory + 'config_' + lens_name[i-1] + '_ECAM.py')
		try :
			subprocess.call([sys.executable, '../scripts/4b_marginalise_spline.py', lens_name[i-1], dataname])
			print(("Sucessully executed 4b_marginalise_spline.py with sigma=%f and the pair %i"%(sigma,i)))
		except :
			print(("Error in script 4b_marginalise_spline.py with sigma=%f and the pair %i"%(sigma,i)))
			sys.exit()
	

def main(name, name_type, number_pair = 1, work_dir = './'):
	skip = 0
	update_and_execute(0, name, name_type, number_pair, work_dir, skip)
	update_and_execute(0.5, name, name_type, number_pair, work_dir, skip)
	update_and_execute(1000, name, name_type, number_pair, work_dir, skip)
	
	

if __name__ == '__main__':
    parser = ap.ArgumentParser(prog="python {}".format(os.path.basename(__file__)),
                               description="Reformat the txt file from the Time Delay Challenge to an usable rdb file.",
                               formatter_class=ap.RawTextHelpFormatter)
    help_name = "name of the sample. Make sure the directory has the same name."
    help_name_type = "Type of the data ie double or quad"
    help_number_pair = "number of pair in the rung folder. Make sure the folder have the format name_pair0"
    help_work_dir = "name of the work directory"
    parser.add_argument(dest='name', type=str,
                        metavar='name', action='store',
                        help=help_name)
    parser.add_argument(dest='name_type', type=str,
                        metavar='name_type', action='store',
                        help=help_name_type)
    parser.add_argument(dest='number_pair', type=int,
                        metavar='number_pair', action='store',
                        help=help_number_pair)                    
    parser.add_argument('--dir', dest='work_dir', type=str,
                        metavar='', action='store', default='./',
                        help=help_work_dir)
    args = parser.parse_args()
    main(args.name, args.name_type, args.number_pair, work_dir=args.work_dir)

