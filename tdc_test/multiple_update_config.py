"""
This script will modify all the single config file of the curves in order to set the simoptfctkw = "regdiff"
"""
import os
import argparse as ap
import py_compile

def main(name, name_type, number_pair = 1, work_dir = './'):	
	config_directory = work_dir + "config/"
	multiple_config_directory = config_directory + "multiple/"
	data_directory = work_dir + "data/"
	guess_directory = data_directory + name + "/guess/"
	dataname = 'ECAM'
    
    ### Modify the config file 
	lens_name = []	
	for i in range(1,number_pair+1):
		lens_name.append(name + '_' + name_type + '_' + 'pair%i'%i)
		line_count = 0
		tmp = ''
		with open(config_directory + 'config_' + lens_name[i-1] + '_ECAM.py', 'r') as f :
			Lines=f.readlines()
			for line in Lines :
				line_count += 1
				if (line_count == 28) :
					tmp += 'simoptfctkw = "regdiff" #function you want to use to optimise the mock curves, currently support spl1 and regdiff\n'
				else :
					tmp += line
		with open(config_directory + 'config_' + lens_name[i-1] + '_ECAM.py', 'w') as f :
			f.write(tmp)
			f.close()
		print(('Done updating the file : ' + config_directory + 'config_' + lens_name[i-1] + '_ECAM.py'))
		#update the pyc file aswell
		py_compile.compile(config_directory + 'config_' + lens_name[i-1] + '_ECAM.py')
		
		
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
