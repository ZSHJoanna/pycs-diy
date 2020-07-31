"""
This script will setup the directories for the multiple pipeline.
It will also create a multiple config file which can then be modified by the user.
"""
import sys 
import os
import argparse as ap
from shutil import copyfile

def main(name, name_type, number_pair = 1, work_dir = './'): 
	data_directory = work_dir + "data/"
	config_directory = work_dir + "config/"
	multiple_config_directory = config_directory + 'multiple/'
	truth_directory = data_directory + "/truth/"
	Simulation_directory = work_dir + "Simulation/"
	Simulation_multiple_directory = Simulation_directory + "multiple/" 
	Simulation_name = Simulation_multiple_directory + name + "_double/"
	if not os.path.exists(data_directory):
		print("I will create the data directory for you ! ")
		os.mkdir(data_directory)
	if not os.path.exists(config_directory):
		print("I will create the config directory for you ! ")
		os.mkdir(config_directory)
	if not os.path.exists(multiple_config_directory):
		print("I will create the data directory for you ! ")
		os.mkdir(multiple_config_directory)
	if not os.path.exists(truth_directory):
		print("I will create the config directory for you ! ")
		os.mkdir(truth_directory)
	if not os.path.exists(Simulation_directory):
		print("I will create the Simulation directory for you ! ")
		os.mkdir(Simulation_directory)
	if not os.path.exists(Simulation_multiple_directory):
		print("I will create the Simulation/multiple directory for you ! ")
		os.mkdir(Simulation_multiple_directory)
	if not os.path.exists(Simulation_name):
		print("I will create the Simulation/multiple/name directory for you ! ")
		os.mkdir(Simulation_name)

	
	### Create the multiple config file 
	copyfile("config_default_multiple_double.py", multiple_config_directory+ "config_multiple_" + name + ".py")

		
		

	
	
	

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

