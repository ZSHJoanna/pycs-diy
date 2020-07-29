#!/bin/bash -l

# ***** Script that launches multiple slurm jobs *****

# set the simulation basename and the file you want to run

export JOB_BASENAME="3all" # 3all for splines, 3c for regdiff
export SERVER_NAME="lesta"  # on Regor : 'r3', 'r4' ; otherwise 'deneb' or 'fidis'
export QUEUE_NAME="p4"  # on Regor : 'r3', 'r4' ; on deneb "debug" or "serial"

export DATA_NAME='ECAM'
export NUM_CORE="16"
export WORK_DIR='./'

max_node=1


start_file="start_3all.slurm" # 3all for splines, 3c for regdiff
job_name_csr=$OBJECT_NAME$JOB_BASENAME

echo "job name = "$job_name_csr
echo "start file = "$start_file
echo "max node = "$max_node


export LENS_NAME='test' # CHANGE NAME HERE : Ex. tdc1_rung0
export NAME_TYPE='double' # CHANGE TYPE HERE : Ex. double or quad
for i in $(seq 1 3); # CHANGE NUMBER OF SIMULATION HERE 
do
	export OBJECT_NAME=$LENS_NAME"_"$NAME_TYPE"_pair"$i
	
	echo "Pair $i : sending job as sbatch -J $job_name_csr -n 1 -c $NUM_CORE (-p $QUEUE_NAME) $start_file"
	sbatch -J $job_name_csr -p $QUEUE_NAME -n $NUM_CORE -N 1 $start_file
	sleep 0.5
done


