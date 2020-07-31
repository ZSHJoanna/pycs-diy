#!/bin/bash -l
#SBATCH -o slurm_%j.out  # STDOUT
#SBATCH -e slurm_%j.err  # STDERR
#SBATCH --exclude=node[202,219,220]

date
set -x
cd ..

run_command="srun -n 1 -c $NUM_CORE -u"
if [ "$SERVER_NAME" = "r4" ]; then
    run_options="--mpi=pmi2"
else
    run_options=""
fi

exec_command="python3 ../scripts/3c_optimise_copy_mocks.py"
exec_options="$OBJECT_NAME $DATA_NAME --dir $WORK_DIR"
whole_command="$run_command $run_options $exec_command $exec_options"

echo "running '$whole_command'"
eval $whole_command

date
