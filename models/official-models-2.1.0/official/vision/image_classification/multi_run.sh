#!/bin/bash

#SBATCH -J tf-tensorflow      # Job name
#SBATCH -o tf-tensorflow.o%j  # Name of stdout output file
#SBATCH -e tf-tensorflow.e%j  # Name of stderr error file
#SBATCH -p rtx                # Queue (partition) name
#SBATCH -N 6                  # Total # of nodes (must be 1 for serial)
#SBATCH -n 6                  # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 02:00:00           # Run time (hh:mm:ss)

ibrun -n 1 -o 0 ./run.sh 5 512 0 &
ibrun -n 1 -o 1 ./run.sh 4 512 0 &
ibrun -n 1 -o 2 ./run.sh 3 512 0 &
ibrun -n 1 -o 2 ./run.sh 1 512 0 &
ibrun -n 1 -o 3 ./run.sh 2 512 0 &
ibrun -n 1 -o 5 ./run.sh 0 512 0 #&

#sleep 140

#module load intel
#module load remora
#remora ./check_process_exits.sh "run.sh"

