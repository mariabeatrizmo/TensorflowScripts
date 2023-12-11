#!/bin/bash

#SBATCH -J tf-tensorflow      # Job name
#SBATCH -o tf-tensorflow.o%j  # Name of stdout output file
#SBATCH -e tf-tensorflow.e%j  # Name of stderr error file
#SBATCH -p rtx                # Queue (partition) name
#SBATCH -N 4                  # Total # of nodes (must be 1 for serial)
#SBATCH -n 4                  # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 02:00:00           # Run time (hh:mm:ss)

ibrun -n 1 -o 0 ./run_homo.sh 0  256 0 &
ibrun -n 1 -o 1 ./run_homo.sh 1  512 0 &
ibrun -n 1 -o 2 ./run_homo.sh 2 1024 0 &
ibrun -n 1 -o 3 ./run_homo.sh 3  256 1 &
ibrun -n 1 -o 4 ./run_homo.sh 4  512 1 &
ibrun -n 1 -o 5 ./run_homo.sh 5 1024 1 #&


#sleep 140

#module load intel
#module load remora
#remora ./check_process_exits.sh "sns_lenet2.py"
