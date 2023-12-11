#!/bin/bash

#SBATCH -J tf-tensorflow      # Job name
#SBATCH -o tf-tensorflow.o%j  # Name of stdout output file
#SBATCH -e tf-tensorflow.e%j  # Name of stderr error file
#SBATCH -p rtx                # Queue (partition) name
#SBATCH -N 4                  # Total # of nodes (must be 1 for serial)
#SBATCH -n 4                  # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 12:00:00           # Run time (hh:mm:ss)

ibrun -n 1 -o 0 ./job_tensorflow_distributed2.sh 0 &
ibrun -n 1 -o 1 ./job_tensorflow_distributed2.sh 1 &
ibrun -n 1 -o 2 ./job_tensorflow_distributed2.sh 2 &
ibrun -n 1 -o 3 ./job_tensorflow_distributed2.sh 3
