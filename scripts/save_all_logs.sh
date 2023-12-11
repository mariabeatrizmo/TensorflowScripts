#!/bin/bash

#SBATCH -J tf-tensorflow      # Job name
#SBATCH -o tf-tensorflow.o%j  # Name of stdout output file
#SBATCH -e tf-tensorflow.e%j  # Name of stderr error file
#SBATCH -p rtx                # Queue (partition) name
#SBATCH -N 3                  # Total # of nodes (must be 1 for serial)
#SBATCH -n 3                  # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 00:30:00           # Run time (hh:mm:ss)

ibrun -n 1 -o 0 ./save_logs_profiler.sh &
ibrun -n 1 -o 1 ./save_logs_profiler.sh &
ibrun -n 1 -o 2 ./save_logs_profiler.sh &
ibrun -n 1 -o 3 ./save_logs_profiler.sh &
ibrun -n 1 -o 4 ./save_logs_profiler.sh &
ibrun -n 1 -o 5 ./save_logs_profiler.sh #&
#ibrun -n 1 -o 6 ./save_logs_profiler.sh &
#ibrun -n 1 -o 7 ./save_logs_profiler.sh &
#ibrun -n 1 -o 8 ./save_logs_profiler.sh &
#ibrun -n 1 -o 9 ./save_logs_profiler.sh
