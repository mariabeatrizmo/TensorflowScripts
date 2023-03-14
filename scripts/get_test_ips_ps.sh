#!/bin/bash

#SBATCH -J tf-tensorflow      # Job name
#SBATCH -o tf-tensorflow.o%j  # Name of stdout output file
#SBATCH -e tf-tensorflow.e%j  # Name of stderr error file
#SBATCH -p rtx                # Queue (partition) name
#SBATCH -N 3                  # Total # of nodes (must be 1 for serial)
#SBATCH -n 3                  # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 00:30:00           # Run time (hh:mm:ss)

HOST_NAME=$(hostname)
FILE_NAME="${HOST_NAME:0:8}_hosts.txt"

ibrun -n 1 -o 0 python3 ip_finder.py > ${FILE_NAME}
ibrun -n 1 -o 1 python3 ip_finder.py >> ${FILE_NAME}
ibrun -n 1 -o 2 python3 ip_finder.py >> ${FILE_NAME}
ibrun -n 1 -o 3 python3 ip_finder.py >> ${FILE_NAME}
#ibrun -n 1 -o 4 python3 ip_finder.py >> ${FILE_NAME}
#ibrun -n 1 -o 5 python3 ip_finder.py >> ${FILE_NAME}
#ibrun -n 1 -o 6 python3 ip_finder.py >> ${FILE_NAME}
#ibrun -n 1 -o 7 python3 ip_finder.py >> ${FILE_NAME}
#ibrun -n 1 -o 8 python3 ip_finder.py >> ${FILE_NAME}
#ibrun -n 1 -o 9 python3 ip_finder.py >> ${FILE_NAME}


WORKER_HOSTS=$(python3 ip_finder_aux.py)
rm -f ${FILE_NAME}
echo -e "${WORKER_HOSTS}"

