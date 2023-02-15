#!/bin/bash

#SBATCH -J install-tensorflow     # Job name
#SBATCH -o install-tensorflow.o%j # Name of stdout output file
#SBATCH -e install-tensorflow.e%j # Name of stderr error file
#SBATCH -p rtx                    # Queue (partition) name
#SBATCH -N 1                      # Total # of nodes (must be 1 for serial)
#SBATCH -n 1                      # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 00:20:00               # Run time (hh:mm:ss)

VENV_DIR="${HOME}/tensorflow-venv"

echo "rm -rf ${VENV_DIR}"
rm -rf $VENV_DIR

echo "python3 -m venv ${VENV_DIR}"
python3 -m venv $VENV_DIR

echo "source ${VENV_DIR}/bin/activate"
source $VENV_DIR/bin/activate

echo "pip3 install --upgrade pip"
pip3 install --upgrade pip

echo "pip install tensorflow==2.3.2"
pip install tensorflow==2.3.2
