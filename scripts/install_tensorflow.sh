#!/bin/bash

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
