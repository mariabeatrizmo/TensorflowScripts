#!/bin/bash

DIR="/scratch1/09111/mbbm/new_homo_m/"
#DIR="/scratch1/09111/mbbm/new_profiling/hetero_4_md_lim"
#DIR="/home1/09111/mbbm/tensorflow_scripts/scripts/tmp"
if [ -d "$DIR" ]; then
    mv /tmp/log_* $DIR
    #mv /tmp/alexnet-100g-bs* $DIR/
    mv /tmp/middleware_output/debugger/run-0-c* $DIR
    #mv /home1/09111/mbbm/tensorflow_scripts/scripts/remora_* $DIR
    #mv /home1/09111/mbbm/tensorflow_scripts/models/official-models-2.1.0/official/vision/image_classification/remora_* $DIR
    #mv /tmp/nvidia-smi_* $DIR
else
    mkdir $DIR
    mv /tmp/alexnet-100g-bs* $DIR/
    mv /tmp/log_* $DIR
    mv /tmp/middleware_output/debugger/run-0-c* $DIR
    #mv /home1/09111/mbbm/tensorflow_scripts/models/official-models-2.1.0/official/vision/image_classification/remora_* $DIR
    #mv /home1/09111/mbbm/tensorflow_scripts/scripts/remora_* $DIR
    #mv /tmp/nvidia-smi_* $DIR
    #rm /tmp/nvidia-smi_*
    #rm -r /tmp/alexnet-100g-bs*
fi
