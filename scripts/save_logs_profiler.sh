#!/bin/bash

#DIR="/scratch1/09111/mbbm/logs"
DIR="/scratch1/09111/mbbm/logs/alexnet_w2_ep2_monarch_with_sharding"
if [ -d "$DIR" ]; then
    mv /tmp/log_c* $DIR
    mv /tmp/middleware_output/debugger/run* $DIR
    # mv /tmp/nvidia-smi_* $DIR
else
    mkdir $DIR
    mv /tmp/log_c* $DIR
    mv /tmp/middleware_output/debugger/run* $DIR
    #mv /tmp/nvidia-smi_* $DIR
fi
