#!/bin/bash

WORKSPACE=$(dirname $(dirname $(realpath $0)))
#DATA_DIR="/home/gsd/100g_tfrecords"
DATA_DIR="/scratch1/09111/mbbm/100g_tfrecords"

cd "${WORKSPACE}/scripts"

EPOCHS=1
BATCH_SIZE=256
DATE="$(date +%Y_%m_%d-%H_%M)"
TARGET_DIR="/tmp"

# 100g

DATASET="${DATA_DIR}"
 
for i in {1..1}; do
#  RUN_DIR="/home/gsd/tensorflow_scripts/scripts/lenet-100g-bs${BATCH_SIZE}-ep${EPOCHS}-${DATE}"
#  ./train.sh -o -m lenet -b $BATCH_SIZE -e $EPOCHS -g 4 -i autotune -l -v -d "$DATASET" -r $RUN_DIR 
#  sleep 10
 #  mv "remora_${SLURM_JOB_ID}"  $RUN_DIR

  RUN_DIR="${TARGET_DIR}/alexnet-100g-bs${BATCH_SIZE}-ep${EPOCHS}-${DATE}"
  remora ./train.sh -o -m alexnet -b $BATCH_SIZE -e $EPOCHS -g 4 -i autotune -l -v -d "$DATASET" -r $RUN_DIR
  sleep 10
#  mv "remora_${SLURM_JOB_ID}" $RUN_DIR
#
#  RUN_DIR="${TARGET_DIR}/resnet-100g-bs${BATCH_SIZE}-ep${EPOCHS}-${DATE}"
#  remora ./train.sh -o -m resnet -b $BATCH_SIZE -e $EPOCHS -g 4 -i autotune -l -v -d "$DATASET" -r $RUN_DIR
#  sleep 10  
#  mv "remora_${SLURM_JOB_ID}" $RUN_DIR
done

# 200g

DATASET="${DATA_DIR}/imagenet_processed/200g_2048_tfrecords"

#for i in {1..1}; do
#  RUN_DIR="${TARGET_DIR}/lenet-200g-bs${BATCH_SIZE}-ep${EPOCHS}-${DATE}"
#  remora ./train.sh -o -m lenet -b $BATCH_SIZE -e $EPOCHS -g 4 -i autotune -v -d "$DATASET" -r $RUN_DIR -s 2048
#  sleep 10
#  mv "remora_${SLURM_JOB_ID}" $RUN_DIR
# 
#  RUN_DIR="${TARGET_DIR}/alexnet-200g-bs${BATCH_SIZE}-ep${EPOCHS}-${DATE}"
#  remora ./train.sh -o -m alexnet -b $BATCH_SIZE -e $EPOCHS -g 4 -i autotune -v -d "$DATASET" -r $RUN_DIR -s 2048
#  sleep 10
#  mv "remora_${SLURM_JOB_ID}" $RUN_DIR
# 
#  RUN_DIR="${TARGET_DIR}/resnet-200g-bs${BATCH_SIZE}-ep${EPOCHS}-${DATE}"
#  remora ./train.sh -o -m resnet -b $BATCH_SIZE -e $EPOCHS -g 4 -i autotune -v -d "$DATASET" -r $RUN_DIR -s 2048
#  sleep 10
#  mv "remora_${SLURM_JOB_ID}" $RUN_DIR
#done
