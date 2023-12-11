#!/bin/bash

WORKSPACE=$(dirname $(dirname $(realpath $0)))
#DATA_DIR="/home/gsd/100g_tfrecords"
DATA_DIR="/scratch1/09111/mbbm/100g_tfrecords"

cd "${WORKSPACE}/scripts"

EPOCHS=20
BATCH_SIZE=512
DATE="$(date +%Y_%m_%d-%H_%M)"
TARGET_DIR="/tmp"
#cp /scratch1/09111/mbbm/lixo.txt /dev/shm

# 100g

DATASET="${DATA_DIR}"
 
for i in {1..1}; do
  RUN_DIR="${TARGET_DIR}/single_node-bs${BATCH_SIZE}-ep${EPOCHS}-${DATE}"
  ./train_single_node.sh -o -m alexnet -b $BATCH_SIZE -e $EPOCHS -g 4 -i autotune -l -v -d "$DATASET" -r $RUN_DIR 
  sleep 10
 #  mv "remora_${SLURM_JOB_ID}"  $RUN_DIR
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
