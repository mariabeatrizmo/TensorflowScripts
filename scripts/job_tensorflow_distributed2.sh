#!/bin/bash

WORKSPACE=$(dirname $(dirname $(realpath $0)))
#DATA_DIR="/scratch1/09111/mbbm"

#echo -e "cp 100g_tfrecords to /tmp"
#cp -r /scratch1/09111/mbbm/100g_tfrecords/ /tmp/100g_tfrecords/

rsync -av /scratch1/09111/mbbm/100g_tfrecords/ /tmp/100g_tfrecords/

DATA_DIR="/tmp"

cd "${WORKSPACE}/scripts"

EPOCHS=1
BATCH_SIZE=1024
DATE="$(date +%Y_%m_%d-%H_%M)"
TARGET_DIR="/tmp"
TASK_INDEX=$1

echo "module load cuda/10.1 cudnn/7.6.5 nccl/2.5.6"
#module load cuda/10.1
#module load cudnn/7.6.5
#module load nccl/2.5.6
module load cuda/11.3 cudnn nccl

# 100g
DATASET="${DATA_DIR}/100g_tfrecords"
 
for i in {1..1}; do
  #RUN_DIR="${DATA_DIR}/alexnet-100g-bs${BATCH_SIZE}-ep${EPOCHS}-${DATE}"
  #RUN_DIR="${TARGET_DIR}/alexnet-100g-bs${BATCH_SIZE}-ep${EPOCHS}-${DATE}"
  #RUN_DIR="${TARGET_DIR}/lenet-100g-bs${BATCH_SIZE}-ep${EPOCHS}-${DATE}"
  RUN_DIR="${TARGET_DIR}/resnet-100g-bs${BATCH_SIZE}-ep${EPOCHS}-${DATE}"
  #./train_distributed.sh -j $TASK_INDEX -o -m alexnet -b $BATCH_SIZE -e $EPOCHS -g 4 -i autotune -l -v -d "$DATASET" -r $RUN_DIR
  #./train_distributed.sh -j $TASK_INDEX -o -m lenet -b $BATCH_SIZE -e $EPOCHS -g 4 -i autotune -l -v -d "$DATASET" -r $RUN_DIR
  ./train_distributed.sh -j $TASK_INDEX -o -m resnet -b $BATCH_SIZE -e $EPOCHS -g 4 -i autotune -l -v -d "$DATASET" -r $RUN_DIR
  sleep 10
#  mv "remora_${SLURM_JOB_ID}"  $RUN_DIR

#  RUN_DIR="${TARGET_DIR}/alexnet-100g-bs${BATCH_SIZE}-ep${EPOCHS}-${DATE}"
#  ./train_distributed.sh -j $TASK_INDEX -o -m alexnet -b $BATCH_SIZE -e $EPOCHS -g 4 -i autotune -l -v -d "$DATASET" -r $RUN_DIR
#  ./train.sh -o -m alexnet -b $BATCH_SIZE -e $EPOCHS -g 4 -i autotune -l -v -d "$DATASET" -r $RUN_DIR
#  sleep 10
#  mv "remora_${SLURM_JOB_ID}" $RUN_DIR
#
#  RUN_DIR="${TARGET_DIR}/resnet-100g-bs${BATCH_SIZE}-ep${EPOCHS}-${DATE}"
#  remora ./train.sh -o -m resnet -b $BATCH_SIZE -e $EPOCHS -g 4 -i autotune -l -v -d "$DATASET" -r $RUN_DIR
#  sleep 10  
#  mv "remora_${SLURM_JOB_ID}" $RUN_DIR
done


#cd /tmp
#rm -r 100g_tfrecords


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
