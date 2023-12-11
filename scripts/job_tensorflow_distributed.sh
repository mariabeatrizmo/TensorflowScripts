#!/bin/bash

WORKSPACE=$(dirname $(dirname $(realpath $0)))
DATA_DIR="/scratch1/09111/mbbm"

rm -r /tmp/500g_tfrecords
rm -r /tmp/alexnet-100g-bs*
cp /scratch1/09111/mbbm/lixo_60.txt /dev/shm
python3.8 ~/creator.py

#echo -e "cp 100g_tfrecords to /tmp"
#cp -r /scratch1/09111/mbbm/100g_tfrecords/ /tmp/100g_tfrecords/

#dd if=/scratch1/09111/mbbm/60g_tfrecords2 of=/dev/shm/60g_tfrecords
#rsync -av /scratch1/09111/mbbm/100g_tfrecords/ /tmp/100g_tfrecords/
#rsync -av /scratch1/09111/mbbm/60g_tfrecords/ /tmp/60g_tfrecords/
#rsync -av /scratch1/09111/mbbm/60g_tfrecords/ /dev/shm/60g_tfrecords/

#DATA_DIR="/tmp"
#DATA_DIR="/dev/shm"
#cd "${WORKSPACE}/scripts"

EPOCHS=100
BATCH_SIZE=1024
DATE="$(date +%Y_%m_%d-%H_%M)"
TARGET_DIR="/tmp"
TASK_INDEX=$1

echo "module load cuda/10.1 cudnn/7.6.5 nccl/2.5.6"
#module load cuda/10.1
#module load cudnn/7.6.5
#module load nccl/2.5.6
#module purge
module load cuda/11.3 cudnn nccl
# module load remora


# 100g
DATA_DIR="/scratch1/09111/mbbm"
#DATASET="${DATA_DIR}/100g_tfrecords"
#DATASET="${DATA_DIR}/60g_tfrecords"
#DATASET="/scratch1/09111/mbbm/imagenet/ILSVRC/Data/CLS-LOC/train/"
DATASET="/scratch1/09111/mbbm/open_images/tfrecords/train"
 
for i in {1..1}; do
  #RUN_DIR="${TARGET_DIR}/lenet-100g-bs${BATCH_SIZE}-ep${EPOCHS}-${DATE}"
  RUN_DIR="${TARGET_DIR}/alexnet-100g-bs${BATCH_SIZE}-ep${EPOCHS}-${DATE}-${TASK_INDEX}"
  #./train_distributed.sh -j $TASK_INDEX -o -m lenet -b $BATCH_SIZE -e $EPOCHS -g 4 -i autotune -l -v -d "$DATASET" -r $RUN_DIR
  ./train_distributed_teste.sh -j $TASK_INDEX -o -m alexnet -b $BATCH_SIZE -e $EPOCHS -g 4 -i autotune -l -v -d "$DATASET" -r $RUN_DIR
  #./train_multi_sns.sh -j $TASK_INDEX -o -m alexnet -b $BATCH_SIZE -e $EPOCHS -g 4 -i autotune -l -v -d "$DATASET" -r $RUN_DIR
  sleep 10
  #mv "remora_${SLURM_JOB_ID}"  $RUN_DIR

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
