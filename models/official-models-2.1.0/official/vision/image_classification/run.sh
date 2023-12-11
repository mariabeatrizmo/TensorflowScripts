#!/bin/bash

#SBATCH -J rt-distributed-single-job      # Job name
#SBATCH -p normal                         # Queue (partition) name
#SBATCH -N 1                              # Total # of nodes (must be 1 for serial)
#SBATCH -n 1                              # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 02:00:00                       # Run time (hh:mm:ss)




source ~/tensorflow-venv/bin/activate
module load cuda/11.3 cudnn nccl
module load gcc
export CXX=/opt/apps/gcc/9.1.0/bin/g++
export CC=/opt/apps/gcc/9.1.0/bin/gcc
export INSTALL_DIR=/home1/09111/mbbm/tensorflow_scripts/scripts/monarch2/
export MONARCH_CONFIGS_PATH=/home1/09111/mbbm/tensorflow_scripts/scripts/monarch2/monarch/configurations/frontera/tf_placement_100g_disk.yaml

rm -r /tmp/500g_tfrecords/
rm -r /tmp/middleware_output

cp /scratch1/09111/mbbm/lixo_60.txt /dev/shm
python3.8 ~/creator.py

export PYTHONPATH=$PYTHONPATH:/home1/09111/mbbm/tensorflow_scripts/models/official-models-2.1.0
# export WRKS_ADDRS="192.168.44.122,192.168.44.128,192.168.44.141,192.168.44.144,192.168.44.145,192.168.44.159"
export WRKS_ADDRS=$(~/tensorflow_scripts/scripts/aux_monarch.sh)

TASK_INDEX=$1
BATCH_SIZE=$2
MODEL=""
EPOCHS=20
#DATA_DIR="/scratch1/09111/mbbm/100g_tfrecords"
DATA_DIR="/scratch1/09111/mbbm/open_images/tfrecords/train"

export TASK_ID=$TASK_INDEX

if [ $TASK_INDEX == 0 ]
then
	MODEL="sns_vgg19.py"
        EPOCHS=2

elif [ $TASK_INDEX == 1 ]
then
        MODEL="sns_inceptionv3.py"
        EPOCHS=5

elif [ $TASK_INDEX == 2 ]
then
        MODEL="sns_shufflenet.py"
        EPOCHS=10 #8

elif [ $TASK_INDEX == 3 ]
then
        MODEL="sns_resnet18.py"
        EPOCHS=10

elif [ $TASK_INDEX == 4 ]
then
        MODEL="sns_lenet.py"
        EPOCHS=10 #20

elif [ $TASK_INDEX == 5 ]
then
        MODEL="sns_alexnet.py"
        EPOCHS=10 #20
fi

EPOCHS=1000


LOG_PATH="/tmp/log_$TASK_INDEX.txt"

#echo -e "$SCRIPT"
#nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv -l 1 -f /tmp/nvidia-smi_${TASK_INDEX}.csv &

timeout 90m python3 $MODEL --skip_eval --train_epochs=$EPOCHS --batch_size=$BATCH_SIZE --model_dir="/tmp/checkpointing" --data_dir=$DATA_DIR --num_gpus=4 |& tee $LOG_PATH  
