#!/bin/bash

# Change these variables (if needed)
# ======================================================================================
WORKSPACE=$(dirname $(dirname $(realpath $0)))
SCRIPT_DIR="${WORKSPACE}/models/official-models-2.1.0/official/vision/image_classification"
VENV_DIR="${HOME}/tensorflow-venv"
CHECKPOINTING_DIR="/tmp/checkpointing0"
# ======================================================================================

# Default values
DATASET_DIR=""
MODEL="resnet"
BATCH_SIZE=256
EPOCHS=10
NUM_GPUS=1
USE_PREFETCH=false
INTERLEAVE="false"
FITS_LOCALLY="false"
LOCAL_TRAIN_SIZE=896772
LUSTRE_TRAIN_SIZE=2946634
SKIP_EVAL=""
SHARD_SIZE=1024

RESOURCES_DIR="${WORKSPACE}/scripts"

#OUTPUT=$(squeue --me -j $SLURM_JOBID | awk 'NR > 1 {print $8}')

#echo "$OUTPUT"

#WORKER_HOSTS=$(python3 split_nodelist_workers.py $OUTPUT | sed 's/ c/,c/g' | sed 's/ //g')
#echo "$WORKER_HOSTS"

#WORKER_HOSTS=$(python3 split_nodelist_workers.py $OUTPUT | sed 's/ c/,c/g' | sed 's/ //g' | sed 's/[^,]*,\(.*\)/\1/g')

DISTRIBUTION_STRATEGY="multi_worker_mirrored"

#WORKER_HOSTS="c196-101:2222,c196-102:2222"
#WORKER_HOSTS="c196-081:2222,c196-082:2222"
#c198-042:2222,c198-051:2222,c198-052:2222

#WORKER_HOSTS="c196-102:2222,c196-111:2222"
#WORKER_HOSTS="c197-091:2222,c197-092:2222,c197-101:2222,c197-102:2222"
WORKER_HOSTS=$(./get_test_ips.sh)

echo "workers: $WORKER_HOSTS"

ALL_REDUCE_ALG="ring"
TASK_INDEX=0


# Functions
function export-vars {
	export PYTHONPATH=$PYTHONPATH:${WORKSPACE}/models/official-models-2.1.0
}

function monitor {
	# Add monitoring code here (if needed)
	#return
        sh $RESOURCES_DIR/iostat-csv.sh > $RUN_DIR/iostat.csv &
	nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv -l 1 -f $RUN_DIR/nvidia-smi.csv &
	#if [[ ! -z $COLLECT_IOPS ]]; then
	#	echo -e "entra COLLECT_IOPS"
	#	$RESOURCES_DIR/collect_lustre_stats.sh -r 5 -o $RUN_DIR -s llite -f scratch1 > /dev/null &
	#fi
}

function train-model {
	if [ "$MODEL" == "resnet" ]
	then 
		echo -e "Model: ResNet-50\nDataset: ImageNet\nBatch size: $BATCH_SIZE\nEpochs: $EPOCHS\nShuffle Buffer: $SHUFFLE_BUFFER\nGPUs: $NUM_GPUS\nFramework: Tensorflow \nDataset:${DATASET_DIR}" > $RUN_DIR/info.txt
		timeout 1h python3 $SCRIPT_DIR/resnet_imagenet_main.py $SKIP_EVAL --train_epochs=$EPOCHS --batch_size=$BATCH_SIZE --model_dir=$CHECKPOINTING_DIR --data_dir=$DATASET_DIR --num_gpus=$NUM_GPUS --distribution_strategy=$DISTRIBUTION_STRATEGY --worker_hosts=$WORKER_HOSTS --all_reduce_alg=$ALL_REDUCE_ALG --task_index=$TASK_INDEX |& tee $RUN_DIR/log.txt
	elif [ "$MODEL" == "alexnet" ]
	then 
		echo -e "Model: AlexNet\nDataset: ImageNet\nBatch size: $BATCH_SIZE\nEpochs: $EPOCHS\nShuffle Buffer: $SHUFFLE_BUFFER\nGPUs: $NUM_GPUS\nFramework: Tensorflow \nDataset:${DATASET_DIR}"> $RUN_DIR/info.txt
		timeout 1h python3 $SCRIPT_DIR/alexnet_imagenet_main.py $SKIP_EVAL --train_epochs=$EPOCHS --batch_size=$BATCH_SIZE --model_dir=$CHECKPOINTING_DIR --data_dir=$DATASET_DIR --num_gpus=$NUM_GPUS --distribution_strategy=$DISTRIBUTION_STRATEGY --worker_hosts=$WORKER_HOSTS --all_reduce_alg=$ALL_REDUCE_ALG --task_index=$TASK_INDEX |& tee $RUN_DIR/log_${TASK_INDEX}.txt
	elif [ "$MODEL" == "lenet" ]
	then 
		echo -e "Model: LeNet\nDataset: ImageNet\nBatch size: $BATCH_SIZE\nEpochs: $EPOCHS\nShuffle Buffer: $SHUFFLE_BUFFER\nGPUs: $NUM_GPUS\nFramework: Tensorflow \nDataset:${DATASET_DIR}"> $RUN_DIR/info.txt
		timeout 1h python3 $SCRIPT_DIR/lenet_imagenet_main.py $SKIP_EVAL --train_epochs=$EPOCHS --batch_size=$BATCH_SIZE --model_dir=$CHECKPOINTING_DIR --data_dir=$DATASET_DIR --num_gpus=$NUM_GPUS --distribution_strategy=$DISTRIBUTION_STRATEGY --worker_hosts=$WORKER_HOSTS --all_reduce_alg=$ALL_REDUCE_ALG --task_index=$TASK_INDEX |& tee $RUN_DIR/log.txt
		#python3 $SCRIPT_DIR/lenet_imagenet_main.py $SKIP_EVAL --train_epochs=$EPOCHS --batch_size=$BATCH_SIZE --model_dir=$CHECKPOINTING_DIR --data_dir=$DATASET_DIR --num_gpus=$NUM_GPUS  --distribution_strategy=$DISTRIBUTION_STRATEGY  --worker_hosts="c197-042:2222" --all_reduce_alg=$ALL_REDUCE_ALG --task_index=$TASK_INDEX |& tee $RUN_DIR/log.txt
	else
		echo "Select a valid model. Run train-model -h to see the available models"
	fi
}

function kill-monitor {
	echo -e "\nKilling jobs: \n$(jobs -p)"
	kill $(jobs -p)
	echo -e $(pgrep iostat)
	kill $(pgrep iostat)
}

function update-prefetch {
  if [ "$USE_PREFETCH" = true  ]
	then
		sed -i "s/#\?dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)/dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)/g" $SCRIPT_DIR/imagenet_preprocessing.py 
	else
		sed -i "s/#\?dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)/#dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)/g" $SCRIPT_DIR/imagenet_preprocessing.py 
	fi
}

function update-interleave {
  if [ "$INTERLEAVE" = "false"  ]
	then
		sed -i "s/#\?dataset = dataset.interleave(tf.data.TFRecordDataset.\+)/#dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=10, num_parallel_calls=tf.data.experimental.AUTOTUNE)/g" $SCRIPT_DIR/imagenet_preprocessing.py 
	elif [ "$INTERLEAVE" = "autotune"  ]
	then
		sed -i "s/#\?dataset = dataset.interleave(tf.data.TFRecordDataset.\+)/dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=10, num_parallel_calls=tf.data.experimental.AUTOTUNE)/g" $SCRIPT_DIR/imagenet_preprocessing.py 
	elif [ "$INTERLEAVE" = "none"  ]
	then
		sed -i "s/#\?dataset = dataset.interleave(tf.data.TFRecordDataset.\+)/dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=10)/g" $SCRIPT_DIR/imagenet_preprocessing.py 
	else
		sed -i "s/#\?dataset = dataset.interleave(tf.data.TFRecordDataset.\+)/dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=10, num_parallel_calls=$INTERLEAVE)/g" $SCRIPT_DIR/imagenet_preprocessing.py 
	fi
}

function update-dataset-size {
  if [ "$FITS_LOCALLY" == "true" ]
  then
    sed -i "s/'train':.*$/'train': ${LOCAL_TRAIN_SIZE},/" $SCRIPT_DIR/imagenet_preprocessing.py
  else
    sed -i "s/'train':.*$/'train': ${LUSTRE_TRAIN_SIZE},/" $SCRIPT_DIR/imagenet_preprocessing.py
  fi
}

function update-shard-size {
	sed -i "s/_NUM_TRAIN_FILES = .*/_NUM_TRAIN_FILES = ${SHARD_SIZE}/" $SCRIPT_DIR/imagenet_preprocessing.py
}

function update-caching {
	# Default is don't use cache 
	sed -i "s|#\?\(if is_training: # Uncomment to use caching\)|#\1|g" $SCRIPT_DIR/imagenet_preprocessing.py
	sed -i "s|#\?dataset = dataset.cache(.*)\( # Uncomment to use caching\)|#dataset = dataset.cache()\1|g" $SCRIPT_DIR/imagenet_preprocessing.py

	if [ -n "$CACHE_LOCATION" ]; then # Use cache
		if [ "$CACHE_LOCATION" == "mem" ]; then # Caching to mem
			sed -i "s|#\?\(if is_training: # Uncomment to use caching\)|\1|g" $SCRIPT_DIR/imagenet_preprocessing.py
			sed -i "s|#\?dataset = dataset.cache(.*)\( # Uncomment to use caching\)|dataset = dataset.cache()\1|g" $SCRIPT_DIR/imagenet_preprocessing.py
		elif [ -d $(dirname $CACHE_LOCATION) ]; then # Caching to file if the cache location is valid
			for cache_file in *CACHE_LOCATION*; do rm -f $cache_file; done # Removes data from previous caches if it exists
			sed -i "s|#\?\(if is_training: # Uncomment to use caching\)|\1|g" $SCRIPT_DIR/imagenet_preprocessing.py
	    sed -i "s|#\?dataset = dataset.cache(.*)\( # Uncomment to use caching\)|dataset = dataset.cache('$CACHE_LOCATION')\1|g" $SCRIPT_DIR/imagenet_preprocessing.py
		else 
			echo "Invalid cache location: $CACHE_LOCATION" >&2
		fi
	fi
}

function update-training-params {
	update-prefetch
	update-interleave
	update-dataset-size
	update-caching
	update-shard-size
}

source $VENV_DIR/bin/activate

# Update env vars
export-vars

# Handle flags
echo -e "\nHandling flags..."
while getopts ":holfvm:b:d:e:g:r:i:s:c:t:w:a:j:" opt; do
	case $opt in
		h)
			echo "$package - train Tensorflow models on ImageNet dataset"
			echo " "
			echo "$package [options] application [arguments]"
			echo " "
			echo "options:"
			echo "-h       show brief help"
			echo "-m       specify the model to train (lenet, resnet or alexnet)"
			echo "-b       specify batch size"
			echo "-e       specify number of epochs"
			echo "-g       specify number of GPUs"
			echo "-i       specify interleave num_parallel_calls argument (N, autotune, none)"
			echo "-f       use TF batch prefetching"
			echo "-l       dataset that fits locally will be used"
			echo "-d       dataset absolute path"
			echo "-c       use TF caching and specify caching location (mem or file path)"
			echo "-v       skip evaluation"
			echo "-s       shard size"
			echo "-t       distribution strategy"
			echo "-w       worker hosts addresses"
			echo "-a       all_reduce_alg (ex. ring)"
			echo "-j       task index for distributed setup"
			echo "-o       collect lustre IOPS"
			exit 0
			;;
		m)
			echo "-m was triggered, MODEL: $OPTARG" >&2
			MODEL=$OPTARG
			;;
		d)
			echo "-d was triggered, DATASET_DIR: $OPTARG" >&2
			DATASET_DIR=$OPTARG
			;;
		l)
			echo "-l was triggered: Datasets fits locally" >&2
			FITS_LOCALLY="true"
			;;
		b)
			echo "-b was triggered, BATCH_SIZE: $OPTARG" >&2
			BATCH_SIZE=$OPTARG
			;;
		e)
			echo "-e was triggered, EPOCHS: $OPTARG" >&2
			EPOCHS=$OPTARG
			;;
		g)
			echo "-g was triggered, NUM_GPUS: $OPTARG" >&2
			NUM_GPUS=$OPTARG
			;;
		i)
			echo "-i was triggered, Interleave: $OPTARG" >&2
			INTERLEAVE=$OPTARG
			;;
		f)
			echo "-f was triggered, Batch prefetch is enabled" >&2
			USE_PREFETCH=true
			;;
		v)
			echo "-v was triggered, SKIPING EVALUATION" >&2
			SKIP_EVAL="--skip_eval"
			;;
		r)
			echo "-r was triggered, run dir: $OPTARG" >&2
			RUN_DIR=$OPTARG
			;;
		c)
			echo "-c was triggered, caching is enable" >&2
			CACHE_LOCATION=$OPTARG
			;;
		s)
			echo "-s was triggered, shard size: $OPTARG" >&2
			SHARD_SIZE=$OPTARG
			;;
                t)
                        echo "-t was triggered, distribution strategy: $OPTARG" >&2
                        DISTRIBUTION_STRATEGY=$OPTARG
                        ;;
                w)
                        echo "-w was triggered, worker hosts adresses: $OPTARG" >&2
                        WORKER_HOSTS=$OPTARG
                        ;;
                a)
                        echo "-a was triggered, all_reduce_alg: $OPTARG" >&2
                        ALL_REDUCE_ALG=$OPTARG
                        ;;
                j)
                        echo "-j was triggered, Task index: $OPTARG" >&2
                        TASK_INDEX=$OPTARG
                        ;;
		o)
			echo "-o was triggered, collect lustre IOPS is enable" >&2
			COLLECT_IOPS=true
			#COLLECT_IOPS=false
			;;
		\?)
			echo "Invalid option: -$OPTARG" >&2
			exit 1
			;;
		:)
			echo "Option -$OPTARG requires an argument." >&2
			exit 1
			;;
	esac
done


# Create results directory
mkdir -p $RUN_DIR

# Create log file
touch $RUN_DIR/log.txt

# Update training parameters
update-training-params

# Start monitoring tools
trap 'kill 0' SIGINT # trap to kill all bg processes when pressing CTRL-C
monitor
sleep 10

# Start training the model
SECONDS=0
#LD_PRELOAD=/home1/09111/mbbm/sdlprof/build/libprofiler.so train-model
train-model
echo "ELAPSED TIME: $SECONDS s" | tee -a $RUN_DIR/log.txt 
sleep 10

# Kill monitor process
kill-monitor
