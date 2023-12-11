#!/bin/bash

# Change these variables (if needed)
# ======================================================================================
WORKSPACE=$(dirname $(dirname $(realpath $0)))
SCRIPT_DIR="${WORKSPACE}/models/official-models-2.1.0/official/vision/image_classification"
VENV_DIR="${HOME}/tensorflow-venv"
CHECKPOINTING_DIR="/tmp/checkpointing"
# ======================================================================================
RESOURCES_DIR="${WORKSPACE}/scripts"

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

# Functions
function export-vars {
	export PYTHONPATH=$PYTHONPATH:${WORKSPACE}/models/official-models-2.1.0
}

function monitor {
	# Add monitoring code here (if needed)
	return
        #sh $RESOURCES_DIR/iostat-csv.sh > $RUN_DIR/iostat.csv &
        #nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv -l 1 -f $RUN_DIR/nvidia-smi.csv# &
        #if [[ ! -z $COLLECT_IOPS ]]; then
        #       $RESOURCES_DIR/collect_lustre_stats.sh -r 5 -o $RUN_DIR -s llite -f scratch1 > /dev/null &
        #fi
}

function train-model {
	#python3.8 -c "from absl import app;from absl import flags;FLAGS = flags.FLAGS; flags.DEFINE_string('model','alexnet'); flags.DEFINE_string('optimizer','adam')"
        #single_node.py
        #echo -e "Python script \n\n\n\"
	#python3.8 $SCRIPT_DIR/sns_inceptionv3.py  $SKIP_EVAL --train_epochs=$EPOCHS --batch_size=$BATCH_SIZE --model_dir=$CHECKPOINTING_DIR --data_dir=$DATASET_DIR --num_gpus=$NUM_GPUS |& tee $RUN_DIR/log.txt
        #python3.8 $SCRIPT_DIR/sns_resnet18.py  $SKIP_EVAL --train_epochs=$EPOCHS --batch_size=$BATCH_SIZE --model_dir=$CHECKPOINTING_DIR --data_dir=$DATASET_DIR --num_gpus=$NUM_GPUS |& tee $RUN_DIR/log.txt
        LD_PRELOAD=/home1/09111/mbbm/tensorflow_scripts/scripts/final_versions/monarch/pastor/build/libmonarch.so  python3.8 $SCRIPT_DIR/sns_vgg19.py  $SKIP_EVAL --train_epochs=$EPOCHS --batch_size=$BATCH_SIZE --model_dir=$CHECKPOINTING_DIR --data_dir=$DATASET_DIR --num_gpus=$NUM_GPUS |& tee $RUN_DIR/log.txt
        #python3.8 $SCRIPT_DIR/sns_lenet.py  $SKIP_EVAL --train_epochs=$EPOCHS --batch_size=$BATCH_SIZE --model_dir=$CHECKPOINTING_DIR --data_dir=$DATASET_DIR --num_gpus=$NUM_GPUS |& tee $RUN_DIR/log.txt
}

function kill-monitor {
	echo -e "\nKilling jobs: \n$(jobs -p)"
	kill $(jobs -p)
        #echo -e $(pgrep iostat)
        #kill $(pgrep iostat)
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
while getopts ":holfvm:b:d:e:g:r:i:s:c:" opt; do
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
#LD_PRELOAD=/home1/09111/mbbm/monarch/monarch/pastor/build/libmonarch.so train-model
#LD_PRELOAD=/home1/09111/mbbm/tensorflow_scripts/scripts/monarch2/monarch/pastor/build/libmonarch.so train-model
train-model
echo "ELAPSED TIME: $SECONDS s" | tee -a $RUN_DIR/log.txt 
sleep 10

# Kill monitor process
kill-monitor
