from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf

from official.benchmark.models import trivial_model
from official.utils.flags import core as flags_core
from official.utils.logs import logger
from official.utils.misc import distribution_utils
from official.utils.misc import keras_utils
from official.utils.misc import model_helpers
from official.vision.image_classification import common
from official.vision.image_classification import imagenet_preprocessing
from official.vision.image_classification import alexnet_model
from official.vision.image_classification import lenet_model
from official.vision.image_classification import resnet_model
from keras.optimizers import adam_v2

DATASET_DIR='/scratch1/09111/mbbm/100g_tfrecords'
EPOCHS=2
BATCH_SIZE=1024



def dataset_fn(_):
  is_training = True
  data_dir = DATASET_DIR
  num_epochs = EPOCHS
  batch_size = BATCH_SIZE
  dtype = tf.float32
  shuffle_buffer = 10000

  filenames = imagenet_preprocessing.get_shuffled_filenames(is_training, data_dir, num_epochs)
  dataset = tf.data.Dataset.from_tensor_slices(filenames)
  dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=40, num_parallel_calls=tf.data.experimental.AUTOTUNE)

  dataset = dataset.shuffle(shuffle_buffer).repeat()
  dataset = dataset.map(
        lambda value: imagenet_preprocessing.parse_record(value, is_training, dtype),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
  dataset = dataset.batch(batch_size, drop_remainder=False)
  dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

  return dataset

def get_distribution_strategy(distribution_strategy="parameter_server",
                              num_gpus=0):
  if distribution_strategy == "parameter_server":
    cluster_resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
    if cluster_resolver.task_type in ("worker", "ps"):
      # Start a TensorFlow server and wait.
      os.environ["GRPC_FAIL_FAST"] = "use_caller"

      server = tf.distribute.Server(
          cluster_resolver.cluster_spec(),
          job_name=cluster_resolver.task_type,
          task_index=cluster_resolver.task_id,
          protocol=cluster_resolver.rpc_layer or "grpc",
          start=True)
      server.join()

    #NUM_PS=1
    #variable_partitioner = (
    #    tf.distribute.experimental.partitioners.MinSizePartitioner(
    #    min_shard_bytes=(256 << 10),
    #    max_shards=NUM_PS))

    return tf.distribute.experimental.ParameterServerStrategy(cluster_resolver) #,
#      variable_partitioner=None)

def configure_cluster(worker_hosts=None, task_index=-1, distribution_strategy="parameter_server"):
  """Set multi-worker cluster spec in TF_CONFIG environment variable.
  Args:
    worker_hosts: comma-separated list of worker ip:port pairs.
  Returns:
    Number of workers in the cluster.
  """
  tf_config = json.loads(os.environ.get('TF_CONFIG', '{}'))
  if tf_config:
    num_workers = (len(tf_config['cluster'].get('chief', [])) +
                   len(tf_config['cluster'].get('ps', [])) +
                   len(tf_config['cluster'].get('worker', [])))
  elif worker_hosts:
    workers = worker_hosts.split(',')
    num_workers = len(workers)
    if num_workers > 1 and task_index < 0:
      raise ValueError('Must specify task_index when number of workers > 1')
    if distribution_strategy=="parameter_server":
      if num_workers < 3:
            raise ValueError('Must have at least a chief, a worker and a ps.')
      else:
            if task_index is 0:
                  worker_type = "chief"
            elif task_index is 1:
                  worker_type = "ps"
                  task_index -= 1
            else:
                  worker_type = "worker"
                  task_index-=2

            os.environ["TF_CONFIG"] = json.dumps({
                  "cluster": {
                        "worker": workers[2:],
                        "ps": [workers[1]],
                        "chief": [workers[0]]
                  },
                  "task": {"type": worker_type, "index": task_index}
            })
    else:
      task_index = 0 if num_workers == 1 else task_index
      os.environ['TF_CONFIG'] = json.dumps({
            'cluster': {
                  'worker': workers
            },
            'task': {'type': 'worker', 'index': task_index}
      })
  else:
    num_workers = 1
  return num_workers

def run(flags_obj):
  data_dir = flags_obj.data_dir
  DATASET_DIR = flags_obj.data_dir
  EPOCHS = flags_obj.train_epochs
  batch_size = flags_obj.batch_size
  BATCH_SIZE = flags_obj.batch_size

  num_workers = configure_cluster(flags_obj.worker_hosts,
                    flags_obj.task_index,
                    distribution_strategy=flags_obj.distribution_strategy)
  strategy = get_distribution_strategy(
      distribution_strategy=flags_obj.distribution_strategy,
      num_gpus=flags_obj.num_gpus)
  print("Cluster initialized")

  global_batch_size = batch_size * num_workers

  #load dataset-----------------------------

  # Define the directory where the Imagenet dataset is stored
  #data_dir=r'home/nfs-dts/public/100g_tfrecords/'

  # Define the file patterns for the training and validation data
  train_files = os.path.join(data_dir, 'train-0')
  val_files = os.path.join(data_dir, 'validation-')

  # Define the number of classes in the dataset
  num_classes = 1000

  #Preprocess for Resnet 50
  def preprocess_image(record):
    # Define the feature description for decoding the record
    feature_description = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/class/label': tf.io.FixedLenFeature([], tf.int64),
    }
    
    # Decode the record using the feature description
    example = tf.io.parse_single_example(record, feature_description)
    image = tf.image.decode_jpeg(example['image/encoded'], channels=3)
    label = tf.one_hot(example['image/class/label'], depth=1000)
    
    # Resize the image
    image = tf.image.resize(image, [224, 224])
    
    # Flip the image horizontally with a 50% chance
    image = tf.image.random_flip_left_right(image)
    
    # Convert the pixel values to the range [0, 1]
    image = tf.cast(image, tf.float32) / 255.0
    
    # Normalize the pixel values to have zero mean and unit variance
    mean = tf.constant([0.485, 0.456, 0.406])
    std = tf.constant([0.229, 0.224, 0.225])
    image = (image - mean) / std
    
    return image, label 


  # Create a dataset of training records and apply preprocessing
  #train_dataset = tf.data.Dataset.list_files(train_files)
  #train_dataset = train_dataset.interleave(
  #    lambda filename: tf.data.TFRecordDataset(filename),
  #    num_parallel_calls=tf.data.AUTOTUNE,
  #    deterministic=False)
  #train_dataset = train_dataset.shuffle(buffer_size=10000)
  #train_dataset = train_dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
  #train_dataset = train_dataset.batch(global_batch_size)
  #train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

  # Create a dataset of validation records and apply preprocessing
  #val_dataset = tf.data.Dataset.list_files(val_files)
  #val_dataset = val_dataset.interleave(
  #    lambda filename: tf.data.TFRecordDataset(filename),
  #    num_parallel_calls=tf.data.AUTOTUNE,
  #    deterministic=False)
  #val_dataset = val_dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
  #val_dataset = val_dataset.batch(batch_size)
  #val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

  train_dataset = tf.keras.utils.experimental.DatasetCreator(dataset_fn)

  global_batch_size = batch_size * num_workers

  #Inception V3
  """def build_and_compile_cnn_model():
      model = tf.keras.applications.inception_v3.InceptionV3(
              include_top=True,
              weights='imagenet',
              input_tensor=None,
              input_shape=None,
              pooling=None,
              classes=1000,
              classifier_activation='softmax')  
      model.compile(loss=tf.keras.losses.f(from_logits=False),
              optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
              metrics=['accuracy','mse'])
      return model
  """

  #Resnet50
  def build_and_compile_cnn_model():
      model = tf.keras.applications.resnet_v2.ResNet50V2(
          include_top=True,
          weights='imagenet',
          input_tensor=None,
          input_shape=None,
          pooling=None,
          classes=1000,
          classifier_activation='softmax'
      )
      model.compile(
          loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
          optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
          metrics=['accuracy', 'mse']
      )
      return model

  print("checkpoint")
  with strategy.scope():
    # Model building/compiling need to be within `strategy.scope()`.
    model = build_and_compile_cnn_model()
  print('ola')


  # Keras' `model.fit()` trains the model with specified number of epochs and
  # number of steps per epoch. Note that the numbers here are for demonstration
  # purposes only and may not sufficiently produce a model with good quality.
  model.fit(train_dataset, epochs=EPOCHS, steps_per_epoch=100)

  options = tf.data.Options()
  options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
  #dataset_no_auto_shard = train_dataset.with_options(options)
  #model_path = '/tmp/keras-model'

  def _is_chief(task_type):
    # assume que o primeiro worker, index 0, Ã© o chief
    return task_type is None or task_type == 'chief'

  def _get_temp_dir(dirpath, task_id):
    base_dirpath = 'workertemp_' + str(task_id)
    temp_dir = os.path.join(dirpath, base_dirpath)
    tf.io.gfile.makedirs(temp_dir)
    return temp_dir

  def write_filepath(filepath, task_type, task_id):
    dirpath = os.path.dirname(filepath)
    base = os.path.basename(filepath)
    if not _is_chief(task_type, task_id):
      dirpath = _get_temp_dir(dirpath, task_id)
    return os.path.join(dirpath, base)

  task_type, task_id = (strategy.cluster_resolver.task_type,
                        strategy.cluster_resolver.task_id)

  # Now that we have the model restored, and can continue with the training.

  checkpoint_dir = '/tmp/ckpt'

  checkpoint = tf.train.Checkpoint(model=model)
  write_checkpoint_dir = write_filepath(checkpoint_dir, task_type, task_id)
  checkpoint_manager = tf.train.CheckpointManager(
    checkpoint, directory=write_checkpoint_dir, max_to_keep=1)

  checkpoint_manager.save()
  if not _is_chief(task_type, task_id):
    tf.io.gfile.rmtree(write_checkpoint_dir)
  latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
  checkpoint.restore(latest_checkpoint)
  model.fit(train_dataset, epochs=EPOCHS, steps_per_epoch=20)
  callbacks = [tf.keras.callbacks.experimental.BackupAndRestore(backup_dir='/tmp/backup')]
  with strategy.scope():
    model = build_and_compile_cnn_model()
  model.fit(train_dataset,
            epochs=EPOCHS,
            steps_per_epoch=100,
            callbacks=callbacks)
  return

def define_imagenet_keras_flags():
  common.define_keras_flags()
  flags_core.set_defaults()
  flags.adopt_module_key_flags(common)


def main(_):
  model_helpers.apply_clean(flags.FLAGS)
  with logger.benchmark_context(flags.FLAGS):
    stats = run(flags.FLAGS)
  logging.info('Run stats:\n%s', stats)


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  define_imagenet_keras_flags()
  app.run(main)
