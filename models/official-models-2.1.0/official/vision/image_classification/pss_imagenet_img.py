from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json

import numpy as np
from absl import app
from absl import flags
import tensorflow as tf
from absl import logging
import tensorflow_datasets as tfds

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
from tensorflow.keras.layers import Input

DATASET_DIR='/scratch1/09111/mbbm/100g_tfrecords'
EPOCHS=2
BATCH_SIZE=1024
strategy = None


def normalize_img(item):
    """Puts the mnist dataset in the format Keras expects, (features, labels)."""
    image = item['image']
    label = item['label']
    image = tf.dtypes.cast(image, 'float32') / 255.
    return image, label

def map_img(image, a):
  input_image = tf.image.resize(image, (224, 224), method='nearest')
  input_image = tf.math.reduce_max(input_image, axis=-1, keepdims=True)
  input_image = tf.concat([input_image, input_image, input_image], axis=-1) 

  return input_image, a

def dataset_fn(input_context):
    global BATCH_SIZE, STRATEGY
    batch_size = BATCH_SIZE
    #tfds.even_splits('train', n=2)
    #"/scratch1/09111/mbbm/imagenet/ILSVRC/Data/CLS-LOC/"
    
    builder = tfds.ImageFolder('/scratch1/09111/mbbm/imagenet/ILSVRC/Data/CLS-LOC/')
    print(builder.info)
    ds_train = builder.as_dataset(split='train', shuffle_files=True)
    #tfds.show_examples(ds, builder.info)

    ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)

    #tf.image.resize(ds_train.element_spec[1], (224,224,3)).numpy()
    #ds_train = reshape(ds_train.shape[0], 224, 224, 3)
    #ds_train = ds_train.map(lambda image,_: tf.reshape(image, [224, 224,3]))
    #ds_train = ds_train.map(lambda image,_: map_img(image))
    #ds_train = ds_train.map(lambda image,_: tf.image.resize(image, [224, 224]))
    #ds_train = ds_train.map(lambda image: image=tf.set_shape(image, [224,224,3]))

    ds_train = ds_train.map(
    map_img, num_parallel_calls=tf.data.AUTOTUNE)

    ds_train = ds_train.shard(
    input_context.num_input_pipelines, input_context.input_pipeline_id)
    #ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(10000).repeat()
    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    #train_ds = STRATEGY.experimental_distribute_dataset(ds_train)
    print("\n\n\n\n\n\nDataset:")
    print(ds_train)   
    
    return ds_train

  

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

    return tf.distribute.experimental.ParameterServerStrategy(cluster_resolver) 

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
            if task_index == 0:
                  worker_type = "chief"
            elif task_index == 1:
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
  global DATASET_DIR, EPOCHS, BATCH_SIZE, STRATEGY
  DATASET_DIR = flags_obj.data_dir
  EPOCHS = flags_obj.train_epochs
  BATCH_SIZE = flags_obj.batch_size

  configure_cluster(flags_obj.worker_hosts,
                    flags_obj.task_index,
                    distribution_strategy=flags_obj.distribution_strategy)
  strategy = get_distribution_strategy(
      distribution_strategy=flags_obj.distribution_strategy,
      num_gpus=flags_obj.num_gpus)
  print("Cluster initialized")

  STRATEGY=strategy
  lr_schedule = 0.1
  # if flags_obj.use_tensor_lr:
  if False:
    lr_schedule = common.PiecewiseConstantDecayWithWarmup(
        batch_size=flags_obj.batch_size,
        epoch_size=imagenet_preprocessing.NUM_IMAGES['train'],
        warmup_epochs=common.LR_SCHEDULE[0][1],
        boundaries=list(p[1] for p in common.LR_SCHEDULE[1:]),
        multipliers=list(p[0] for p in common.LR_SCHEDULE),
        compute_lr_on_cpu=True)

  with strategy.scope():
    #model = alexnet_model.alexnet()
    #model = lenet_model.lenet()
    #model = resnet_model.resnet()
    #model = resnet_model.resnet50(num_classes=imagenet_preprocessing.NUM_CLASSES)
    #model  = tf.keras.applications.resnet.ResNet152(


    model = tf.keras.applications.InceptionV3(
        #include_top=False,
        weights=None,
        input_tensor=Input(shape=(224, 224, 3)),
        classes=imagenet_preprocessing.NUM_CLASSES
    )

    optimizer = adam_v2.Adam(learning_rate=lr_schedule, decay=lr_schedule/flags_obj.train_epochs)
    #optimizer = tf.keras.optimizers.legacy.SGD()
    #model.compile(optimizer, loss = "mse")
    model.compile(
    loss='sparse_categorical_crossentropy',
    #loss='categorical_crossentropy',
    optimizer=optimizer,
    metrics=(['sparse_categorical_accuracy']
              if flags_obj.report_accuracy_metrics else None)
    )


  steps_per_epoch=imagenet_preprocessing.NUM_IMAGES['train'] // flags_obj.batch_size
  dataset = tf.keras.utils.experimental.DatasetCreator(dataset_fn)
  #dataset = strategy.experimental_distribute_dataset(dataset_fn)
  print("Pre fit")
  model.fit(dataset, epochs=flags_obj.train_epochs, steps_per_epoch=steps_per_epoch)

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
