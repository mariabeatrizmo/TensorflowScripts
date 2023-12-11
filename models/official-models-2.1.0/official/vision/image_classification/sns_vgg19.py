
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
from official.vision.image_classification import imagenet_preprocessing  #2 as imagenet_preprocessing
from official.vision.image_classification import alexnet_model
from official.vision.image_classification import lenet_model
from official.vision.image_classification import resnet_model
from official.vision.image_classification import inception_v4_model
from official.vision.image_classification import shufflenet_model
from keras.optimizers import adam_v2
from tensorflow.keras.layers import Input
from keras_resnet import models as resnet_models
from official.vision.image_classification import resnet18_model
from official.vision.image_classification import resnet18_model_2

DATASET_DIR='/scratch1/09111/mbbm/100g_tfrecords'
EPOCHS=2
BATCH_SIZE=1024
TASK_ID=4
NUM_WORKERS=4


def dataset_fn(input_context):
  global DATASET_DIR, EPOCHS, BATCH_SIZE, TASK_ID, NUM_WORKERS
  is_training = True
  data_dir = DATASET_DIR
  num_epochs = EPOCHS
  batch_size = BATCH_SIZE
  dtype = tf.float32
  shuffle_buffer = 10000
    
  filenames = imagenet_preprocessing.get_shuffled_filenames(is_training, data_dir, num_epochs)
  dataset = tf.data.Dataset.from_tensor_slices(filenames)
  dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=40, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  
  dataset = dataset.shard(
    input_context.num_input_pipelines, input_context.input_pipeline_id)

  dataset = dataset.shuffle(shuffle_buffer).repeat()
  dataset = dataset.map(
        lambda value: imagenet_preprocessing.parse_record(value, is_training, dtype),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
  dataset = dataset.batch(batch_size, drop_remainder=False)
  #dataset = dataset.shuffle(shuffle_buffer).repeat()
  dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

  #dist_dataset = parameter_server.experimental_distribute_dataset(dataset)

  #options = tf.data.Options()
  #options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
  #dataset = dataset.with_options(options)

  #ic = tf.distribute.InputContext(
  #  num_input_pipelines=NUM_WORKERS, input_pipeline_id=TASK_ID
  #)

  #print("INPUT_CONTEXT")
  #print(ic.num_input_pipelines)
  #print(ic.input_pipeline_id)

  dataset = dataset.shard(
    input_context.num_input_pipelines, input_context.input_pipeline_id)
  #  ic.num_input_pipelines, ic.input_pipeline_id)

  return dataset


def run(flags_obj):
  global DATASET_DIR, EPOCHS, BATCH_SIZE, TASK_ID, NUM_WORKERS
  DATASET_DIR = flags_obj.data_dir
  EPOCHS = flags_obj.train_epochs
  BATCH_SIZE = flags_obj.batch_size
  TASK_ID = flags_obj.task_index
  
  NUM_WORKERS = distribution_utils.configure_cluster(flags_obj.worker_hosts,
                                                     flags_obj.task_index)
  strategy = distribution_utils.get_distribution_strategy(
      distribution_strategy=flags_obj.distribution_strategy,
      num_gpus=flags_obj.num_gpus,
      num_workers=NUM_WORKERS,
      all_reduce_alg=flags_obj.all_reduce_alg,
      num_packs=flags_obj.num_packs,
      tpu_address=flags_obj.tpu)
  print("Cluster initialized")

  lr_schedule = 0.1
  with strategy.scope():
    #model = alexnet_model.alexnet()
    #model = lenet_model.lenet()
    #model = resnet_model.resnet50(
    #      num_classes=imagenet_preprocessing.NUM_CLASSES)
    #model = resnet_model.resnet50(num_classes=imagenet_preprocessing.NUM_CLASSES)

    #model = tf.keras.applications.resnet.ResNet152(
    #model = tf.keras.applications.ResNet50V2(
    #model = tf.keras.applications.InceptionV3(
    model = tf.keras.applications.VGG19(
	#include_top=False,
        weights=None,
        input_tensor=Input(shape=(224, 224, 3)),
	classes=600 #imagenet_preprocessing.NUM_CLASSES #600
    )

    #model = shufflenet_model.ShuffleNetV2(include_top=True, input_shape=(224, 224, 3), load_model=None, classes=imagenet_preprocessing.NUM_CLASSES)

    #model = resnet_models.ResNet18(include_top=False, inputs=Input(shape=(224, 224, 3)),  classes=imagenet_preprocessing.NUM_CLASSES)
    #model = resnet18_model.ResNet18(input_shape=Input(shape=(224, 224, 3)), classes=imagenet_preprocessing.NUM_CLASSES)

    #model = resnet18_model_2.ImageNetRN18()

#    model = inception_v4_model.create_model(
#	include_top=False,
#	num_classes=imagenet_preprocessing.NUM_CLASSES,
#        weights=None)

    optimizer = adam_v2.Adam(learning_rate=lr_schedule, decay=lr_schedule/flags_obj.train_epochs)
    #optimizer = tf.keras.optimizers.legacy.SGD()
    #model.compile(optimizer, loss = "mse")
    model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=optimizer,
    metrics=(['sparse_categorical_accuracy']
              if flags_obj.report_accuracy_metrics else None)
    )
  

  steps_per_epoch=imagenet_preprocessing.NUM_IMAGES['train'] // flags_obj.batch_size
  #dataset = tf.keras.utils.experimental.DatasetCreator(dataset_fn)

  dataset = imagenet_preprocessing.input_fn(
      is_training=True,
      data_dir=flags_obj.data_dir,
      batch_size=flags_obj.batch_size,
      num_epochs=flags_obj.train_epochs,
      parse_record_fn=imagenet_preprocessing.parse_record,
      datasets_num_private_threads=flags_obj.datasets_num_private_threads,
      dtype=tf.float16,
      drop_remainder=flags_obj.enable_xla,
      tf_data_experimental_slack=flags_obj.tf_data_experimental_slack,
      training_dataset_cache=flags_obj.training_dataset_cache,
  )

  model.fit(dataset, epochs=flags_obj.train_epochs, steps_per_epoch=steps_per_epoch, verbose=2)

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

