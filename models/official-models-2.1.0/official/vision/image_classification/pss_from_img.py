from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
import numpy as np

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

  mapping_path = '/scratch1/09111/mbbm/imagenet/LOC_synset_mapping.txt'

  # Creation of mapping dictionaries to obtain the image classes

  class_mapping_dict = {}
  class_mapping_dict_number = {}
  mapping_class_to_number = {}
  mapping_number_to_class = {}
  labels=[]
  i = 0
  for line in open(mapping_path):
      class_mapping_dict[line[:9].strip()] = line[9:].strip()
      class_mapping_dict_number[i] = line[9:].strip()
      mapping_class_to_number[line[:9].strip()] = i
      mapping_number_to_class[i] = line[:9].strip()
      labels.append(i)
      i+=1

  train_path = '/scratch1/09111/mbbm/imagenet/ILSVRC/Data/CLS-LOC/train/'

  # Creation of dataset_array and true_classes
  #true_classes = []
  #images_array = []
  #for train_class in os.listdir(train_path):
  #    i = 0
  #    for el in os.listdir(train_path + '/' + train_class):
  #        if i < 10:
  #            path = train_path + '/' + train_class + '/' + el
  #            image = tf.keras.utils.load_img(path,target_size=(224,224,3))
  #            image_array = tf.keras.utils.img_to_array(image).astype(np.uint8)
  #            images_array.append(image_array)
  #            true_class = class_mapping_dict[path.split('/')[-2]]
  #            true_classes.append(true_class)
  #            i+=1
  #        else:
  #            break
  #images_array = np.array(images_array)
  #true_classes = np.array(true_classes)
  
  #return images_array

  """
  train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./224,
        rotation_range=10,
        zoom_range=0.4,
        horizontal_flip=True,
        validation_split=0.01
        )

  train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(256, 256),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
        )
  
  return train_generator
  """


  dataset = tf.keras.preprocessing.image_dataset_from_directory(
    train_path,
    #labels=labels,  
    labels='inferred',
    #label_mode='int',
    label_mode='categorical',
    batch_size=batch_size,
    image_size=(224,224),
    shuffle=True)

  #dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=40, num_parallel_calls=tf.data.experimental.AUTOTUNE)

  dataset = dataset.shuffle(shuffle_buffer).repeat()
  dataset = dataset.map(lambda x,_: tf.reshape(x, [224, 224, 3]),
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
  #dataset = dataset.map(
  #      lambda value: imagenet_preprocessing.parse_record(value, is_training, dtype),
  #      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  dataset = dataset.batch(batch_size, drop_remainder=False)
  dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

  #dataset = tf.reshape(dataset, [224, 224, 3])
  #dataset.element_spec.shape = dataset.element_spec.shape[0][0, :, :, :, :]
  #print(dataset.element_spec.shape)
  #print("Dataset: ")
  #for element in dataset:
  #  print(element)

  #print(dataset.element_spec.shape)
  #dataset = dataset.map(lambda x: tf.reshape(dataset, [224, 224, 3]))

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
    model = alexnet_model.alexnet()
    #model = lenet_model.lenet()
    #model = resnet_model.resnet()
    #model = resnet_model.resnet50(num_classes=imagenet_preprocessing.NUM_CLASSES)
    #model  = tf.keras.applications.resnet.ResNet152(


#    model = tf.keras.applications.InceptionV3()
#       include_top=False,
#        weights=None,
#       classes=imagenet_preprocessing.NUM_CLASSES
#    )

    optimizer = adam_v2.Adam(learning_rate=lr_schedule, decay=lr_schedule/flags_obj.train_epochs)
    #optimizer = tf.keras.optimizers.legacy.SGD()
    #model.compile(optimizer, loss = "mse")
    model.compile(
    #loss='sparse_categorical_crossentropy',
    loss='categorical_crossentropy',
    optimizer=optimizer,
    metrics=(['sparse_categorical_accuracy']
              if flags_obj.report_accuracy_metrics else None)
    )


  steps_per_epoch=imagenet_preprocessing.NUM_IMAGES['train'] // flags_obj.batch_size
  dataset = tf.keras.utils.experimental.DatasetCreator(dataset_fn)
  #dataset = dataset_fn(steps_per_epoch)
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
