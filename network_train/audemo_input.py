import argparse
import os.path

import tensorflow as tf

NUM_DATA_BATCH = 8
NUM_TEST_BATCH = 2
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 2000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 560

IMAGE_HEIGHT = 128
IMAGE_WIDTH = 1000
IMAGE_DEPTH = 3
IMAGE_PIXELS = IMAGE_HEIGHT * IMAGE_WIDTH

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir', 'audemo_data/tf_db/', 'Path to data directory.')

# Decode the serialized example into TFRecord format
def decode(serialized_example):
    """Parses an image and label from the given `serialized_example`."""
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'image/encoded': tf.FixedLenFeature([], tf.string),
            'image/format': tf.FixedLenFeature([], tf.string),
            'image/class/label': tf.FixedLenFeature([], tf.int64),
            'image/height': tf.FixedLenFeature([], tf.int64),
            'image/width': tf.FixedLenFeature([], tf.int64),
            'image/depth': tf.FixedLenFeature([], tf.int64)
        })

    height = tf.cast(features['image/height'], tf.int32)
    width = tf.cast(features['image/width'], tf.int32)
    depth = tf.cast(features['image/depth'], tf.int32)

    # Convert from a scalar string tensor (whose single string has
    # length IMAGE.HEIGHT * IMAGE.WIDTH * IMAGE.DEPTH) to a uint8 tensor with shape
    # [IMAGE.HEIGHT, IMAGE.WIDTH, IMAGE.DEPTHS].

    image = tf.decode_raw(features['image/encoded'], tf.uint8)
    image_shape = tf.stack([height, width, depth])
    image = tf.cast(tf.reshape(image, image_shape), tf.float32)

    # Convert label from a scalar uint8 tensor to an int32 scalar.
    label = tf.cast(features['image/class/label'], tf.int32)

    return image, label

# Randomly crop the input images
def crop(image, label):
    crop_image = tf.random_crop(image, [IMAGE_HEIGHT, int(IMAGE_WIDTH*0.8), IMAGE_DEPTH])
    # crop_image = image[:,:500,:]
    return crop_image, label


# Dataset augmentation
def augment(image, label):
    # Random flip image to the left or right
    distorted_image = tf.image.random_flip_left_right(image)

    # Random adjust the brightness of the image
    distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)

    # Random adjust contrast of the image
    distorted_image = tf.image.random_contrast(distorted_image, lower=0.5, upper=1.5)

    return distorted_image, label


# For each image perform normalization on it
def normalize(image, label):
    float_image = tf.image.per_image_standardization(image)
    return float_image, label


def inputs(is_train, batch_size, num_epochs, is_augment=False):
    # Input pipeline definitions

    # Filename array
    if not num_epochs:
        num_epochs = None
    if is_train:
        filenames = [os.path.join(FLAGS.data_dir, 'emodb_train_%02d.tfrecord' % i)
                     for i in xrange(NUM_DATA_BATCH)]
    else:
        filenames = [os.path.join(FLAGS.data_dir, 'emodb_validation_%02d.tfrecord' % i)
                     for i in xrange(NUM_TEST_BATCH)]

    with tf.name_scope('input'):
        # Filename pipeline
        files = tf.data.Dataset.list_files(filenames)
        dataset = files.apply(tf.contrib.data.parallel_interleave(tf.data.TFRecordDataset, cycle_length=8))

        # Apply the following operaions on each data point
        dataset = dataset.map(decode, num_parallel_calls=12)
        dataset = dataset.map(crop, num_parallel_calls=12)
        if is_augment:
            dataset = dataset.map(augment, num_parallel_calls=12)
        dataset = dataset.map(normalize, num_parallel_calls=12)

        # The shuffle transformation uses a finite-sized buffer to shuffle elements
        # in memory. The parameter is the number of elements in the buffer. For
        # completely uniform shuffling, set the parameter to be the same as the
        # number of elements in the dataset.
        # if is_train: dataset = dataset.shuffle(3000 + 3 * batch_size)

        # Batch the dataset and shuffle them
        dataset = dataset.repeat(num_epochs)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=15)

        # Define iterator to fetch next batch of dataset
        iterator = dataset.make_one_shot_iterator()

    return iterator.get_next()
