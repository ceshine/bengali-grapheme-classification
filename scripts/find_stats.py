import re

import numpy as np
import tensorflow as tf

from bengali.dataset import IMAGE_SIZE

AUTOTUNE = tf.data.experimental.AUTOTUNE


def tfrecord_dataset(filenames, batch_size):
    opt = tf.data.Options()
    opt.experimental_deterministic = False

    features_description = {
        "image": tf.io.FixedLenFeature([IMAGE_SIZE[0] * IMAGE_SIZE[1]], tf.int64),
        "labels": tf.io.FixedLenFeature([3], tf.int64),
    }

    def _parse_function(example_proto):
        # Parse the input `tf.Example` proto using the dictionary above.
        example = tf.io.parse_single_example(
            example_proto, features_description)
        image = tf.reshape(example["image"], [IMAGE_SIZE[0], IMAGE_SIZE[1], 1])
        return image

    def _normalize(image):
        # Normalization
        image = tf.cast(image, tf.float32) / 255.
        return image

    raw_dataset = tf.data.TFRecordDataset(
        filenames, num_parallel_reads=1, compression_type="GZIP"
    ).with_options(opt).shuffle(
        128, reshuffle_each_iteration=True
    )
    dataset = raw_dataset.map(_parse_function)

    dataset = dataset.map(
        _normalize, num_parallel_calls=AUTOTUNE
    ).batch(
        batch_size
    )
    return dataset


def main():
    dataset = tfrecord_dataset(
        "data/tfrecords/train/01-10000.tfrec", batch_size=5000)
    for tensor in dataset:
        arr = tensor.numpy()
        print(arr.shape)
        print(np.mean(arr), np.std(arr))


if __name__ == "__main__":
    main()
