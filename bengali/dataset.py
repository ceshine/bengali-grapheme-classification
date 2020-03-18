import re
from typing import Tuple, Optional
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from .augmentations import get_rotate_func

IMAGE_SIZE = (137, 236, 3)
CLASS_COUNTS = [168, 11, 7]
SPLIT_POINT = [
    0, CLASS_COUNTS[0], CLASS_COUNTS[0] + CLASS_COUNTS[1], sum(CLASS_COUNTS)
]
AUTOTUNE = tf.data.experimental.AUTOTUNE
CENTER = 0.053
SCALE = 0.165
DEBUG = False


def count_data_items(filenames):
    # trick: the number of data items is written in the name of
    # the .tfrec files a flowers00-230.tfrec = 230 data items
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1))
         for filename in filenames]
    return np.sum(n)


def mixup_augment(alpha: float):
    """
       Adapted from:
       https://github.com/tensorpack/tensorpack/blob/master/examples/ResNet/cifar10-preact18-mixup.py
    """
    dist = tfp.distributions.Beta(alpha, alpha)

    def _mixup_augment(images, labels):
        batch_size = tf.shape(images)[0]
        lambd = dist.sample([batch_size])
        lambd = tf.math.reduce_max(
            tf.stack([lambd, 1-lambd]), axis=0
        )
        lambd = tf.reshape(lambd, [batch_size, 1, 1, 1])
        index = tf.random.shuffle(tf.range(batch_size))
        new_images = images * lambd + tf.gather(images, index) * (1 - lambd)
        return new_images, {"labels_1": labels, "labels_2": tf.gather(labels, index), "lambd": lambd[:, 0, 0, 0]}
    return _mixup_augment


def get_cutmix_func(alpha: float):
    dist = tfp.distributions.Beta(alpha, alpha)

    def cutmix_(images, labels):
        batch_size = tf.shape(images)[0]
        image_height = tf.shape(images)[1]
        image_width = tf.shape(images)[2]
        index = tf.random.shuffle(tf.range(batch_size))

        def sample_func(lambda_):
            cutout_center_height = tf.random.uniform(
                shape=[], minval=0, maxval=image_height,
                dtype=tf.int32)

            cutout_center_width = tf.random.uniform(
                shape=[], minval=0, maxval=image_width,
                dtype=tf.int32)

            mask_width = tf.math.round(
                tf.cast(image_width, tf.float32) * tf.math.sqrt(1 - lambda_)
            )
            mask_height = tf.math.round(
                tf.cast(image_height, tf.float32) * tf.math.sqrt(1 - lambda_)
            )

            lower_pad = tf.maximum(
                0, cutout_center_height -
                tf.cast(tf.math.floor(mask_height / 2.), tf.int32)
            )
            upper_pad = tf.maximum(
                0, image_height - cutout_center_height -
                tf.cast(tf.math.ceil(mask_height / 2.), tf.int32)
            )
            left_pad = tf.maximum(
                0, cutout_center_width -
                tf.cast(tf.math.floor(mask_width / 2.), tf.int32)
            )
            right_pad = tf.maximum(
                0, image_width - cutout_center_width -
                tf.cast(tf.math.ceil(mask_width / 2.), tf.int32)
            )

            cutout_shape = [image_height - (lower_pad + upper_pad),
                            image_width - (left_pad + right_pad)]
            padding_dims = [[lower_pad, upper_pad], [left_pad, right_pad]]
            mask = tf.pad(
                tf.zeros(cutout_shape, dtype=images.dtype),
                padding_dims, constant_values=1
            )
            mask = tf.expand_dims(mask, -1)
            lambda_adj = (
                1 -
                tf.cast(cutout_shape[0] * cutout_shape[1], tf.float32) /
                tf.cast(image_height * image_width, tf.float32)
            )
            return {"mask": mask, "lambda": lambda_adj}

        lambdas = dist.sample([batch_size])
        lambdas = tf.math.reduce_max(
            tf.stack([lambdas, 1-lambdas]), axis=0
        )
        masks = tf.map_fn(
            sample_func, lambdas,
            dtype={"mask": images.dtype, "lambda": tf.float32}
        )
        if DEBUG:
            return (
                tf.zeros_like(images) * masks["mask"] +
                tf.ones_like(images) * (1-masks["mask"]) * 0.5,
                {
                    "labels_1": labels,
                    "labels_2": tf.gather(labels, index),
                    "lambd": masks["lambda"]
                }
            )
        return (
            images * masks["mask"] +
            tf.gather(images, index) * (1-masks["mask"]),
            {
                "labels_1": labels,
                "labels_2": tf.gather(labels, index),
                "lambd": masks["lambda"]
            }
        )
    return cutmix_


def tfrecord_dataset(
    filenames, batch_size, strategy, is_train: bool = True,
    resize: Optional[Tuple[int, int]] = None,
    mixup_alpha: float = -1, cutmix_alpha: float = -1,
    cutmix_ratio: float = 0.5, max_rotate_deg: float = 0.
):
    opt = tf.data.Options()
    opt.experimental_deterministic = False

    features_description = {
        "image": tf.io.FixedLenFeature([IMAGE_SIZE[0] * IMAGE_SIZE[1]], tf.int64),
        "labels": tf.io.FixedLenFeature([3], tf.int64),
    }

    offset = tf.constant([CENTER] * 3, shape=[1, 1, 3])
    scale = tf.constant([SCALE] * 3, shape=[1, 1, 3])

    def _parse_function(example_proto):
        # Parse the input `tf.Example` proto using the dictionary above.
        example = tf.io.parse_single_example(
            example_proto, features_description)
        image = tf.reshape(example["image"], [IMAGE_SIZE[0], IMAGE_SIZE[1], 1])
        image = tf.tile(image, tf.constant([1, 1, 3]))
        label = tf.cast(example["labels"], tf.int32)
        return image, label

    def _normalize(images, label):
        # Normalization
        images = tf.cast(images, tf.float32) / 255.
        images -= offset
        images /= scale
        return images, label

    def _resize(images, labels):
        images = tf.image.resize(
            images, resize, method=tf.image.ResizeMethod.BICUBIC  # LANCZOS3
        )
        return images, labels

    def _data_augment(images, labels):
        # image = tf.image.random_flip_left_right(image)
        images = tf.image.random_saturation(images, 0, 2)
        return images, labels

    raw_dataset = tf.data.TFRecordDataset(
        filenames, num_parallel_reads=8, compression_type="GZIP"
    ).with_options(opt).shuffle(
        32, reshuffle_each_iteration=True
    )
    dataset = raw_dataset.map(_parse_function)
    if is_train:
        dataset = dataset.shuffle(
            4096, reshuffle_each_iteration=True
        ).repeat()

    if is_train:
        rand = None
        if mixup_alpha > 0 and cutmix_alpha > 0:
            rand = tf.random.uniform(
                shape=[], minval=0., maxval=1., dtype=tf.float32)
        # if max_rotate_deg > 0:
        #     # Only rotate in mixup
        #     # if (rand is None and mixup_alpha > 0) or (rand is not None and rand > cutmix_ratio):
        #     if True:
        #         dataset = dataset.map(
        #             get_rotate_func(max_rotate_deg),
        #             num_parallel_calls=AUTOTUNE
        #         )
    else:
        # usually fewer validation files than workers so disable FILE auto-sharding on validation
        # option not useful if there is no sharding (not harmful either)
        if strategy.num_replicas_in_sync > 1:
            opt = tf.data.Options()
            opt.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
            dataset = dataset.with_options(opt)

    if resize is not None:
        dataset = dataset.map(_resize, num_parallel_calls=AUTOTUNE)

    dataset = dataset.batch(
        batch_size,
        drop_remainder=is_train
    )
    dataset = dataset.map(
        _normalize, num_parallel_calls=AUTOTUNE
    )
    if is_train:
        mixup_func = None
        if mixup_alpha > 0:
            mixup_func = mixup_augment(mixup_alpha)
        if cutmix_alpha > 0:
            cutmix_func = get_cutmix_func(cutmix_alpha)
            if mixup_func:
                assert rand is not None
                if rand > cutmix_ratio:
                    dataset = dataset.map(
                        mixup_func, num_parallel_calls=AUTOTUNE)
                else:
                    dataset = dataset.map(
                        cutmix_func, num_parallel_calls=AUTOTUNE)
            else:
                print("Warning: only CutMix is enabled because mixup_alpha < 0")
                dataset = dataset.map(cutmix_func, num_parallel_calls=AUTOTUNE)
        elif mixup_func:
            dataset = dataset.map(mixup_func, num_parallel_calls=AUTOTUNE)
    dataset = dataset.prefetch(AUTOTUNE)
    n = count_data_items(filenames)
    return dataset, int(np.ceil(n / batch_size))
