import math

import tensorflow as tf


def get_rotate_func(max_deg: float = 15., upsample: float = 2, min_deg: float = 2):
    # Adapted from https://www.kaggle.com/cdeotte/rotation-augmentation-gpu-tpu-0-96
    # CONVERT DEGREES TO RADIANS
    def rotate_(tensor, label):
        deg = (
            max_deg * 2 *
            tf.random.uniform(
                [1],
                dtype='float32'
            )
        ) - max_deg
        if deg <= min_deg:
            return tf.cast(tensor, tf.uint8), label
        rotation = math.pi * deg / 180.
        c1 = tf.math.cos(rotation)
        s1 = tf.math.sin(rotation)
        rotation_matrix = tf.reshape(
            tf.concat([c1, s1, -1 * s1, c1], axis=0), [2, 2])
        height = tf.shape(tensor)[0]
        width = tf.shape(tensor)[1]
        if upsample > 1:
            original_height = height
            original_width = width
            tensor = tf.cast(tf.image.resize(
                tensor, (int(height * upsample), int(width * upsample)),
                method=tf.image.ResizeMethod.BILINEAR
            ), tf.uint8)
            height = tf.shape(tensor)[0]
            width = tf.shape(tensor)[1]
        height_is_odd = height % 2
        width_is_odd = width % 2
        height_half = height // 2
        width_half = width // 2

        # LIST DESTINATION PIXEL INDICES
        # rotate around the center of the image
        x = tf.tile(
            tf.range(
                -1 * width_half,
                width_half + width_is_odd
            ), [height])
        y = tf.repeat(tf.range(
            height_half + height_is_odd - 1,
            -1 * (height_half) - 1, -1
        ), width)

        idx = tf.stack([x, y])

        # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS
        idx2 = tf.matmul(rotation_matrix, tf.cast(idx, dtype='float32'))
        idx2 = tf.cast(tf.math.round(idx2), dtype='int32')

        # also discard 1 pixel at each edge
        within_bound = (
            (idx2[0, ] >= -1 * width_half + 1) &
            (idx2[0, ] < width_half + width_is_odd - 1) &
            (idx2[1, ] < height_half + height_is_odd - 1) &
            (idx2[1, ] >= -1 * height_half + 1)
        )
        new_x = tf.where(
            within_bound,
            idx2[0, ] + 1 + width_half,
            tf.zeros_like(idx2[0, ])
        )
        new_y = tf.where(
            within_bound,
            (height_half + height_is_odd - 1)-idx2[1, ],
            tf.zeros_like(idx2[1, ])
        )

        # FIND ORIGIN PIXEL VALUES
        idx3 = tf.stack([new_y, new_x], axis=1)
        gathered_tensor = tf.gather_nd(tf.pad(
            tensor, [[0, 0], [1, 0], [0, 0]], mode='CONSTANT', constant_values=0
        ), idx3)
        rotated_tensor = tf.reshape(gathered_tensor, tf.shape(tensor))
        if upsample > 1:
            rotated_tensor = tf.cast(tf.image.resize(
                tensor, (original_height, original_width),
                method=tf.image.ResizeMethod.AREA
            ), tf.uint8)
        return rotated_tensor, label
    return rotate_
