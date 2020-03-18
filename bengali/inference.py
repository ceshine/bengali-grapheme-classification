from glob import glob
from pathlib import Path
from typing import Optional, Tuple

import fire
import numpy as np
import pandas as pd
import tensorflow as tf

from .model import get_model
from .dataset import SPLIT_POINT, CENTER, SCALE
from .prepare_tfrecords import FLATTEN_IMAGE, crop_resize

INPUT_IMAGE_SIZE = (137, 236, 1)
AUTOTUNE = tf.data.experimental.AUTOTUNE


def get_dataset(df, batch_size, resize: Optional[Tuple[int, int]] = None):
    dataset = tf.data.Dataset.from_tensor_slices(
        255 - df.loc[:, FLATTEN_IMAGE].values)

    offset = tf.constant([CENTER] * 3, shape=[1, 1, 3])
    scale = tf.constant([SCALE] * 3, shape=[1, 1, 3])

    def _tile(images):
        images = tf.reshape(images, [tf.shape(images)[0], *INPUT_IMAGE_SIZE])
        images = tf.tile(images, tf.constant([1, 1, 1, 3]))
        return images

    def _normalize(images):
        # Normalization
        images = tf.cast(images, tf.float32) / 255.
        images -= offset
        images /= scale
        return images

    def _resize(images):
        images = tf.image.resize(
            images, resize, method=tf.image.ResizeMethod.BICUBIC
        )
        return images

    dataset = dataset.batch(batch_size).map(_tile, num_parallel_calls=AUTOTUNE)

    if resize:
        dataset = dataset.map(_resize, num_parallel_calls=AUTOTUNE)

    dataset = dataset.map(
        _normalize, num_parallel_calls=AUTOTUNE
    )
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset


def extract_predictions(output_array):
    tmp = []
    for i in range(3):
        tmp.append(tf.argmax(
            output_array[:, SPLIT_POINT[i]:SPLIT_POINT[i+1]],
            axis=1
        ))
    tmp = tf.stack(tmp, axis=1)
    return tf.reshape(tmp, [-1])


@tf.function
def ensemble_predictions(input_images, models, method="arithmetic"):
    outputs = []
    for model in models:
        outputs.append(model(input_images, training=False))
    stacked = tf.stack(outputs)
    if method == "harmonic":
        results = 1. / tf.reduce_mean(1. / stacked, axis=0)
    elif method == "geometric":
        results = tf.math.exp(tf.reduce_mean(tf.math.log(stacked), axis=0))
    elif method == "arithmetic":
        results = tf.reduce_mean(stacked, axis=0)
    else:
        raise ValueError("averaging method unknown!")
    return extract_predictions(results)


def main(
    model_paths="cache/b2.h5",
    parquet_pattern: str = "data/test_image_data_*.parquet",
    batch_size: int = 128,
    resize: Optional[Tuple[int, int]] = None,
    method: str = "arithmetic"
):
    models = []
    for model_path in model_paths.split(","):
        model = get_model(
            Path(model_path).stem.split("_")[0], pretrained=None,
            image_size=(
                (INPUT_IMAGE_SIZE[0], INPUT_IMAGE_SIZE[1], 3)
                if resize is None else (*resize, 3)
            )
        )
        model.load_weights(model_path)
        models.append(model)
    predictions, ids = [], []
    for parquet_file in glob(parquet_pattern):
        print(f"Reading {parquet_file}")
        df = pd.read_parquet(parquet_file)
        dataset = get_dataset(df, batch_size, resize)
        for name in df["image_id"].values:
            ids.extend([
                f'{name}_grapheme_root',
                f'{name}_vowel_diacritic',
                f'{name}_consonant_diacritic'
            ])
        del df
        for images in dataset:
            predictions.append(ensemble_predictions(
                images, models, method).numpy())
        del dataset
    del models
    predictions = np.concatenate(predictions)
    df_sub = pd.DataFrame({
        'row_id': ids,
        'target': predictions
    })
    df_sub.to_csv("submission.csv", index=False)


if __name__ == '__main__':
    fire.Fire(main)
