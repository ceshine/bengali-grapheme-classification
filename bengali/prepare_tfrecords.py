from glob import glob
from pathlib import Path

import cv2
import fire
import numpy as np
import pandas as pd
import tensorflow as tf


FLATTEN_IMAGE = [str(x) for x in range(32332)]
LABEL_COLUMNS = ("grapheme_root", "vowel_diacritic", "consonant_diacritic")


def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax


def crop_resize(img0, size=128, pad=16):
    img0 = img0.reshape(137, 236)
    # crop a box around pixels large than the threshold
    # some images contain line at the sides
    ymin, ymax, xmin, xmax = bbox(img0[5:-5, 5:-5] > 80)
    # cropping may cut too much, so we need to add it back
    xmin = xmin - 13 if (xmin > 13) else 0
    ymin = ymin - 10 if (ymin > 10) else 0
    xmax = xmax + 13 if (xmax < 236 - 13) else 236
    ymax = ymax + 10 if (ymax < 137 - 10) else 137
    img = img0[ymin:ymax, xmin:xmax]
    # remove lo intensity pixels as noise
    img[img < 28] = 0
    lx, ly = xmax-xmin, ymax-ymin
    l = max(lx, ly) + pad
    # make sure that the aspect ratio is kept in rescaling
    img = np.pad(img, [((l-ly)//2,), ((l-lx)//2,)], mode='constant')
    return cv2.resize(img, (size, size)).reshape(-1)


def to_example(img_values, labels):
    feature = {
        "image": tf.train.Feature(
            int64_list=tf.train.Int64List(value=img_values.tolist())
        ),
        "labels": tf.train.Feature(
            int64_list=tf.train.Int64List(value=labels.tolist())
        )
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def _write_tfrecords(buffer, n_shards, output_path):
    filepath = (
        output_path /
        "{:02d}-{}.tfrec".format(n_shards, len(buffer))
    )
    options = tf.io.TFRecordOptions(
        compression_type="GZIP"  # , compression_level=9
    )
    with tf.io.TFRecordWriter(str(filepath), options=options) as writer:
        for example in buffer:
            writer.write(example.SerializeToString())
    print("Wrote file {} containing {} records".format(
        filepath, len(buffer)))


def write_tfrecords(df: pd.DataFrame, pattern: str, output_path: Path, shard_size: int = 2048, crop: bool = False):
    print("Writing TFRecords")
    # Shuffle
    df = df.sample(frac=1)
    buffer_examples = []
    n_shards = 0
    def transform_func(x): return x
    if crop is True:
        transform_func = crop_resize
    for parquet_file in glob(pattern):
        print(f"Reading {parquet_file}")
        df_merged = pd.merge(
            pd.read_parquet(parquet_file),
            df, on='image_id', how="inner"
        ).drop(['image_id'], axis=1)
        df_image = df_merged.loc[:, FLATTEN_IMAGE]
        df_labels = df_merged.loc[:, LABEL_COLUMNS]
        for idx in range(df_merged.shape[0]):
            buffer_examples.append(
                to_example(
                    transform_func(
                        255 - df_image.iloc[idx].values
                    ),
                    df_labels.iloc[idx].values
                )
            )
            if len(buffer_examples) == shard_size:
                _write_tfrecords(
                    buffer_examples, n_shards, output_path
                )
                n_shards += 1
                buffer_examples = []
    if buffer_examples:
        _write_tfrecords(
            buffer_examples, n_shards, output_path
        )


def main(
        df_path: str = "data/train_split.csv",
        output_path: str = "data/tfrecords/train/",
        parquet_pattern: str = "data/train_image_data_*.parquet",
        shard_size: int = 10000):
    df = pd.read_csv(df_path)
    output_path_ = Path(output_path)
    output_path_.mkdir(parents=True, exist_ok=True)
    write_tfrecords(df, parquet_pattern, output_path_, shard_size)


if __name__ == '__main__':
    fire.Fire(main)
