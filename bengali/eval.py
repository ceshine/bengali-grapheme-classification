from glob import glob
from pathlib import Path
from typing import Optional, Tuple

import typer
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as amp

from .model import get_model
from .metrics import MacroAveragedRecall
from .prepare_tfrecords import FLATTEN_IMAGE, LABEL_COLUMNS
from .inference import get_dataset, INPUT_IMAGE_SIZE


@tf.function
def predict(model, images):
    return model(images, training=False)


def main(
    df_path="data/valid_split.csv",
    model_path="cache/b2.h5",
    parquet_pattern: str = "data/train_image_data_*.parquet",
    batch_size: int = 128,
    resize: Tuple[int, int] = typer.Option((None, None)),
    mixed_precision: bool = False
):
    if mixed_precision:
        policy = amp.Policy('mixed_float16')
        amp.set_policy(policy)
        print('Compute dtype: %s' % policy.compute_dtype)
        print('Variable dtype: %s' % policy.variable_dtype)

    model = get_model(
        Path(model_path).stem.split("_")[0], pretrained=None,
        image_size=(
            (INPUT_IMAGE_SIZE[0], INPUT_IMAGE_SIZE[1], 3)
            if resize[0] is None else (*resize, 3)
        )
    )
    model.load_weights(model_path)
    df = pd.read_csv(df_path)
    predictions, labels = [], []
    for parquet_file in glob(parquet_pattern):
        print(f"Reading {parquet_file}")
        df_merged = pd.merge(
            pd.read_parquet(parquet_file),
            df, on='image_id', how="inner"
        ).drop(['image_id'], axis=1)
        dataset = get_dataset(df_merged, batch_size, resize)
        labels.append(df_merged.loc[:, LABEL_COLUMNS].values)
        for images in dataset:
            predictions.append(predict(model, images).numpy())
    predictions = np.concatenate(predictions)
    labels = np.concatenate(labels)
    score, formatted = MacroAveragedRecall()(labels, predictions)
    print(formatted)
    print(f"Macro Averaged Recall {score * -1 * 100:.4f}%")


if __name__ == '__main__':
    typer.run(main)
