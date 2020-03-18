import io
import glob
from pathlib import Path
from typing import List

import typer
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

HEIGHT = 137
WIDTH = 236
N_CHANNELS = 1
FLATTEN_IMAGE = [str(x) for x in range(32332)]

# Avoids cudnn initialization problem
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def predict_file(model_path, parquet_file, batch_size):
    print(f"Reading {parquet_file}")
    df = pd.read_parquet(parquet_file)

    models = {}
    for label_type in ("grapheme_root", "consonant_diacritic", "vowel_diacritic"):
        models[label_type] = tf.saved_model.load(
            str(Path(model_path) / label_type)
        ).signatures["serving_default"]

    preds, ids = [], []

    for name in df["image_id"].values:
        ids.extend([
            f'{name}_grapheme_root',
            f'{name}_vowel_diacritic',
            f'{name}_consonant_diacritic'
        ])

    def predict_(model, buffer):
        return model(
            image_bytes=tf.convert_to_tensor(buffer),
            key=tf.convert_to_tensor(
                [str(x) for x in range(len(buffer))])
        )

    def predict(buffer, preds):
        for label_type in ("grapheme_root", "consonant_diacritic", "vowel_diacritic"):
            outputs = predict_(models[label_type], buffer)
            picked = np.argmax(outputs["scores"].numpy(), axis=1)
            labels = outputs["labels"][0].numpy()
            preds.extend([
                int(labels[x]) for x in picked
            ])

    def encode(arr):
        image = Image.fromarray(
            255 - arr.reshape(HEIGHT, WIDTH),
            mode="L"
        )
        with io.BytesIO() as output:
            image.save(output, format="PNG")
            output.seek(0)
            encoded = output.read()
        return encoded

    with joblib.Parallel(n_jobs=2) as parallel:
        for i in range(0, df.shape[0], batch_size):
            batch = df.iloc[
                i:i+batch_size
            ][FLATTEN_IMAGE].values.astype(
                "uint8"
            )
            predict(
                parallel(
                    joblib.delayed(encode)(arr)
                    for arr in batch
                ),
                preds
            )

    return ids, preds


def main(
    model_path: str = "cache/automl/",
    parquet_pattern: str = "data/test_image_data_*.parquet",
    batch_size: int = 128
):
    predictions: List[float] = []
    ids: List[str] = []
    for parquet_file in glob.glob(parquet_pattern):
        preds_tmp, ids_tmp = predict_file(
            model_path, parquet_file, batch_size
        )
        predictions += preds_tmp
        ids += ids_tmp
    df_sub = pd.DataFrame({
        'row_id': ids,
        'target': predictions
    })
    df_sub.to_csv("submission.csv", index=False)


if __name__ == '__main__':
    typer.run(main)
