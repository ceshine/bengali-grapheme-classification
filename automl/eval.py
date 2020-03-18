import io
import base64

import sklearn.metrics
import tensorflow as tf
from PIL import Image
import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm

from prepare_data import HEIGHT, WIDTH, FLATTEN_IMAGE

BATCH_SIZE = 128

# Avoids cudnn initialization problem
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def eval_single(label_type):
    model = tf.saved_model.load(f"cache/automl/{label_type}/")
    # print(dir(model.signatures["serving_default"]))
    # print(model.signatures["serving_default"].inputs)
    # print(model.signatures["serving_default"].outputs)
    # print(model.signatures["serving_default"].structured_outputs)
    model = model.signatures["serving_default"]
    df_labels = pd.read_csv("data/train.csv")
    HEAD = 5000
    df = pd.read_parquet("data/train_image_data_0.parquet").iloc[:HEAD]
    df_labels = df[["image_id"]].merge(df_labels, on="image_id", how="left")
    print(df_labels.shape)
    preds = []

    def predict_(buffer):
        return model(
            image_bytes=tf.convert_to_tensor(buffer),
            key=tf.convert_to_tensor(
                [str(x) for x in range(len(buffer))])
        )

    def predict(buffer, preds):
        outputs = predict_(buffer)
        picked = np.argmax(outputs["scores"].numpy(), axis=1)
        labels = outputs["labels"][0].numpy()
        preds.extend(
            int(labels[x]) for x in picked
        )

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
        for i in tqdm(range(0, HEAD, BATCH_SIZE)):
            batch = df.iloc[
                i:i+BATCH_SIZE
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

    labels = df_labels[label_type].values
    return sklearn.metrics.recall_score(
        labels, preds, average='macro')


def main():
    scores = []
    for label_type in ("grapheme_root", "consonant_diacritic", "vowel_diacritic"):
        scores.append(eval_single(label_type))
        print(f"{label_type}: {scores[-1]:.4f}")
    print(f"Overall: {(2 * scores[0] + scores[1] + scores[2]) / 4:.4f}")


if __name__ == "__main__":
    main()
