# Reference: https://www.kaggle.com/wardenga/bengali-handwritten-graphemes-with-automl
from glob import glob
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

HEIGHT = 137
WIDTH = 236
N_CHANNELS = 1
FLATTEN_IMAGE = [str(x) for x in range(32332)]

CACHE_FOLDER = "cache/automl/"
BUCKET = "gs://kaggle-140808-bengali/imgs/"


def main():
    image_folder = CACHE_FOLDER + "imgs/"
    Path(image_folder).mkdir(exist_ok=True, parents=True)
    image_list = []

    # Convert images
    for parquet_file in glob("data/train_image_data_*.parquet"):
        print(parquet_file)
        df = pd.read_parquet(parquet_file)
        for i in tqdm(range(df.shape[0])):
            row = df.iloc[i]
            image_path = row.image_id + '.png'
            image = Image.fromarray(
                255 - row[FLATTEN_IMAGE].values.astype(
                    "uint8"
                ).reshape(HEIGHT, WIDTH),
                mode="L"
            )
            image.save(image_folder + image_path)
        image_list += [
            BUCKET + f"{image_id}.png" for image_id in df["image_id"]]

    # Prepare csv files
    del df
    df = pd.read_csv("data/train.csv")
    for col in ("grapheme_root", "vowel_diacritic", "consonant_diacritic"):
        df_tmp = pd.DataFrame({
            "image_path": image_list,
            "label": df[col].values
        })
        df_tmp.to_csv(CACHE_FOLDER + f"{col}.csv", index=False)


if __name__ == "__main__":
    main()
