import typer
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def main(folds: int = 6):
    df = pd.read_csv("data/train.csv")
    skf = StratifiedKFold(n_splits=folds, shuffle=True)
    for i, (train_idx, valid_idx) in enumerate(skf.split(df, df.grapheme_root)):
        df_valid = df.iloc[valid_idx]
        df_train = df.iloc[train_idx]
        df_valid.to_csv(f"data/valid_split_{i}.csv", index=False)
        df_train.to_csv(f"data/train_split_{i}.csv", index=False)


if __name__ == '__main__':
    typer.run(main)
