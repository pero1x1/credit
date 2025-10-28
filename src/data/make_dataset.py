import argparse
from pathlib import Path
import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema, Check
from sklearn.model_selection import train_test_split

TARGET = "default.payment.next.month"

def validate(df: pd.DataFrame) -> pd.DataFrame:
    schema = DataFrameSchema({
        "LIMIT_BAL": Column(pa.Int64, Check.ge(0)),
        "SEX": Column(pa.Int64, Check.isin([1, 2])),
        "EDUCATION": Column(pa.Int64),
        "MARRIAGE": Column(pa.Int64),
        "AGE": Column(pa.Int64, Check.between(18, 100)),
        TARGET: Column(pa.Int64, Check.isin([0, 1])),
    }, coerce=True)
    return schema.validate(df, lazy=True)

def main(raw_path: str, out_dir: str, test_size: float = 0.2, seed: int = 42):
    raw_path, out_dir = Path(raw_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(raw_path)
    df = validate(df)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test, y_test], axis=1)
    train.to_csv(out_dir / "train.csv", index=False)
    test.to_csv(out_dir / "test.csv", index=False)
    print("Prepared: data/processed/train.csv, data/processed/test.csv")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default="data/raw/UCI_Credit_Card.csv")
    p.add_argument("--out", default="data/processed")
    args = p.parse_args()
    main(args.raw, args.out)
