import argparse
from pathlib import Path
import joblib, pandas as pd
import yaml

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

TARGET = "default.payment.next.month"

def make_preprocess(df):
    all_cols = list(df.columns)
    cat = [c for c in ["SEX","EDUCATION","MARRIAGE"] if c in all_cols] + [c for c in all_cols if c.startswith("PAY_")]
    num = [c for c in all_cols if c not in cat]

    num_tf = Pipeline([("imputer", SimpleImputer(strategy="median")),
                       ("scaler",  StandardScaler())])

    cat_tf = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                       ("onehot",  OneHotEncoder(handle_unknown="ignore"))])

    return ColumnTransformer([("num",  num_tf, num), ("cat",  cat_tf, cat)])

def train(train_path, test_path, model_out, metrics_out):
    # читаю параметры из params.yaml
    P = yaml.safe_load((Path(__file__).parents[1] / "params.yaml").read_text(encoding="utf-8"))
    lr = float(P["model"]["learning_rate"])
    n_est = int(P["model"]["n_estimators"])

    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)
    if "ID" in train.columns: train.drop(columns=["ID"], inplace=True)
    if "ID" in test.columns:  test.drop(columns=["ID"],  inplace=True)

    X_train, y_train = train.drop(columns=[TARGET]), train[TARGET]
    X_test,  y_test  = test.drop(columns=[TARGET]),  test[TARGET]

    pipe = Pipeline([
        ("preprocess", make_preprocess(train)),
        ("clf", GradientBoostingClassifier(learning_rate=lr, n_estimators=n_est, random_state=42)),
    ])

    pipe.fit(X_train, y_train)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    y_pred = pipe.predict(X_test)

    metrics = {
        "model": "GradientBoostingClassifier",
        "model_params": {"learning_rate": lr, "n_estimators": n_est},
        "roc_auc":  float(roc_auc_score(y_test, y_prob)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }

    Path(model_out).parent.mkdir(parents=True, exist_ok=True)
    Path(metrics_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, model_out)
    Path(metrics_out).write_text(__import__("json").dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test",  required=True)
    ap.add_argument("--model-out", required=True)
    ap.add_argument("--metrics-out", required=True)
    args = ap.parse_args()
    train(args.train, args.test, args.model_out, args.metrics_out)
