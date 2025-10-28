
import json, argparse
from pathlib import Path
import joblib, pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

TARGET = "default.payment.next.month"

def make_preprocess(X):
    all_cols = X.columns.tolist()
    cat = [c for c in ["SEX","EDUCATION","MARRIAGE"] if c in all_cols] + [c for c in all_cols if c.startswith("PAY_")]
    cat = sorted(list(dict.fromkeys(cat)))
    num = [c for c in all_cols if c not in cat]

    num_tf = Pipeline([("imputer", SimpleImputer(strategy="median")),
                       ("scaler", StandardScaler())])
    cat_tf = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                       ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", num_tf, num), ("cat", cat_tf, cat)])

def main(train_path, test_path, model_out, metrics_out):
    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)
    for df in (train, test):
        if "ID" in df.columns: df.drop(columns=["ID"], inplace=True)

    X_train, y_train = train.drop(columns=[TARGET]), train[TARGET]
    X_test,  y_test  = test.drop(columns=[TARGET]),  test[TARGET]

    pipe = Pipeline([
        ("preprocess", make_preprocess(X_train)),
        ("clf", GradientBoostingClassifier(learning_rate=0.1, n_estimators=150, random_state=42))
    ])
    pipe.fit(X_train, y_train)

    y_proba = pipe.predict_proba(X_test)[:,1]
    y_pred  = pipe.predict(X_test)

    metrics = {
        "model": "GradientBoostingClassifier",
        "model_params": {"learning_rate": 0.1, "n_estimators": 150},
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0))
    }

    model_out = Path(model_out); model_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out = Path(metrics_out); metrics_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, model_out)
    Path(metrics_out).write_text(json.dumps(metrics, indent=2), encoding="utf-8")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--model-out", required=True)
    p.add_argument("--metrics-out", required=True)
    args = p.parse_args()
    main(args.train, args.test, args.model_out, args.metrics_out)
