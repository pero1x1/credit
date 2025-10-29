
import sys, json, joblib, pandas as pd
from pathlib import Path
import yaml
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split

def main(train_path, test_path, out_dir):
    params = yaml.safe_load(open('params.yaml'))['model']
    model_type = params.get('type', 'GradientBoostingClassifier')

    X_train = pd.read_csv(train_path).drop(columns=['default.payment.next.month'])
    y_train = pd.read_csv(train_path)['default.payment.next.month']
    X_test  = pd.read_csv(test_path).drop(columns=['default.payment.next.month'])
    y_test  = pd.read_csv(test_path)['default.payment.next.month']

    if model_type == 'RandomForestClassifier':
        model = RandomForestClassifier(
            n_estimators=params.get('n_estimators', 200),
            max_depth=params.get('max_depth', None),
            class_weight=params.get('class_weight', None),
            random_state=params.get('random_state', 42)
        )
    else:
        model = GradientBoostingClassifier(
            n_estimators=params.get('n_estimators', 150),
            learning_rate=params.get('learning_rate', 0.1),
            random_state=params.get('random_state', 42)
        )

    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:,1] if hasattr(model,'predict_proba') else model.decision_function(X_test)
    pred  = (proba >= 0.5).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "model": model.__class__.__name__,
        "model_params": {k: getattr(model, k, None) for k in ['n_estimators','learning_rate','max_depth','class_weight']}
    }

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out/'credit_default_model.pkl')
    (out/'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')

if __name__ == "__main__":
    train_path, test_path, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    main(train_path, test_path, out_dir)
