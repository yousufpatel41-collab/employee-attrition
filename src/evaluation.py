"""Evaluate complete pipelines without resampling validation rows."""
import hashlib
import json
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score, f1_score, confusion_matrix

ROOT=Path(__file__).resolve().parents[1]

def compare(models, X, y, data_path):
    X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
    results={}
    for name, model in models.items():
        cv=cross_val_score(model,X_train,y_train,cv=StratifiedKFold(5,shuffle=True,random_state=42),scoring="roc_auc")
        results[name]={"cv_auc_mean":float(cv.mean()),"cv_auc_std":float(cv.std())}
    selected=max(results,key=lambda name:results[name]["cv_auc_mean"])
    # Model selection is complete before looking at held-out outcomes.
    for name, model in models.items():
        model.fit(X_train,y_train)
        prob=model.predict_proba(X_test)[:,1]
        pred=(prob>=.5).astype(int)
        results[name].update(roc_auc=float(roc_auc_score(y_test,prob)),accuracy=float(accuracy_score(y_test,pred)),
            recall=float(recall_score(y_test,pred)),precision=float(precision_score(y_test,pred,zero_division=0)),
            f1=float(f1_score(y_test,pred)),confusion_matrix=confusion_matrix(y_test,pred).tolist())
    metrics={"dataset_sha256":hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
        "train_rows":len(X_train),"test_rows":len(X_test),"positive_rows":int(y.sum()),"random_state":42,
        "threshold":.5,"selected_by_training_cv_auc":selected,"models":results}
    (ROOT/"outputs").mkdir(exist_ok=True)
    (ROOT/"models").mkdir(exist_ok=True)
    (ROOT/"outputs/metrics.json").write_text(json.dumps(metrics,indent=2)+"\n")
    with (ROOT/"models/selected_pipeline.pkl").open("wb") as f:
        pickle.dump(models[selected],f)
    print(json.dumps(metrics,indent=2))
    return metrics
