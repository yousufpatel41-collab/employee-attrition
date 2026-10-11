"""Compare baseline and random oversampling on IBM's HR sample."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from xgboost import XGBClassifier
from evaluation import ROOT, compare

DATA_PATH=ROOT/"data/WA_Fn-UseC_-HR-Employee-Attrition.csv"

def load_data(path=DATA_PATH):
    df=pd.read_csv(path)
    y=df.pop("Attrition").map({"No":0,"Yes":1})
    if y.isna().any(): raise ValueError("Attrition must be Yes or No")
    X=df.drop(columns=["EmployeeCount","StandardHours","Over18","EmployeeNumber"],errors="ignore")
    return X,y.astype(int)

def build_model(oversample=False):
    prep=ColumnTransformer([
        ("numeric",SimpleImputer(strategy="median"),make_column_selector(dtype_include=np.number)),
        ("categorical",SkPipeline([("missing",SimpleImputer(strategy="most_frequent")),
            ("encode",OneHotEncoder(handle_unknown="ignore",sparse_output=False))]),make_column_selector(dtype_include=object)),
    ])
    steps=[("preprocess",prep)]
    if oversample: steps.append(("resample",RandomOverSampler(random_state=42)))
    steps.append(("classifier",XGBClassifier(n_estimators=400,max_depth=4,learning_rate=.04,
        subsample=.75,colsample_bytree=.75,eval_metric="auc",random_state=42,n_jobs=2)))
    return Pipeline(steps)

if __name__=="__main__":
    X,y=load_data()
    compare({"baseline":build_model(),"random_oversampling":build_model(True)},X,y,DATA_PATH)
