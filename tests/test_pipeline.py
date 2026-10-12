import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from train import build_model

class PipelineTests(unittest.TestCase):
    def test_prediction_keeps_original_rows(self):
        X=pd.DataFrame({"Age":list(range(20)),"Department":["Sales","Research"]*10})
        model=build_model(True)
        model.set_params(classifier__n_estimators=2)
        model.fit(X,[0]*12+[1]*8)
        self.assertEqual(len(model.predict(X)),len(X))
        self.assertTrue(np.isfinite(model.predict_proba(X)).all())

    def test_imputer_uses_training_statistics(self):
        X=pd.DataFrame({"Age":list(range(20)),"Department":["Sales","Research"]*10})
        model=build_model(True)
        model.set_params(classifier__n_estimators=2)
        model.fit(X,[0]*12+[1]*8)
        prep=model.named_steps.get("preprocess")
        imputer=prep.named_transformers_["numeric"] if prep is not None else model.named_steps["impute"]
        before=imputer.statistics_.copy()
        sample=X.iloc[[0]].copy(); sample.iloc[0,0]=100000
        model.predict(sample)
        np.testing.assert_array_equal(before,imputer.statistics_)

if __name__=="__main__": unittest.main()
