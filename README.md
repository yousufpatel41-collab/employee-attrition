# Employee attrition

Uses IBM’s public HR sample data to compare an XGBoost baseline with random oversampling. This is a portfolio experiment, not an employer deployment.

## Run

Use Python 3.11 or 3.12 in a virtual environment, from the repository root:

```shell
pip install -r requirements.txt
python src/download_data.py
python src/train.py
python -m unittest discover -s tests
```

## Evaluation

Random oversampling duplicates training records without creating fractional categorical values. It replaces the previous SMOTE approach. Resampling happens inside each cross-validation training fold. A stratified 20% test set is reserved before fitting preprocessing. Models are selected by five-fold ROC-AUC on the training split before viewing test outcomes. Held-out comparisons for all candidates are descriptive; do not repeatedly tune against this test set.

`outputs/metrics.json` records dataset checksum, split sizes, seed, threshold, training CV scores and test metrics. `models/selected_pipeline.pkl` includes preprocessing and the selected classifier. Only load artifacts you trained yourself. Earlier chart and accuracy claims were removed because they did not match the committed code. No EDA notebook, Power BI dashboard or SHAP explanation is included.

## Data and limits

Source: [IBM employee attrition sample](https://github.com/IBM/employee-attrition-aif360/blob/master/data/emp_attrition.csv). The sample has no work-mode or country fields. `remote_work.py` explains why it cannot estimate remote-work attrition effects. No 2.1× risk claim or causal claim is supported. `sql_queries.sql` contains example queries; import the source CSV as `hr_employees` and adapt your SQL dialect.
