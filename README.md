# Freight Rate Prediction Challenge

## Overview
Predict freight load rates from the provided development data. Outputs include `validation_predictions.csv`, completed December inputs, and `scorer_results/candidate_december.png`.

## Model
- Target: `posted_rate`
- Model: Random Forest Regressor
- Final common features: `pickup`, `delivery`, `distance`, `equipment`, `weight`, `year`, `month_num`, `day`, `day_of_week`
- Numeric missing values: median imputation
- Categorical missing values: most-frequent imputation
- Categorical encoding: one-hot encoding with unknown categories ignored
- Selected parameters: `n_estimators=300`, `max_depth=20`, `min_samples_leaf=5`, `random_state=42`

## Holdout results
| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Median baseline (initial split) | 1146.79 | 1567.97 | -0.0522 |
| Random Forest (initial split) | 148.20 | 671.19 | 0.8072 |
| Common-feature Random Forest | 158.79 | 662.08 | 0.8115 |
| Tuned common-feature Random Forest | 143.14 | 639.79 | 0.8240 |

These are local holdout metrics, not official Spotter validation metrics. The supplied scorer validates output files and creates the chart; it says official final validation metrics are calculated by Spotter after submission.

## Expected layout
```text
freight-rate-prediction/
├── data/
│   ├── train_test.csv
│   ├── validation.csv
│   ├── validation_predictions_template.csv
│   └── december_chart_inputs.csv
├── scorer_results/candidate_december.png
├── Freight_Rate_Prediction.ipynb
├── score.py
├── requirements.txt
├── validation_predictions.csv
├── Freight_Rate_Prediction_Assessment_Report.docx
└── README.md
```
Use your actual notebook filename if different.

## Setup
From the project root:
```bash
python -m pip install -r requirements.txt
```

## Run scorer
```bash
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```
The scorer should validate 12,000 validation predictions and 31 December predictions and create `scorer_results/candidate_december.png`.

## Submission checklist
- [ ] `validation_predictions.csv` has exactly `load_id,predicted_rate` in that order and 12,000 rows.
- [ ] December CSV has exactly `pickup,delivery,distance,equipment,weight,date,predicted_rate` and 31 dates.
- [ ] Include the report with the December chart.
- [ ] Include notebook/solution code, dependencies, and run instructions.
- [ ] Record the required 2–3 minute Loom video with camera on.
- [ ] Check whether supplied assessment data may be published before adding it to a public repository.
