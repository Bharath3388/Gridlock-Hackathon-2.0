# Traffic Demand Prediction — Gridlock Hackathon 2.0

Predict traffic demand for 41,778 test samples using an 8-model ensemble with stacking.

**Final CV Score: 97.59** (R² × 100)

## Project Structure

```
├── config/
│   └── settings.py          # Hyperparameters, paths, feature lists
├── src/
│   ├── data_loader.py       # Data loading & validation
│   ├── features.py          # Feature engineering pipeline (78 features)
│   ├── models.py            # Model training (XGB, LGB, CatBoost, ET, RF)
│   ├── ensemble.py          # Weighted blend + Ridge stacking
│   └── utils.py             # Logging, timing, submission generation
├── dataset/                  # train.csv, test.csv (not in git)
├── outputs/                  # Generated submissions (not in git)
├── research/                 # Problem analysis & approach documentation
├── run.py                    # Main entry point
├── requirements.txt          # Pinned dependencies
└── .gitignore
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Place dataset files in dataset/
# dataset/train.csv (77,299 × 11)
# dataset/test.csv  (41,778 × 10)

# Run the full pipeline
python run.py
```

Output: `outputs/submission.csv`

## Models

| Model | CV Score |
|-------|----------|
| CatBoost v1 | 97.57 |
| CatBoost v2 | 97.52 |
| LightGBM v1 | 97.34 |
| LightGBM v2 | 97.27 |
| XGBoost v1 | 97.25 |
| XGBoost v2 | 97.16 |
| RandomForest | 97.28 |
| ExtraTrees | 97.24 |
| **Stacking Ensemble** | **97.59** |

## Key Features

- **78 engineered features**: geospatial, temporal (cyclical), target encoding (leak-free), interactions
- **Leak-free pipeline**: Target encodings computed only on training data
- **5-Fold CV**: Consistent evaluation across all models
- **Stacking**: Ridge meta-learner on OOF predictions + important raw features

## Competition

[Gridlock Hackathon 2.0 on HackerEarth](https://www.hackerearth.com/challenges/competitive/gridlock-hackathon-20/machine-learning/traffic-demand-prediction-12/)
