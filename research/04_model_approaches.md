# Model Approaches Research

## Approach 1: Simple Ensemble (Score ~90.58)
**Source**: harshsingh9151

### Models
- **RandomForestRegressor**: n_estimators=500, max_depth=25, min_samples_leaf=2
- **ExtraTreesRegressor**: n_estimators=500, max_depth=25

### Ensemble Strategy
```python
final_preds = 0.75 * rf_preds + 0.25 * et_preds
```

### Pros
- Simple, fast to implement
- No hyperparameter tuning complexity

### Cons
- Lower score (90.58)
- No cross-validation for robust estimates
- Limited feature engineering

---

## Approach 2: Advanced Multi-Model Stacking (Score ~97.55)
**Source**: Ayush-Kumar0207

### Base Models (6 diverse models)
1. **XGBoost v1**: depth=8, lr=0.03, 3000 trees, colsample=0.7
2. **XGBoost v2**: depth=10, lr=0.02, 4000 trees, colsample=0.65
3. **LightGBM v1**: depth=8, lr=0.03, 3000 trees, 127 leaves
4. **LightGBM v2**: unlimited depth, lr=0.02, 4000 trees, 255 leaves
5. **CatBoost v1**: depth=8, lr=0.03, 3000 iters, Bayesian bootstrap
6. **CatBoost v2**: depth=10, lr=0.02, 4000 iters, high regularization

### Training
- 5-Fold Cross-Validation
- Early stopping on each fold
- Out-of-Fold predictions collected for stacking

### Ensemble Strategy
1. **Weighted Blend**: Grid search for optimal weights across 6 models
2. **Stacking Meta-Learner**: Ridge regression on OOF predictions + important original features

### Key Design Decisions
- No deep learning: Tabular data with ~77K rows favors gradient boosting
- Model diversity: Three different GBDT implementations × two configs
- Stacking captures complementary strengths

---

## Approach 3: CatBoost-Dominant with KFold (Score ~92+)
**Source**: GuptaNidhish

### Models
- **CatBoostRegressor**: 4000 iters, depth=8, early_stopping=150
- **LightGBM**: 2000 iters, lr=0.02, 127 leaves
- **XGBoost**: 2000 iters, used in final stacking

### Ensemble
```python
# Third try: LightGBM + CatBoost
final_preds = 0.4 * lgb_test_preds + 0.6 * cat_test_preds

# Fourth try: Ridge stacking of all 6 models
```

### Unique Features
- `traffic_cluster_id`: KMeans clustering on spatial features
- Demand lag features
- Geohash frequency encoding
- CatBoost handles categorical features natively (no encoding needed)

---

## Key Hyperparameters Summary

### XGBoost
```python
xgb_params = {
    'n_estimators': 3000,
    'max_depth': 8,
    'learning_rate': 0.03,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'min_child_weight': 5,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'tree_method': 'hist',
    'random_state': 42
}
```

### LightGBM
```python
lgb_params = {
    'n_estimators': 3000,
    'max_depth': 8,
    'learning_rate': 0.03,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'min_child_samples': 20,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'num_leaves': 127,
    'random_state': 42
}
```

### CatBoost
```python
cat_params = {
    'iterations': 3000,
    'depth': 8,
    'learning_rate': 0.03,
    'l2_leaf_reg': 3,
    'random_seed': 42,
    'bootstrap_type': 'Bayesian',
    'bagging_temperature': 0.5,
    'task_type': 'CPU'
}
```

---

## Important Notes
- **Clipping**: `np.clip(final_preds, 0, 1)` — demand cannot be negative
- **Early stopping**: Always use to prevent overfitting
- **Cross-validation**: 5-fold is standard for this dataset size
- **Seed diversity**: Use different seeds for v1/v2 of each model for diversity
