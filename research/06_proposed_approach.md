# Proposed Approach — Better Than Existing Solutions

## Target: Score > 97.55 (beat the best public solution)

## Strategy: Enhanced Multi-Model Stacking with Advanced Feature Engineering

---

## Phase 1: Data Preprocessing

### 1.1 Missing Value Handling (Smart Imputation)
```python
# RoadType: Fill by geohash group mode (roads at same location are same type)
train['RoadType'] = train.groupby('geohash')['RoadType'].transform(
    lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 'Residential')
)

# Temperature: Fill by geohash group median (nearby areas have similar temp)
train['Temperature'] = train.groupby('geohash')['Temperature'].transform(
    lambda x: x.fillna(x.median())
).fillna(train['Temperature'].median())

# Weather: Fill by timestamp group mode (weather is time-dependent)
train['Weather'] = train.groupby('timestamp')['Weather'].transform(
    lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 'Sunny')
)
```

---

## Phase 2: Feature Engineering (Target 80+ features)

### 2.1 Geospatial Features
- Decode geohash → lat, lon
- Geohash prefixes (3, 4, 5 chars)
- Distance from geographic center
- Spatial grid binning (lat_bin × lon_bin)
- Geohash frequency (how many rows per geohash)

### 2.2 Temporal Features
- Hour, minute, minutes_of_day, quarter_of_day
- Cyclical encoding (sin/cos for hour and minute)
- Rush hour indicators (morning 7-9, evening 16-19)
- Night/midday/time_period categories
- Day features (day_of_week, is_weekend if applicable)

### 2.3 Target Encoding (LEAK-FREE)
- Mean/std/median/count demand per geohash
- Mean demand per geohash_prefix (4-char, 5-char)
- Mean demand per RoadType
- Mean demand per hour
- Mean demand per geohash+hour
- Mean demand per RoadType+hour
- Mean demand per geohash+RoadType
- Mean demand per RoadType+NumberofLanes
- Mean demand per timestamp

### 2.4 Interaction Features
- road_lanes = RoadType_encoded × 10 + NumberofLanes
- highway_flag = (RoadType == Highway) OR (NumberofLanes >= 4)
- lane_pressure = NumberofLanes / (RoadType_encoded + 1)
- lat × hour, lon × hour (spatial-temporal)

### 2.5 Novel Features (to beat existing solutions)
- **Geohash neighbor encoding**: Decode nearby geohashes, compute avg demand of neighbors
- **Time-weighted demand**: Recent time slots weighted higher
- **Demand volatility**: Std/mean ratio per geohash (coefficient of variation)
- **Geohash clustering**: KMeans on (lat, lon, mean_demand) → cluster_id
- **Temperature anomaly**: temp - geohash_avg_temp (deviation from location norm)
- **Day transition features**: Difference in demand patterns between day 48 and 49

---

## Phase 3: Model Training

### 3.1 Base Models (8 diverse models)
```
1. XGBoost v1: depth=8, lr=0.03, trees=3000
2. XGBoost v2: depth=10, lr=0.02, trees=4000
3. LightGBM v1: depth=8, lr=0.03, leaves=127
4. LightGBM v2: depth=-1, lr=0.02, leaves=255
5. CatBoost v1: depth=8, lr=0.03, iters=3000 (use native categoricals!)
6. CatBoost v2: depth=10, lr=0.02, iters=4000
7. ExtraTreesRegressor: n_estimators=500, max_depth=30
8. RandomForestRegressor: n_estimators=500, max_depth=25
```

### 3.2 Cross-Validation Strategy
- **5-Fold KFold** with shuffle
- Collect OOF (out-of-fold) predictions for each model
- Compute per-fold R² to check stability

### 3.3 Ensemble Strategy (Multi-Level)

**Level 1**: Weighted Blend
```python
# Grid search optimal weights across 8 models
# Weights should sum to 1.0
```

**Level 2**: Stacking Meta-Learner
```python
# Stack features: OOF predictions from all 8 models
# Add top-5 most important original features
# Meta-learner: Ridge(alpha=1.0) or ElasticNet
```

**Level 3**: Final Selection
```python
# Compare blend vs stacking OOF R²
# Pick the best, or average both
final_preds = np.clip(best_preds, 0, 1)
```

---

## Phase 4: Post-Processing
- Clip predictions: `np.clip(preds, 0, max_train_demand)`
- Validate submission format: 41,778 rows, 2 columns (Index, demand)

---

## Implementation Order
1. Load & explore data → EDA notebook
2. Feature engineering pipeline
3. Train individual models with CV
4. Ensemble & stacking
5. Generate submission
6. Iterate on features/hyperparams based on CV scores

---

## Expected Improvements Over Existing Solutions
| Enhancement | Why Better |
|-------------|-----------|
| 8 models instead of 6 | More diversity in ensemble |
| Geohash neighbor encoding | Captures spatial smoothness |
| Native CatBoost categoricals | Better geohash handling |
| Temperature anomaly feature | Location-relative signal |
| Geohash clustering | Reduce dimensionality while preserving spatial info |
| Time-weighted demand | Day 49 patterns may differ from Day 48 |
| ElasticNet meta-learner | Better regularization than Ridge alone |

---

## Libraries Needed
```
pandas numpy scikit-learn
xgboost lightgbm catboost
pygeohash (for geohash decoding)
matplotlib seaborn (for EDA)
```
