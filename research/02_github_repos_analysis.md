# GitHub Repositories Analysis

## Repos Found for This Exact Problem

### 1. Ayush-Kumar0207/Traffic-demand-prediction ⭐ BEST (Score: 97.55)
- **URL**: https://github.com/Ayush-Kumar0207/Traffic-demand-prediction
- **Language**: Jupyter Notebook + Python
- **Approach**: Advanced 6-model ensemble (2×XGBoost + 2×LightGBM + 2×CatBoost) with Ridge stacking
- **Features**: 71 engineered features
- **Key**: Geohash decoded to lat/lon, target encoding, spatial binning, optimal weight blending

### 2. harshsingh9151/traffic-demand-prediction (Score: 90.58)
- **URL**: https://github.com/harshsingh9151/traffic-demand-prediction
- **Language**: Jupyter Notebook
- **Approach**: Random Forest + Extra Trees ensemble (75%/25% weighted)
- **Features**: Target encoding (geo_mean, road_mean, road_lane_mean, time_mean), cyclical time features
- **Key**: Simpler approach but solid baseline

### 3. GuptaNidhish/GridLock-Challenge (Multiple Approaches)
- **URL**: https://github.com/GuptaNidhish/GridLock-Challenge
- **Language**: Jupyter Notebook
- **Approach 1 (fourth_try)**: CatBoost-dominant ensemble with Ridge stacking, 6 models
- **Approach 2 (third_try)**: LightGBM + CatBoost KFold (0.4/0.6 blend)
- **Approach 3 (notebooks)**: XGBoost + LightGBM with demand lag features
- **Key Features**: pygeohash decoding, demand_lag_1/lag_2, traffic_cluster_id, KMeans clustering

### 4. vanshh4/Traffic-Demand-Prediction-Model (3 stars, Flipkart Gridlock specific)
- **URL**: https://github.com/vanshh4/Traffic-Demand-Prediction-Model

### 5. Other Repos (0-1 stars)
- danybinuluke/gridlock-hackathon
- im25Morningstar/Gridlock-Hackathon
- Krishnachaitanyakoppaku/Traffic_demand_prediction
- shriyanssahoo/Traffic-Demand-Prediction
- arpan0926/Flipkart_Gridlock

## Score Comparison from Research

| Repo/Approach | Score |
|---------------|-------|
| Ayush-Kumar0207 (6-model stacking) | **97.55** |
| harshsingh9151 (RF+ET ensemble) | 90.58 |
| GuptaNidhish (CatBoost KFold) | ~90+ |
| Basic CatBoost baseline | ~86.75 |
