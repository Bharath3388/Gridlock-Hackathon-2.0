# Key Insights from Research

## Data Insights

### RoadType is the STRONGEST predictor
| RoadType | Mean Demand |
|----------|-------------|
| Residential | 0.057 |
| Street | 0.273 |
| Highway | 0.611 |

→ Highway demand is **10x** Residential demand!

### NumberofLanes correlates with RoadType
- Lanes 4-5 → strongly indicates Highway → high demand (~0.60)
- Lanes 1-3 → mostly Residential/Street → low demand (~0.08)

### Weather has LOW impact
| Weather | Mean Demand |
|---------|-------------|
| Snowy | 0.093 |
| Foggy | 0.093 |
| Sunny | 0.094 |
| Rainy | 0.094 |

→ Almost no difference! Don't over-engineer weather features.

### Geohash-based target encoding is CRITICAL
- 1,249 unique geohashes with very different demand profiles
- Some geohashes: demand mean = 0.0005 (very quiet areas)
- Some geohashes: demand mean = 0.96 (very busy areas)
- geo_mean target encoding provides massive signal

### Temporal patterns exist
- Peak demand timestamps around morning/evening rush hours
- Night hours (0:00-5:00) have lower demand
- Day 48 vs Day 49 may have different distributions

### Demand characteristics
- 76,715 unique demand values (nearly all unique → continuous target)
- Continuous regression problem
- Range: approximately 0 to 1
- Right-skewed distribution (most values are low)

---

## Model Selection Insights

1. **Gradient Boosting >> Linear Models** for this tabular data
2. **CatBoost handles categoricals natively** — very strong for geohash
3. **LightGBM is fastest** to train with this data size
4. **XGBoost is most stable** with good regularization
5. **Stacking with Ridge** captures complementary model strengths
6. **Model diversity matters** — same algorithm with different hyperparams helps

---

## Common Pitfalls to Avoid

1. ❌ **Data leakage**: Using test data for target encoding
2. ❌ **Ignoring geohash**: It's the most important feature via target encoding
3. ❌ **Not handling missing values smartly**: Use group-based imputation
4. ❌ **Overfitting to Day 48**: Most training data is Day 48, but test is Day 49
5. ❌ **Not using cross-validation**: Single train/test split is unreliable
6. ❌ **Negative predictions**: Clip to 0 minimum
7. ❌ **Too many features without importance check**: Can lead to noise
