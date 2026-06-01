# Feature Engineering Research

## Key Findings from Top Solutions

### 1. Geospatial Features (HIGH IMPACT)
- **Geohash decoding**: Convert geohash → latitude, longitude using `pygeohash`
- **Geohash prefixes**: `geohash[:3]`, `geohash[:4]`, `geohash[:5]` for hierarchical location
- **Distance from center**: Euclidean distance from centroid of all points
- **Spatial binning**: lat_bin × lon_bin grid
- **Label encoding**: For tree-based models to capture geohash patterns

### 2. Temporal Features (HIGH IMPACT)
- **Hour/Minute extraction**: Parse "H:M" format timestamps
- **Minutes of day**: `hour * 60 + minute`
- **Quarter of day**: `minutes // 15` (96 unique 15-min slots)
- **Cyclical encoding** (critical for time-based features):
  - `hour_sin = sin(2π × hour / 24)`
  - `hour_cos = cos(2π × hour / 24)`
  - `minute_sin = sin(2π × minutes / 1440)`
  - `minute_cos = cos(2π × minutes / 1440)`
- **Rush hour flags**:
  - `is_morning_rush`: hour 7-9
  - `is_evening_rush`: hour 16-19
  - `is_night`: hour >= 22 or <= 5
  - `is_midday`: hour 10-15
- **Time period categories**: 6 discrete bins
- **Day features**:
  - `day_of_week = day % 7`
  - `is_weekend = day_of_week >= 5`

### 3. Target Encoding / Aggregation Features (HIGHEST IMPACT)
- **geo_demand_mean**: Mean demand per geohash (from training data only!)
- **geo_demand_std**: Std of demand per geohash
- **geo_demand_median**: Median demand per geohash
- **road_demand_mean**: Mean demand per RoadType
- **hour_demand_mean**: Mean demand per hour
- **geo_hour_demand_mean**: Mean demand per geohash + hour combination
- **road_hour_demand_mean**: Mean demand per RoadType + hour
- **geo_road_demand_mean**: Mean demand per geohash + RoadType
- **road_lane_mean**: Mean demand per RoadType + NumberofLanes combo
- **time_mean**: Mean demand per timestamp
- **Geohash prefix demand means**: Per 4-char and 5-char prefix

⚠️ **CRITICAL**: All aggregations must be computed from training data only to prevent data leakage!

### 4. Categorical Encoding
- **RoadType**: Ordinal mapping (Residential=0, Street=1, Highway=2)
- **LargeVehicles**: Binary (Allowed=1, Not Allowed=0)
- **Landmarks**: Binary (Yes=1, No=0)
- **Weather**: Ordinal + One-hot encoding
- **Missing value indicators**: Binary flags for missing RoadType, Temperature, Weather

### 5. Temperature Features
- `temp_squared`: Temperature²
- `temp_abs`: |Temperature|
- `temp_bin`: Binned temperature (10 bins)
- `is_cold`: Temperature < 5
- `is_hot`: Temperature > 30
- `is_moderate`: 10 ≤ Temperature ≤ 25

### 6. Interaction Features
- `road_lanes`: RoadType_encoded × 10 + NumberofLanes
- `highway_flag`: RoadType == Highway OR NumberofLanes >= 4
- `road_time`: RoadType × time_period
- `lanes_large`: NumberofLanes × LargeVehicles_encoded
- `temp_weather`: Temperature × Weather_encoded
- `hour_road`: hour × 10 + RoadType_encoded
- `lat_hour`: latitude × hour
- `lon_hour`: longitude × hour
- `lane_pressure`: NumberofLanes / (RoadType + 1)
- `vehicle_lane_interaction`: LargeVehicles × NumberofLanes

### 7. Lag Features (from GuptaNidhish approach)
- `demand_lag_1`: Previous timestamp's demand for same geohash
- `demand_lag_2`: Two timestamps back
- Fill NaN with median demand

### 8. Missing Value Handling
- **RoadType**: Fill with geohash-group mode, fallback to global mode
- **Temperature**: Fill with geohash-group median, fallback to global median
- **Weather**: Fill with geohash-group mode, fallback to "Unknown" or global mode
- **NumberofLanes**: Fill with RoadType-group mode
