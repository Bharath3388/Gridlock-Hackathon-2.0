# Problem Statement Analysis

## Competition: Gridlock Hackathon 2.0 (Flipkart)
- **Platform**: HackerEarth
- **Duration**: May 26 - Jun 07, 2026

## Objective
Design a system that provides valuable insights into passenger travel patterns, booking behavior, and trip cancellations. Predict demand in the travel industry.

## Dataset
| File | Shape |
|------|-------|
| train.csv | 77,299 × 11 |
| test.csv | 41,778 × 10 |
| sample_submission.csv | 5 × 2 |

## Variables
| Column | Description |
|--------|-------------|
| Index | Unique identification of datapoint |
| geohash | Geographic information regarding a place |
| day | Day number (48-49) |
| timestamp | Time of day (format: "H:M", 15-min intervals) |
| RoadType | Type of road (Residential/Street/Highway) |
| NumberofLanes | Number of lanes (1-5) |
| LargeVehicles | Whether large vehicles allowed (Allowed/Not Allowed) |
| Landmarks | Whether landmarks near location (Yes/No) |
| Temperature | Temperature of the place |
| Weather | Weather condition (Sunny/Rainy/Foggy/Snowy) |
| **demand** | **TARGET** - demand of traffic at the timestamp |

## Evaluation Metric
```
score = max(0, 100 * metrics.r2_score(actual, predicted))
```
- R² score scaled to 0-100
- Max score: 100

## Submission Format
- CSV with 2 columns: `Index`, `demand`
- Size: 41,778 × 2
- Index values from test file

## Missing Values (from research)
- **Train**: RoadType (600 missing), Temperature (2495 missing), Weather (797 missing)
- **Test**: RoadType (324 missing), Temperature (1349 missing), Weather (431 missing)

## Key Data Insights
- Day 48 dominates training data (~69,427 rows)
- Day 49 has limited training data (~7,872 rows) + all test data (41,778)
- 96 unique timestamps (15-min intervals across 24 hours)
- ~1,249 unique geohashes
- Demand is continuous, range roughly 0-1
