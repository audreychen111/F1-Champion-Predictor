"""
Goal: Predict future Formula 1 Champions based off performance of past champions.

High-level flow:
1. Read past F1 race datasets
2. Identify features that attributed to past F1 champions
4. Train using logistic regression and gradient boosting classifier with set time period (1950-2016)
"""

# Import necessary libraries
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Read all csv files ------------------------------
races = pd.read_csv('races.csv')
ds = pd.read_csv('driver_standings.csv')
constructor = pd.read_csv('constructor_standings.csv')
res = pd.read_csv('results.csv')

# Determine if drivers are champions for each season -----------------------
ds = (ds.merge(races[['raceId', 'year']], on='raceId', how='left')
    .rename(columns={'year':'season'})
    .sort_values(['season', 'driverId', 'raceId']))

ds_final = ds.groupby(['season', 'driverId'], as_index=False).tail(1)
ds_final['is_champ'] = (ds_final['position'] == 1).astype(int)

# Create dataframe for previous season stats for each driver -----------------
feature_cols = [
    'points',
    'wins',
    'position'
]
prev_season = ds_final[['season', 'driverId'] + feature_cols].copy()
prev_season['season'] += 1
prev_season = prev_season.rename(columns={
    'points':   'points_prev',
    'wins':     'wins_prev',
    'position': 'position_prev',
})

data = ds_final.merge(prev_season, on=['season', 'driverId'], how='left')
data = data.dropna(subset=['points_prev', 'position_prev', 'wins_prev']).copy()

# Find total number of points of champions for each season
champ_prev = (ds_final
              .loc[ds_final['position'] == 1, ['season', 'points']]
              .rename(columns={'points': 'champ_points'}))
champ_prev['season'] += 1
data = data.merge(champ_prev, on='season', how='left')

# Calculate feature that measures driver points relative to champion points ----------------
data['percent_points_of_champ_prev'] = data['points_prev'] / data['champ_points']
data = (data
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=['percent_points_of_champ_prev']))

# Create constructor dataframe ---------------------------
constructor = (constructor.merge(races[['raceId', 'year']], on='raceId', how='left')
       .rename(columns={'year': 'season'})
       .sort_values(['season', 'constructorId', 'raceId']))

constructor_final = constructor.groupby(['season', 'constructorId'], as_index=False).tail(1)

total_constructor_points = (constructor_final
                    .groupby('season')['points']
                    .sum()
                    .rename('total_cst_points'))
constructor_final = constructor_final.merge(total_constructor_points, on='season', how='left')

# Calculates feature that measures percentage of total constructor points for each team -----------------
constructor_final['cst_share'] = constructor_final['points'] / constructor_final['total_cst_points']

constructor_ranked = constructor_final.sort_values(['season', 'points'], ascending=[True, False])
p2_points  = (constructor_ranked
              .groupby('season', as_index=False)
              .nth(1)[['season', 'points']]
              .rename(columns={'points': 'p2_points'}))
constructor_final = constructor_final.merge(p2_points, on='season', how='left')

# Calculates feature that measures each contructor's gap from second place team ---------------------
constructor_final['p2_gap'] = constructor_final['points'] - constructor_final['p2_points']

constructor_prev = constructor_final[['season', 'constructorId', 'cst_share', 'p2_gap']].copy()
constructor_prev['season'] += 1
constructor_prev = constructor_prev.rename(columns={
    'cst_share': 'cst_share_prev',
    'p2_gap':    'p2_gap_prev',
})

driver_constructor = (res
                      .merge(races[['raceId', 'year']], on='raceId', how='left')
                      .rename(columns={'year': 'season'})
                      .sort_values(['season', 'driverId', 'raceId'])
                      .groupby(['season', 'driverId'], as_index=False)
                      .tail(1)[['season', 'driverId', 'constructorId']])

# Completes 'data' to include information about driver and constructor stats ----------------
data = (data
        .merge(driver_constructor, on=['season', 'driverId'], how='left')
        .merge(constructor_prev, on=['season', 'constructorId'], how='left'))

# Calculates feature that measures previous season position inverse ----------------------
data['inv_position_prev'] = 1 / data['position_prev'].replace(0, np.nan)

# Calculates feature that measures how efficient drivers are at converting points to wins -------------------
data['wins_per_point_prev'] = data['wins_prev'] / data['points_prev'].replace(0, np.nan)
data = data.replace([np.inf, -np.inf], np.nan)

features = [
    'percent_points_of_champ_prev',
    'wins_prev',
    'inv_position_prev',
    'cst_share_prev',
    'p2_gap_prev',
    'wins_per_point_prev',
]

TRAIN_CUTOFF = 2016

train_data = data[data['season'] <= TRAIN_CUTOFF].dropna(subset=features).copy()
test_data  = data[data['season'] >  TRAIN_CUTOFF].dropna(subset=features).copy()

X_train, y_train = train_data[features], train_data['is_champ']
X_test,  y_test  = test_data[features],  test_data['is_champ']

lr_final = Pipeline([('scaler', StandardScaler()),
                     ('clf',    LogisticRegression(max_iter=2000,
                                                   class_weight='balanced',
                                                   C=0.5))])
gb_final = GradientBoostingClassifier(n_estimators=100,
                                      max_depth=3,
                                      learning_rate=0.05,
                                      subsample=0.8,
                                      random_state=42)
final_model = VotingClassifier(estimators=[('lr', lr_final), ('gb', gb_final)],
                               voting='soft')
final_model.fit(X_train, y_train)

probability = final_model.predict_proba(X_test)[:, 1]

print("\n── Test Set Data (2017+) ───────────────────────────────")
print("Test ROC-AUC :", round(roc_auc_score(y_test, probability), 4))
print("Test LogLoss :", round(log_loss(y_test, probability), 4))

prediction = test_data[['season', 'driverId']].copy()
prediction['prob_champ'] = probability

top = (prediction
       .sort_values(['season', 'prob_champ'], ascending=[True, False])
       .groupby('season')
       .head(1))

actual = test_data[test_data['is_champ'] == 1][['season', 'driverId']]
merged = top.merge(actual, on=['season', 'driverId'], how='left', indicator=True)
accuracy = (merged['_merge'] == 'both').mean()
print(f"Top-1 Season Accuracy: {accuracy:.2%}")

top3 = (prediction
        .sort_values(['season', 'prob_champ'], ascending=[True, False])
        .groupby('season')
        .head(3))
merged3   = top3.merge(actual, on=['season', 'driverId'], how='left', indicator=True)
top3_acc  = merged3.groupby('season')['_merge'].apply(lambda x: (x == 'both').any()).mean()
print(f"Top-3 Season Accuracy: {top3_acc:.2%}")

print("\n── Predicted champions per season ─────────────────────────")
print(top[['season', 'driverId', 'prob_champ']].to_string(index=False))
