# B2B Sales Forecasting Solution
# =============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import cross_val_score, GroupKFold
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV

# Set random seed for reproducibility
np.random.seed(42)

# 1. Load the datasets
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
economic_df = pd.read_csv('EconomicIndicators.csv')

# 2. Data Exploration
print("Train data shape:", train_df.shape)
print("Test data shape:", test_df.shape)
print("Economic indicators shape:", economic_df.shape)

# Display basic information about the datasets
print("\nTrain data info:")
print(train_df.info())
print("\nSample of train data:")
print(train_df.head())

print("\nTest data info:")
print(test_df.info())
print("\nSample of test data:")
print(test_df.head())

print("\nEconomic indicators info:")
print(economic_df.info())
print("\nSample of economic indicators:")
print(economic_df.head())

# Check for missing values
print("\nMissing values in train data:")
print(train_df.isnull().sum())
print("\nMissing values in test data:")
print(test_df.isnull().sum())
print("\nMissing values in economic indicators:")
print(economic_df.isnull().sum())

# 3. Feature Engineering

# Extract quarter number from Quarter column
def extract_quarter_num(quarter_str):
    return int(quarter_str.replace('Q', ''))

train_df['QuarterNum'] = train_df['Quarter'].apply(extract_quarter_num)
test_df['QuarterNum'] = test_df['Quarter'].apply(extract_quarter_num)

# Map Quarter to corresponding months
quarter_to_months = {
    'Q1': [1, 2, 3],
    'Q2': [4, 5, 6],
    'Q3': [7, 8, 9],
    'Q4': [10, 11, 12],
    'Q5': [13, 14, 15],
    'Q6': [16, 17, 18],
    'Q7': [19, 20, 21],
    'Q8': [22, 23, 24],
    'Q9': [25, 26, 27]
}

# Function to merge economic indicators with sales data
def merge_economic_indicators(df, economic_df, quarter_to_months):
    result_df = df.copy()
    
    # Create new columns for aggregated economic indicators
    economic_features = ['Consumer Sentiment', 'Interest Rate', 'PMI', 
                         'Money Supply', 'NationalEAI', 'EastEAI', 
                         'WestEAI', 'SouthEAI', 'NorthEAI']
    
    for feature in economic_features:
        result_df[f'Avg_{feature}'] = 0.0
        result_df[f'Min_{feature}'] = 0.0
        result_df[f'Max_{feature}'] = 0.0
        result_df[f'Trend_{feature}'] = 0.0
    
    # Aggregate economic indicators for each quarter in each row
    for _, row in result_df.iterrows():
        quarter = row['Quarter']
        months = quarter_to_months.get(quarter, [])
        
        # Filter economic data for the corresponding months
        quarter_economic = economic_df[economic_df['Month'].isin(months)]
        
        if not quarter_economic.empty:
            for feature in economic_features:
                result_df.loc[_, f'Avg_{feature}'] = quarter_economic[feature].mean()
                result_df.loc[_, f'Min_{feature}'] = quarter_economic[feature].min()
                result_df.loc[_, f'Max_{feature}'] = quarter_economic[feature].max()
                
                # Calculate trend (difference between last and first month in quarter)
                if len(quarter_economic) > 1:
                    first_month = quarter_economic.iloc[0][feature]
                    last_month = quarter_economic.iloc[-1][feature]
                    result_df.loc[_, f'Trend_{feature}'] = last_month - first_month
    
    # Add region-specific economic indicators
    for _, row in result_df.iterrows():
        region = row['Region']
        if region == 'East':
            result_df.loc[_, 'RegionalEAI'] = result_df.loc[_, 'Avg_EastEAI']
        elif region == 'West':
            result_df.loc[_, 'RegionalEAI'] = result_df.loc[_, 'Avg_WestEAI']
        elif region == 'South':
            result_df.loc[_, 'RegionalEAI'] = result_df.loc[_, 'Avg_SouthEAI']
        elif region == 'North':
            result_df.loc[_, 'RegionalEAI'] = result_df.loc[_, 'Avg_NorthEAI']
    
    return result_df

# Apply economic indicators to both train and test data
train_df = merge_economic_indicators(train_df, economic_df, quarter_to_months)
test_df = merge_economic_indicators(test_df, economic_df, quarter_to_months)

# Create lagged features for time series aspects
# Group by company to create company-specific features
def create_company_features(df):
    result_df = df.copy()
    
    # Sort by company and quarter for proper lag creation
    result_df = result_df.sort_values(['Company', 'QuarterNum'])
    
    # Initialize columns for lagged features
    result_df['Sales_Lag1'] = np.nan
    result_df['Sales_Lag2'] = np.nan
    result_df['Sales_Lag3'] = np.nan
    result_df['Sales_Growth_Rate'] = np.nan
    result_df['Sales_Rolling_Mean'] = np.nan
    result_df['Sales_Rolling_Std'] = np.nan
    
    # Calculate lagged features for each company
    for company in df['Company'].unique():
        company_mask = result_df['Company'] == company
        company_data = result_df[company_mask].copy()
        
        if 'Sales' in company_data.columns:
            # Create lag features
            company_data['Sales_Lag1'] = company_data['Sales'].shift(1)
            company_data['Sales_Lag2'] = company_data['Sales'].shift(2)
            company_data['Sales_Lag3'] = company_data['Sales'].shift(3)
            
            # Calculate growth rate
            company_data['Sales_Growth_Rate'] = company_data['Sales'].pct_change()
            
            # Calculate rolling statistics
            company_data['Sales_Rolling_Mean'] = company_data['Sales'].rolling(window=3, min_periods=1).mean()
            company_data['Sales_Rolling_Std'] = company_data['Sales'].rolling(window=3, min_periods=1).std()
            
            # Update the main dataframe
            result_df.loc[company_mask] = company_data
    
    return result_df

# Apply company features to train data
train_df = create_company_features(train_df)

# Create additional features based on the bond and stock ratings
def process_ratings(df):
    result_df = df.copy()
    
    # Convert bond ratings to numeric scores
    bond_mapping = {
        'AAA': 10, 'AA+': 9.5, 'AA': 9, 'AA-': 8.5,
        'A+': 8, 'A': 7.5, 'A-': 7, 'BBB+': 6.5,
        'BBB': 6, 'BBB-': 5.5, 'BB+': 5, 'BB': 4.5,
        'BB-': 4, 'B+': 3.5, 'B': 3, 'B-': 2.5,
        'CCC+': 2, 'CCC': 1.5, 'CCC-': 1, 'D': 0.5
    }
    
    result_df['BondScore'] = result_df['Bond rating'].map(bond_mapping)
    
    # Convert stock ratings to numeric scores
    stock_mapping = {
        'Strong Buy': 5, 'Buy': 4, 'Hold': 3, 'Sell': 2, 'Strong Sell': 1
    }
    
    result_df['StockScore'] = result_df['Stock rating'].map(stock_mapping)
    
    # Create a combined rating score
    result_df['CombinedScore'] = result_df['BondScore'] * result_df['StockScore']
    
    return result_df

# Apply rating processing to both train and test data
train_df = process_ratings(train_df)
test_df = process_ratings(test_df)

# 4. Feature Selection and Data Preparation

# Fill missing values in lag features with appropriate values (medians by company)
def fill_missing_values(df):
    result_df = df.copy()
    
    # Fill missing lag values with company medians where available
    for company in result_df['Company'].unique():
        company_mask = result_df['Company'] == company
        for col in ['Sales_Lag1', 'Sales_Lag2', 'Sales_Lag3', 'Sales_Growth_Rate', 'Sales_Rolling_Mean', 'Sales_Rolling_Std']:
            if col in result_df.columns:
                median_val = result_df.loc[company_mask, col].median()
                if not pd.isna(median_val):
                    result_df.loc[company_mask, col] = result_df.loc[company_mask, col].fillna(median_val)
    
    # Fill any remaining missing values with global medians
    for col in result_df.columns:
        if result_df[col].isna().sum() > 0:
            median_val = result_df[col].median()
            if not pd.isna(median_val):
                result_df[col] = result_df[col].fillna(median_val)
            else:
                # If median is still NaN, fill with 0
                result_df[col] = result_df[col].fillna(0)
    
    return result_df

# Apply missing value filling to train data
train_df = fill_missing_values(train_df)

# Define features and target
def prepare_features(train_df, test_df):
    # Define which features to use
    numeric_features = [
        'QuickRatio', 'InventoryRatio', 'RevenueGrowth', 'Marketshare',
        'QuarterNum', 'BondScore', 'StockScore', 'CombinedScore', 
        'Avg_Consumer Sentiment', 'Avg_Interest Rate', 'Avg_PMI', 
        'Avg_Money Supply', 'Avg_NationalEAI', 'RegionalEAI',
        'Trend_Interest Rate', 'Trend_PMI', 'Trend_NationalEAI'
    ]
    
    # Add lag features if they exist in train_df
    lag_features = [
        'Sales_Lag1', 'Sales_Lag2', 'Sales_Lag3', 
        'Sales_Growth_Rate', 'Sales_Rolling_Mean', 'Sales_Rolling_Std'
    ]
    
    for feature in lag_features:
        if feature in train_df.columns:
            numeric_features.append(feature)
    
    categorical_features = ['Region', 'Industry']
    
    # Prepare X_train, y_train
    X_train = train_df[numeric_features + categorical_features]
    y_train = train_df['Sales']
    
    # Prepare X_test
    X_test = test_df[numeric_features + categorical_features]
    
    return X_train, y_train, X_test, numeric_features, categorical_features

# Prepare features
X_train, y_train, X_test, numeric_features, categorical_features = prepare_features(train_df, test_df)

# Check for missing values in prepared data
print("\nMissing values in X_train:")
print(X_train.isnull().sum())
print("\nMissing values in X_test:")
print(X_test.isnull().sum())

# Handle any remaining missing values
X_train = pd.DataFrame(X_train).fillna(0)
X_test = pd.DataFrame(X_test).fillna(0)

# 5. Model Building and Evaluation

# Create a preprocessor with scaling for numeric features and one-hot encoding for categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Define various models to try
models = {
    'RandomForest': RandomForestRegressor(random_state=42),
    'GradientBoosting': GradientBoostingRegressor(random_state=42),
    'ElasticNet': ElasticNet(random_state=42),
    'XGBoost': xgb.XGBRegressor(random_state=42),
    'LightGBM': lgb.LGBMRegressor(random_state=42)
}

# Cross-validation strategy (use GroupKFold to prevent data leakage between companies)
cv = GroupKFold(n_splits=5)
groups = train_df['Company']  # Group by company

# Evaluate models
model_scores = {}
for name, model in models.items():
    # Create pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Perform cross-validation
    scores = cross_val_score(
        pipeline, X_train, y_train, 
        cv=cv, scoring='neg_mean_absolute_error', 
        groups=groups
    )
    
    # Store the absolute value of the mean score (since MAE is negative)
    model_scores[name] = -scores.mean()
    
    print(f"{name} CV MAE: {-scores.mean():.2f} (± {scores.std():.2f})")

# Select the best model based on cross-validation
best_model_name = min(model_scores, key=model_scores.get)
print(f"\nBest model: {best_model_name} with MAE: {model_scores[best_model_name]:.2f}")

# Fine-tune the best model using GridSearchCV
if best_model_name == 'RandomForest':
    param_grid = {
        'model__n_estimators': [100, 200, 300],
        'model__max_depth': [None, 10, 20, 30],
        'model__min_samples_split': [2, 5, 10]
    }
elif best_model_name == 'GradientBoosting':
    param_grid = {
        'model__n_estimators': [100, 200, 300],
        'model__learning_rate': [0.01, 0.05, 0.1],
        'model__max_depth': [3, 5, 7]
    }
elif best_model_name == 'XGBoost':
    param_grid = {
        'model__n_estimators': [100, 200, 300],
        'model__learning_rate': [0.01, 0.05, 0.1],
        'model__max_depth': [3, 5, 7],
        'model__subsample': [0.8, 0.9, 1.0]
    }
elif best_model_name == 'LightGBM':
    param_grid = {
        'model__n_estimators': [100, 200, 300],
        'model__learning_rate': [0.01, 0.05, 0.1],
        'model__num_leaves': [31, 63, 127]
    }
else:  # ElasticNet
    param_grid = {
        'model__alpha': [0.1, 0.5, 1.0],
        'model__l1_ratio': [0.1, 0.5, 0.9]
    }

# Create pipeline with best model
best_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', models[best_model_name])
])

# Perform grid search
grid_search = GridSearchCV(
    best_pipeline,
    param_grid=param_grid,
    cv=cv,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train, groups=groups)

# Get the best parameters and score
best_params = grid_search.best_params_
best_score = -grid_search.best_score_  # Convert back to positive MAE
print(f"\nBest parameters: {best_params}")
print(f"Best cross-validation MAE: {best_score:.2f}")

# 6. Ensemble Modeling (Stacking)
from sklearn.ensemble import StackingRegressor

# Get the top 3 models based on CV scores
top_models = sorted(model_scores.items(), key=lambda x: x[1])[:3]
top_model_names = [model[0] for model in top_models]
print(f"\nTop 3 models for stacking: {top_model_names}")

# Create base estimators from the top models
base_estimators = []
for name in top_model_names:
    base_estimators.append((name, Pipeline([
        ('preprocessor', preprocessor),
        ('model', models[name])
    ])))

# Define the final estimator
final_estimator = models[best_model_name]

# Create the stacking regressor
stacking_regressor = StackingRegressor(
    estimators=base_estimators,
    final_estimator=final_estimator,
    cv=cv
)

# Train the stacking regressor
stacking_regressor.fit(X_train, y_train)

# 7. Generate Predictions and Create Submission File

# Train the best model on the full training data
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)

# For test data, we need to handle the lack of lag features
# Fill test lag features with the most recent values from train
def prepare_test_lag_features(train_df, test_df):
    test_with_lags = test_df.copy()
    
    # For each company in test, find the most recent values from train
    for company in test_with_lags['Company'].unique():
        # Get data for this company from train
        company_train = train_df[train_df['Company'] == company].sort_values('QuarterNum', ascending=False)
        
        if len(company_train) > 0:
            # Get the most recent data point
            latest_data = company_train.iloc[0]
            
            # For each test row of this company, fill in the lag values
            company_mask = test_with_lags['Company'] == company
            for lag_col in ['Sales_Lag1', 'Sales_Lag2', 'Sales_Lag3']:
                if lag_col in train_df.columns:
                    shift_amount = int(lag_col.split('_Lag')[1])
                    
                    # If we have enough history, use the appropriate lag
                    if len(company_train) >= shift_amount:
                        lag_value = company_train.iloc[shift_amount-1]['Sales'] if shift_amount <= len(company_train) else 0
                        test_with_lags.loc[company_mask, lag_col] = lag_value
                    else:
                        # Not enough history, use the median sales for this company
                        test_with_lags.loc[company_mask, lag_col] = company_train['Sales'].median()
            
            # Fill other derived features
            if 'Sales_Growth_Rate' in train_df.columns:
                test_with_lags.loc[company_mask, 'Sales_Growth_Rate'] = latest_data['Sales_Growth_Rate']
            
            if 'Sales_Rolling_Mean' in train_df.columns:
                test_with_lags.loc[company_mask, 'Sales_Rolling_Mean'] = latest_data['Sales_Rolling_Mean']
            
            if 'Sales_Rolling_Std' in train_df.columns:
                test_with_lags.loc[company_mask, 'Sales_Rolling_Std'] = latest_data['Sales_Rolling_Std']
    
    return test_with_lags

# Prepare test data with lag features
test_with_lags = prepare_test_lag_features(train_df, test_df)

# Prepare features for prediction
X_test_final, _, _, numeric_features, categorical_features = prepare_features(train_df, test_with_lags)

# Handle missing values
X_test_final = pd.DataFrame(X_test_final).fillna(0)

# Make predictions with best model and stacking ensemble
best_predictions = best_model.predict(X_test_final)
stacking_predictions = stacking_regressor.predict(X_test_final)

# Average the predictions (ensemble of ensembles)
final_predictions = (best_predictions + stacking_predictions) / 2

# Create submission file
submission = pd.DataFrame({
    'ID': test_df['RowID'],
    'Sales': final_predictions
})

# Ensure predictions are non-negative
submission['Sales'] = submission['Sales'].clip(lower=0)

# Save to CSV
submission.to_csv('submission.csv', index=False)
print("\nSubmission file created.")

# 8. Feature Importance Analysis
def plot_feature_importance(model, feature_names):
    if hasattr(model, 'feature_importances_'):
        # For tree-based models
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 8))
        plt.title('Feature Importances')
        plt.bar(range(len(indices)), importances[indices], align='center')
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.show()
    elif hasattr(model, 'coef_'):
        # For linear models
        importances = np.abs(model.coef_)
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 8))
        plt.title('Feature Importances')
        plt.bar(range(len(indices)), importances[indices], align='center')
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.show()
    else:
        print("Model doesn't have feature_importances_ or coef_ attribute")

# Try to extract the actual model from the pipeline
try:
    # Get the list of feature names after preprocessing
    preprocessed_features = []
    for name, transformer, features in preprocessor.transformers_:
        if name == 'num':
            preprocessed_features.extend(features)
        elif name == 'cat':
            # For categorical features, get the one-hot encoded feature names
            for feature in features:
                categories = list(transformer.categories_[features.index(feature)])
                preprocessed_features.extend([f"{feature}_{category}" for category in categories])
    
    # Extract the model from the pipeline
    model = best_model.named_steps['model']
    
    # Plot feature importance
    plot_feature_importance(model, preprocessed_features)
except Exception as e:
    print(f"Could not plot feature importance: {e}")

print("\nB2B Sales Forecasting project completed successfully!")