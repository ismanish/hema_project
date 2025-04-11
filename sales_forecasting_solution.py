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

# Set random seed for reproducibility
np.random.seed(42)

# 1. Load the datasets
print("Loading datasets...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
economic_df = pd.read_csv('EconomicIndicators.csv')

# 2. Data Exploration
print("Train data shape:", train_df.shape)
print("Test data shape:", test_df.shape)
print("Economic indicators shape:", economic_df.shape)

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
    for idx, row in result_df.iterrows():
        quarter = row['Quarter']
        months = quarter_to_months.get(quarter, [])
        
        # Filter economic data for the corresponding months
        quarter_economic = economic_df[economic_df['Month'].isin(months)]
        
        if not quarter_economic.empty:
            for feature in economic_features:
                result_df.loc[idx, f'Avg_{feature}'] = quarter_economic[feature].mean()
                result_df.loc[idx, f'Min_{feature}'] = quarter_economic[feature].min()
                result_df.loc[idx, f'Max_{feature}'] = quarter_economic[feature].max()
                
                # Calculate trend (difference between last and first month in quarter)
                if len(quarter_economic) > 1:
                    first_month = quarter_economic.iloc[0][feature]
                    last_month = quarter_economic.iloc[-1][feature]
                    result_df.loc[idx, f'Trend_{feature}'] = last_month - first_month
    
    # Add region-specific economic indicators
    for idx, row in result_df.iterrows():
        region = row['Region']
        if region == 'East':
            result_df.loc[idx, 'RegionalEAI'] = result_df.loc[idx, 'Avg_EastEAI']
        elif region == 'West':
            result_df.loc[idx, 'RegionalEAI'] = result_df.loc[idx, 'Avg_WestEAI']
        elif region == 'South':
            result_df.loc[idx, 'RegionalEAI'] = result_df.loc[idx, 'Avg_SouthEAI']
        elif region == 'North':
            result_df.loc[idx, 'RegionalEAI'] = result_df.loc[idx, 'Avg_NorthEAI']
    
    return result_df

print("Merging economic indicators...")
# Apply economic indicators to both train and test data
train_df = merge_economic_indicators(train_df, economic_df, quarter_to_months)
test_df = merge_economic_indicators(test_df, economic_df, quarter_to_months)

# Create company-specific features
def create_company_features(df):
    print("Creating company-specific features...")
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
    print("Processing ratings...")
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
    print("Filling missing values...")
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

# For test data, we need to handle the lack of lag features
# Fill test lag features with the most recent values from train
def prepare_test_lag_features(train_df, test_df):
    print("Preparing test lag features...")
    test_with_lags = test_df.copy()
    
    # Initialize all necessary lag columns
    lag_columns = ['Sales_Lag1', 'Sales_Lag2', 'Sales_Lag3', 
                   'Sales_Growth_Rate', 'Sales_Rolling_Mean', 'Sales_Rolling_Std']
    
    for col in lag_columns:
        if col not in test_with_lags.columns:
            test_with_lags[col] = np.nan
    
    # For each company in test, find the most recent values from train
    for company in test_with_lags['Company'].unique():
        # Get data for this company from train
        company_train = train_df[train_df['Company'] == company].sort_values('QuarterNum', ascending=False)
        
        if len(company_train) > 0:
            # Get the most recent data point
            latest_data = company_train.iloc[0]
            
            # For each test row of this company, fill in the lag values
            company_mask = test_with_lags['Company'] == company
            
            # Fill lag features
            for lag_col in ['Sales_Lag1', 'Sales_Lag2', 'Sales_Lag3']:
                shift_amount = int(lag_col.split('_Lag')[1])
                
                # If we have enough history, use the appropriate lag
                if len(company_train) >= shift_amount:
                    lag_value = company_train.iloc[shift_amount-1]['Sales'] if shift_amount <= len(company_train) else 0
                    test_with_lags.loc[company_mask, lag_col] = lag_value
                else:
                    # Not enough history, use the median sales for this company
                    test_with_lags.loc[company_mask, lag_col] = company_train['Sales'].median()
            
            # Fill other derived features
            for col in ['Sales_Growth_Rate', 'Sales_Rolling_Mean', 'Sales_Rolling_Std']:
                test_with_lags.loc[company_mask, col] = latest_data[col]
    
    return test_with_lags

# Apply test lag features preparation
test_with_lags = prepare_test_lag_features(train_df, test_df)

# Define features and target
def prepare_features(train_df, test_df):
    print("Preparing features...")
    # Define which features to use
    numeric_features = [
        'QuickRatio', 'InventoryRatio', 'RevenueGrowth', 'Marketshare',
        'QuarterNum', 'BondScore', 'StockScore', 'CombinedScore', 
        'Avg_Consumer Sentiment', 'Avg_Interest Rate', 'Avg_PMI', 
        'Avg_Money Supply', 'Avg_NationalEAI', 'RegionalEAI',
        'Trend_Interest Rate', 'Trend_PMI', 'Trend_NationalEAI',
        'Sales_Lag1', 'Sales_Lag2', 'Sales_Lag3', 
        'Sales_Growth_Rate', 'Sales_Rolling_Mean', 'Sales_Rolling_Std'
    ]
    
    categorical_features = ['Region', 'Industry']
    
    # Prepare X_train, y_train
    X_train = train_df[numeric_features + categorical_features]
    y_train = train_df['Sales']
    
    # Prepare X_test
    X_test = test_df[numeric_features + categorical_features]
    
    return X_train, y_train, X_test, numeric_features, categorical_features

# Prepare features
X_train, y_train, X_test, numeric_features, categorical_features = prepare_features(train_df, test_with_lags)

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

# Define models to try - using only RandomForest and GradientBoosting to avoid compatibility issues
models = {
    'RandomForest': RandomForestRegressor(random_state=42, n_estimators=200, max_depth=15),
    'GradientBoosting': GradientBoostingRegressor(random_state=42, n_estimators=200, learning_rate=0.1, max_depth=5),
    'ElasticNet': ElasticNet(random_state=42, alpha=0.5, l1_ratio=0.5)
}

# Cross-validation strategy (use GroupKFold to prevent data leakage between companies)
print("\nEvaluating models with cross-validation...")
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

# Skip grid search for simplicity and to avoid compatibility issues
print("\nTraining best model...")
best_model = Pipeline([
    ('preprocessor', preprocessor),
    ('model', models[best_model_name])
])

# Train the best model on the full training data
best_model.fit(X_train, y_train)

# 7. Generate Predictions and Create Submission File
print("\nGenerating predictions...")
# Make predictions with the best model
predictions = best_model.predict(X_test)

# Create submission file
submission = pd.DataFrame({
    'ID': test_df['RowID'],
    'Sales': predictions
})

# Ensure predictions are non-negative
submission['Sales'] = submission['Sales'].clip(lower=0)

# Save to CSV
submission.to_csv('submission.csv', index=False)
print("\nSubmission file created: submission.csv")

print("\nB2B Sales Forecasting project completed successfully!")