#!/usr/bin/env python3
# B2B Sales Forecasting Solution for Steel Manufacturing Company
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

def load_data():
    """Load and prepare the datasets"""
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    economic = pd.read_csv('EconomicIndicators.csv')
    return train, test, economic

def map_quarter_to_months(quarter):
    """Map quarter to corresponding months"""
    quarter_num = int(quarter[1])
    # Q1 corresponds to months 1-3, Q2 to 4-6, etc.
    start_month = (quarter_num - 1) * 3 + 1
    return list(range(start_month, start_month + 3))

def preprocess_data(train, test, economic):
    """Preprocess the data for modeling"""
    # Map quarters to economic data
    def add_economic_features(df):
        df_copy = df.copy()
        # Extract quarter number and create a feature
        df_copy['quarter_num'] = df_copy['Quarter'].str[1].astype(int)
        
        # Create mappings for each quarter to corresponding months
        quarterly_eco_data = {}
        for q in range(1, 10):  # Q1 to Q9
            quarter = f"Q{q}"
            months = map_quarter_to_months(quarter)
            
            # Filter economic data for these months (only use months that exist in the data)
            valid_months = [m for m in months if m <= len(economic)]
            if valid_months:
                # Average the economic indicators for the quarter
                quarter_eco = economic[economic['Month'].isin(valid_months)].mean(numeric_only=True)
                quarterly_eco_data[quarter] = quarter_eco
        
        # Add economic indicators as features
        for quarter, eco_data in quarterly_eco_data.items():
            mask = df_copy['Quarter'] == quarter
            for col in economic.columns[1:]:  # Skip 'Month' column
                df_copy.loc[mask, f'eco_{col}'] = eco_data[col]
        
        return df_copy
    
    # Apply economic features to both train and test
    train_processed = add_economic_features(train)
    test_processed = add_economic_features(test)
    
    # Feature engineering
    def engineer_features(df):
        df_copy = df.copy()
        
        # Create lag features for companies (if possible)
        companies = df_copy['Company'].unique()
        
        # Fill missing values in economic features
        eco_cols = [col for col in df_copy.columns if col.startswith('eco_')]
        df_copy[eco_cols] = df_copy[eco_cols].fillna(df_copy[eco_cols].mean())
        
        # Encode categorical variables
        df_copy['Bond_rating_encoded'] = df_copy['Bond rating'].map({
            'AAA': 7, 'AA': 6, 'A': 5, 'BBB': 4, 'BB': 3, 'B': 2, 'CCC': 1
        })
        
        df_copy['Stock_rating_encoded'] = df_copy['Stock rating'].map({
            'Buy': 3, 'Hold': 2, 'Sell': 1
        })
        
        return df_copy
    
    train_processed = engineer_features(train_processed)
    test_processed = engineer_features(test_processed)
    
    return train_processed, test_processed

def build_model(train_processed):
    """Build and train the forecasting model"""
    # Prepare features and target
    X = train_processed.drop(['Sales', 'Quarter', 'Company', 'Bond rating', 'Stock rating', 'Region', 'Industry'], axis=1)
    y = train_processed['Sales']
    
    # Define categorical and numerical features
    categorical_features = ['Region', 'Industry']
    numerical_features = [col for col in X.columns if col not in categorical_features]
    
    print(f"Training with {len(numerical_features)} numerical features")
    
    # Create preprocessing pipelines
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features),
            ('num', numerical_transformer, numerical_features)
        ])
    
    # Replace XGBoost with RandomForest which has better compatibility
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    # Manual cross-validation to avoid compatibility issues
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mae_scores = []
    
    print("Performing cross-validation...")
    for train_idx, valid_idx in kf.split(X):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_valid)
        mae = mean_absolute_error(y_valid, y_pred)
        mae_scores.append(mae)
    
    print(f"Cross-validation MAE: {np.mean(mae_scores):.2f} ± {np.std(mae_scores):.2f}")
    
    # Train on full dataset
    print("Training final model on full dataset...")
    model.fit(X, y)
    
    return model, X.columns

def make_predictions(model, feature_cols, train_processed, test_processed):
    """Make predictions on the test data"""
    # Prepare test features
    X_test = test_processed[feature_cols]
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Create submission dataframe
    submission = pd.DataFrame({
        'ID': test_processed['RowID'],
        'Sales': predictions
    })
    
    return submission

def evaluate_and_visualize(train_processed, model, feature_cols):
    """Evaluate model performance and visualize important features"""
    # Prepare features and target
    X = train_processed[feature_cols]
    y = train_processed['Sales']
    
    # Make predictions on training data
    y_pred = model.predict(X)
    
    # Calculate MAE
    mae = mean_absolute_error(y, y_pred)
    print(f"Training MAE: {mae:.2f}")
    
    # Visualize actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y, y_pred, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.xlabel('Actual Sales')
    plt.ylabel('Predicted Sales')
    plt.title('Actual vs Predicted Sales')
    plt.savefig('actual_vs_predicted.png')
    
    # Visualize feature importance
    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    
    return feature_importance

def main():
    """Main execution function"""
    print("Loading data...")
    train, test, economic = load_data()
    
    print("Preprocessing data...")
    train_processed, test_processed = preprocess_data(train, test, economic)
    
    print("Building and training model...")
    model, feature_cols = build_model(train_processed)
    
    print("Evaluating model...")
    feature_importance = evaluate_and_visualize(train_processed, model, feature_cols)
    print("\nTop 10 most important features:")
    print(feature_importance.head(10))
    
    print("\nMaking predictions on test data...")
    submission = make_predictions(model, feature_cols, train_processed, test_processed)
    
    print("Saving submission file...")
    submission.to_csv('submission.csv', index=False)
    print("Done!")

if __name__ == "__main__":
    main()
