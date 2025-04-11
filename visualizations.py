# B2B Sales Forecasting Visualizations
# ===================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.inspection import permutation_importance
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
import os

# Create output directory for visualizations
os.makedirs('visualizations', exist_ok=True)

# Set plot style
plt.style.use('ggplot')
sns.set_palette("Set2")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Load datasets
print("Loading datasets...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
economic_df = pd.read_csv('EconomicIndicators.csv')

# Function to prepare the data (Same as in the main script)
def prepare_data():
    # Extract quarter number from Quarter column
    train_df['QuarterNum'] = train_df['Quarter'].apply(lambda x: int(x.replace('Q', '')))
    test_df['QuarterNum'] = test_df['Quarter'].apply(lambda x: int(x.replace('Q', '')))
    
    # Convert bond ratings to numeric scores
    bond_mapping = {
        'AAA': 10, 'AA+': 9.5, 'AA': 9, 'AA-': 8.5,
        'A+': 8, 'A': 7.5, 'A-': 7, 'BBB+': 6.5,
        'BBB': 6, 'BBB-': 5.5, 'BB+': 5, 'BB': 4.5,
        'BB-': 4, 'B+': 3.5, 'B': 3, 'B-': 2.5,
        'CCC+': 2, 'CCC': 1.5, 'CCC-': 1, 'D': 0.5
    }
    
    train_df['BondScore'] = train_df['Bond rating'].map(bond_mapping)
    test_df['BondScore'] = test_df['Bond rating'].map(bond_mapping)
    
    # Convert stock ratings to numeric scores
    stock_mapping = {
        'Strong Buy': 5, 'Buy': 4, 'Hold': 3, 'Sell': 2, 'Strong Sell': 1
    }
    
    train_df['StockScore'] = train_df['Stock rating'].map(stock_mapping)
    test_df['StockScore'] = test_df['Stock rating'].map(stock_mapping)
    
    # Create a combined rating score
    train_df['CombinedScore'] = train_df['BondScore'] * train_df['StockScore']
    test_df['CombinedScore'] = test_df['BondScore'] * test_df['StockScore']
    
    # Fill missing values
    train_df['InventoryRatio'] = train_df['InventoryRatio'].fillna(train_df['InventoryRatio'].median())
    test_df['InventoryRatio'] = test_df['InventoryRatio'].fillna(test_df['InventoryRatio'].median())
    
    return train_df, test_df, economic_df

# Prepare the data
train_df, test_df, economic_df = prepare_data()

# 1. Distribution of Sales across Industries
def plot_sales_distribution_by_industry():
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='Industry', y='Sales', data=train_df)
    plt.title('Sales Distribution by Industry', fontsize=16)
    plt.xlabel('Industry', fontsize=14)
    plt.ylabel('Sales', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('visualizations/1_sales_distribution_by_industry.png', dpi=300)
    plt.close()

# 2. Sales Trends over Quarters by Industry
def plot_sales_trends_by_industry():
    # Aggregate sales by quarter and industry
    quarterly_sales = train_df.groupby(['Quarter', 'Industry'])['Sales'].mean().reset_index()
    quarterly_sales['QuarterNum'] = quarterly_sales['Quarter'].apply(lambda x: int(x.replace('Q', '')))
    quarterly_sales = quarterly_sales.sort_values('QuarterNum')
    
    plt.figure(figsize=(14, 8))
    industries = train_df['Industry'].unique()
    
    for industry in industries:
        industry_data = quarterly_sales[quarterly_sales['Industry'] == industry]
        plt.plot(industry_data['QuarterNum'], industry_data['Sales'], marker='o', linewidth=2, label=industry)
    
    plt.title('Average Sales Trends by Industry Over Time', fontsize=16)
    plt.xlabel('Quarter', fontsize=14)
    plt.ylabel('Average Sales', fontsize=14)
    plt.xticks(quarterly_sales['QuarterNum'].unique(), [f'Q{q}' for q in quarterly_sales['QuarterNum'].unique()], rotation=45)
    plt.legend(title='Industry', title_fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('visualizations/2_sales_trends_by_industry.png', dpi=300)
    plt.close()

# 3. Correlation Heatmap of Key Features
def plot_correlation_heatmap():
    # Select numerical features for correlation analysis
    features = ['Sales', 'QuickRatio', 'InventoryRatio', 'RevenueGrowth', 
                'Marketshare', 'BondScore', 'StockScore', 'CombinedScore', 'QuarterNum']
    
    # Calculate correlation matrix
    corr_matrix = train_df[features].corr()
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', 
                mask=mask, vmin=-1, vmax=1, center=0, 
                square=True, linewidths=.5)
    
    plt.title('Correlation Matrix of Key Features', fontsize=16)
    plt.tight_layout()
    plt.savefig('visualizations/3_correlation_heatmap.png', dpi=300)
    plt.close()

# 4. Regional Sales Performance
def plot_regional_sales():
    # Aggregate sales by region
    region_sales = train_df.groupby('Region')['Sales'].agg(['mean', 'median', 'std']).reset_index()
    
    plt.figure(figsize=(12, 8))
    x = np.arange(len(region_sales['Region']))
    width = 0.3
    
    plt.bar(x - width/2, region_sales['mean'], width, label='Mean Sales')
    plt.bar(x + width/2, region_sales['median'], width, label='Median Sales')
    
    # Add error bars
    plt.errorbar(x - width/2, region_sales['mean'], yerr=region_sales['std']/2, 
                 fmt='none', ecolor='black', capsize=5)
    
    plt.title('Regional Sales Performance', fontsize=16)
    plt.xlabel('Region', fontsize=14)
    plt.ylabel('Sales', fontsize=14)
    plt.xticks(x, region_sales['Region'])
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.tight_layout()
    plt.savefig('visualizations/4_regional_sales.png', dpi=300)
    plt.close()

# 5. Impact of Bond and Stock Ratings on Sales
def plot_ratings_impact():
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Bond Rating vs Sales
    sns.boxplot(x='Bond rating', y='Sales', data=train_df, ax=axes[0])
    axes[0].set_title('Bond Rating vs Sales', fontsize=14)
    axes[0].set_xlabel('Bond Rating', fontsize=12)
    axes[0].set_ylabel('Sales', fontsize=12)
    axes[0].tick_params(axis='x', rotation=90)
    
    # Stock Rating vs Sales
    sns.boxplot(x='Stock rating', y='Sales', data=train_df, ax=axes[1])
    axes[1].set_title('Stock Rating vs Sales', fontsize=14)
    axes[1].set_xlabel('Stock Rating', fontsize=12)
    axes[1].set_ylabel('Sales', fontsize=12)
    
    # Combined Score vs Sales
    sns.scatterplot(x='CombinedScore', y='Sales', hue='Industry', 
                   data=train_df, palette='deep', ax=axes[2])
    axes[2].set_title('Combined Score vs Sales', fontsize=14)
    axes[2].set_xlabel('Combined Score (Bond × Stock)', fontsize=12)
    axes[2].set_ylabel('Sales', fontsize=12)
    axes[2].legend(title='Industry', loc='upper right')
    
    plt.tight_layout()
    plt.savefig('visualizations/5_ratings_impact.png', dpi=300)
    plt.close()

# 6. Economic Indicators Trend
def plot_economic_indicators_trend():
    fig, axes = plt.subplots(3, 2, figsize=(16, 15))
    axes = axes.flatten()
    
    indicators = ['Consumer Sentiment', 'Interest Rate', 'PMI', 
                  'Money Supply', 'NationalEAI']
    
    for i, indicator in enumerate(indicators):
        if i < len(axes):
            sns.lineplot(x='Month', y=indicator, data=economic_df, 
                        marker='o', linewidth=2, ax=axes[i])
            axes[i].set_title(f'{indicator} Over Time', fontsize=14)
            axes[i].set_xlabel('Month', fontsize=12)
            axes[i].set_ylabel(indicator, fontsize=12)
            axes[i].grid(True, linestyle='--', alpha=0.7)
    
    # Last subplot for regional EAI comparison
    regional_indicators = ['EastEAI', 'WestEAI', 'SouthEAI', 'NorthEAI']
    for indicator in regional_indicators:
        sns.lineplot(x='Month', y=indicator, data=economic_df, 
                    marker='o', linewidth=2, ax=axes[5], label=indicator.replace('EAI', ''))
    
    axes[5].set_title('Regional Economic Activity Indices', fontsize=14)
    axes[5].set_xlabel('Month', fontsize=12)
    axes[5].set_ylabel('Economic Activity Index', fontsize=12)
    axes[5].legend(title='Region')
    axes[5].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('visualizations/6_economic_indicators_trend.png', dpi=300)
    plt.close()

# 7. Train a simple model for feature importance
def plot_feature_importance():
    # Prepare features for Random Forest
    numeric_features = ['QuickRatio', 'InventoryRatio', 'RevenueGrowth', 'Marketshare',
                        'BondScore', 'StockScore', 'CombinedScore', 'QuarterNum']
    
    categorical_features = ['Region', 'Industry']
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )
    
    # Create and train a Random Forest model
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    # Fit the model
    model.fit(train_df[numeric_features + categorical_features], train_df['Sales'])
    
    # Extract feature names after one-hot encoding
    feature_names = (
        numeric_features +
        list(model.named_steps['preprocessor']
            .transformers_[1][1]
            .get_feature_names_out(categorical_features))
    )
    
    # Get feature importances
    importances = model.named_steps['model'].feature_importances_
    
    # Create DataFrame for plotting
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False).head(15)
    
    # Plot feature importances
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
    plt.title('Top 15 Feature Importances for Sales Prediction', fontsize=16)
    plt.xlabel('Importance', fontsize=14)
    plt.ylabel('Feature', fontsize=14)
    plt.tight_layout()
    plt.savefig('visualizations/7_feature_importance.png', dpi=300)
    plt.close()
    
    # Also compute permutation importance for validation
    try:
        print("Computing permutation importance...")
        X = train_df[numeric_features + categorical_features]
        y = train_df['Sales']
        X_preprocessed = preprocessor.transform(X)
        rf_model = model.named_steps['model']
        
        perm_importance = permutation_importance(rf_model, X_preprocessed, y, 
                                               n_repeats=10, random_state=42)
        
        sorted_idx = perm_importance.importances_mean.argsort()[-15:]
        
        plt.figure(figsize=(12, 8))
        plt.barh(np.array(feature_names)[sorted_idx], 
                perm_importance.importances_mean[sorted_idx])
        plt.title('Permutation Feature Importance (Validation)', fontsize=16)
        plt.xlabel('Permutation Importance', fontsize=14)
        plt.tight_layout()
        plt.savefig('visualizations/7b_permutation_importance.png', dpi=300)
        plt.close()
    except Exception as e:
        print(f"Error computing permutation importance: {e}")

# 8. Sales Distribution by Company Size (based on Marketshare)
def plot_sales_by_company_size():
    # Create company size categories based on market share percentiles
    train_df['CompanySize'] = pd.qcut(train_df['Marketshare'], 
                                     q=[0, 0.25, 0.5, 0.75, 1.0], 
                                     labels=['Small', 'Medium', 'Large', 'Very Large'])
    
    plt.figure(figsize=(12, 8))
    sns.violinplot(x='CompanySize', y='Sales', data=train_df, inner='box')
    plt.title('Sales Distribution by Company Size (Market Share)', fontsize=16)
    plt.xlabel('Company Size', fontsize=14)
    plt.ylabel('Sales', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.tight_layout()
    plt.savefig('visualizations/8_sales_by_company_size.png', dpi=300)
    plt.close()

# 9. Inventory Ratio vs Sales by Industry
def plot_inventory_vs_sales():
    plt.figure(figsize=(14, 8))
    sns.scatterplot(x='InventoryRatio', y='Sales', hue='Industry', 
                   size='Marketshare', sizes=(20, 200),
                   data=train_df)
    
    plt.title('Inventory Ratio vs Sales by Industry and Market Share', fontsize=16)
    plt.xlabel('Inventory Ratio', fontsize=14)
    plt.ylabel('Sales', fontsize=14)
    plt.legend(title='Industry', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('visualizations/9_inventory_vs_sales.png', dpi=300)
    plt.close()

# 10. QuickRatio vs Sales with Revenue Growth
def plot_quickratio_vs_sales():
    plt.figure(figsize=(14, 8))
    scatter = plt.scatter(x='QuickRatio', y='Sales', c='RevenueGrowth', 
                        cmap='coolwarm', s=100, alpha=0.7,
                        data=train_df)
    
    plt.colorbar(scatter, label='Revenue Growth')
    plt.title('Quick Ratio vs Sales colored by Revenue Growth', fontsize=16)
    plt.xlabel('Quick Ratio', fontsize=14)
    plt.ylabel('Sales', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('visualizations/10_quickratio_vs_sales.png', dpi=300)
    plt.close()

# 11. Prediction vs Actual (Cross-Validation)
def plot_prediction_vs_actual():
    # Prepare features
    numeric_features = ['QuickRatio', 'InventoryRatio', 'RevenueGrowth', 'Marketshare',
                       'BondScore', 'StockScore', 'CombinedScore', 'QuarterNum']
    categorical_features = ['Region', 'Industry']
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )
    
    # Create a Random Forest model
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    # Simple train-test split for visualization purposes
    from sklearn.model_selection import train_test_split
    X = train_df[numeric_features + categorical_features]
    y = train_df['Sales']
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Predict on validation set
    y_pred = model.predict(X_val)
    
    # Calculate MAE
    mae = mean_absolute_error(y_val, y_pred)
    
    # Plot predicted vs actual
    plt.figure(figsize=(12, 8))
    plt.scatter(y_val, y_pred, alpha=0.6)
    
    # Add perfect prediction line
    max_value = max(max(y_val), max(y_pred))
    min_value = min(min(y_val), min(y_pred))
    plt.plot([min_value, max_value], [min_value, max_value], 'r--')
    
    plt.title(f'Predicted vs Actual Sales (MAE: {mae:.2f})', fontsize=16)
    plt.xlabel('Actual Sales', fontsize=14)
    plt.ylabel('Predicted Sales', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.annotate(f'MAE: {mae:.2f}', xy=(0.05, 0.95), xycoords='axes fraction', 
                fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))
    plt.tight_layout()
    plt.savefig('visualizations/11_prediction_vs_actual.png', dpi=300)
    plt.close()

# 12. Seasonal Patterns in Sales
def plot_seasonal_patterns():
    # Group by quarter and calculate average sales
    quarterly_sales = train_df.groupby('Quarter')['Sales'].mean().reset_index()
    quarterly_sales['QuarterNum'] = quarterly_sales['Quarter'].apply(lambda x: int(x.replace('Q', '')))
    quarterly_sales = quarterly_sales.sort_values('QuarterNum')
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    # Line plot of average sales by quarter
    sns.lineplot(x='QuarterNum', y='Sales', data=quarterly_sales, 
                marker='o', linewidth=2, color='blue', ax=ax1)
    ax1.set_title('Average Sales by Quarter', fontsize=16)
    ax1.set_xlabel('Quarter', fontsize=14)
    ax1.set_ylabel('Average Sales', fontsize=14)
    ax1.set_xticks(quarterly_sales['QuarterNum'])
    ax1.set_xticklabels([f'Q{q}' for q in quarterly_sales['QuarterNum']])
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Heatmap of average sales by quarter and industry
    industry_quarter_sales = train_df.groupby(['Industry', 'Quarter'])['Sales'].mean().reset_index()
    industry_quarter_sales['QuarterNum'] = industry_quarter_sales['Quarter'].apply(lambda x: int(x.replace('Q', '')))
    
    # Pivot for heatmap
    heatmap_data = industry_quarter_sales.pivot(index='Industry', columns='Quarter', values='Sales')
    
    # Sort columns by quarter number
    heatmap_data = heatmap_data[[f'Q{i}' for i in range(1, 10) if f'Q{i}' in heatmap_data.columns]]
    
    # Create heatmap
    sns.heatmap(heatmap_data, annot=True, fmt=".0f", cmap='YlGnBu', ax=ax2, cbar_kws={'label': 'Average Sales'})
    ax2.set_title('Average Sales by Industry and Quarter', fontsize=16)
    ax2.set_xlabel('Quarter', fontsize=14)
    ax2.set_ylabel('Industry', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('visualizations/12_seasonal_patterns.png', dpi=300)
    plt.close()

# Run all visualizations
def generate_all_visualizations():
    print("Generating sales distribution by industry...")
    plot_sales_distribution_by_industry()
    
    print("Generating sales trends by industry...")
    plot_sales_trends_by_industry()
    
    print("Generating correlation heatmap...")
    plot_correlation_heatmap()
    
    print("Generating regional sales performance...")
    plot_regional_sales()
    
    print("Generating ratings impact visualization...")
    plot_ratings_impact()
    
    print("Generating economic indicators trend...")
    plot_economic_indicators_trend()
    
    print("Generating feature importance plot...")
    plot_feature_importance()
    
    print("Generating sales by company size plot...")
    plot_sales_by_company_size()
    
    print("Generating inventory vs sales plot...")
    plot_inventory_vs_sales()
    
    print("Generating quick ratio vs sales plot...")
    plot_quickratio_vs_sales()
    
    print("Generating prediction vs actual plot...")
    plot_prediction_vs_actual()
    
    print("Generating seasonal patterns plot...")
    plot_seasonal_patterns()
    
    print("All visualizations generated successfully!")

if __name__ == "__main__":
    generate_all_visualizations()