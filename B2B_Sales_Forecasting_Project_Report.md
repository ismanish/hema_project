# B2B Sales Forecasting Project Report
## Steel Manufacturing Company

### Executive Summary

This report presents a machine learning solution developed to forecast quarterly sales for a steel manufacturing company that supplies to various B2B customers across Auto, Metal Fabrication, and Infrastructure sectors. Our approach integrates company-specific metrics with macroeconomic indicators to create a robust forecasting model, achieving a cross-validation Mean Absolute Error (MAE) of 1428.07.

The model identifies financial liquidity (QuickRatio) and inventory management (InventoryRatio) as the primary drivers of sales performance, accounting for over 53% of predictive power. This insight provides actionable business intelligence for optimizing sales strategies.

### Problem Definition

The challenge was to predict quarterly sales for 75 unique B2B customers based on:
- Company-specific financial and operational metrics
- Industry and regional factors
- Macroeconomic indicators

This forecasting problem is critical for resource planning, inventory management, and financial projections in the volatile steel manufacturing sector.

### Dataset Analysis

**Data Sources:**
- `train.csv`: 525 historical sales records
- `test.csv`: 150 records requiring sales predictions
- `EconomicIndicators.csv`: 28 monthly economic data points

**Key Variables:**
1. **Company Metrics**: QuickRatio, InventoryRatio, RevenueGrowth, MarketshareChange
2. **Financial Ratings**: Bond ratings, Stock ratings
3. **Categorical Features**: Region, Industry
4. **Target Variable**: Sales

**Data Characteristics:**
- Time periods: 9 quarters (Q1-Q9)
- Geographic distribution across multiple regions
- Industry segments: Auto, Metal Fabrication, and Infrastructure

### Methodology

#### 1. Data Preprocessing
- Merged quarterly company data with relevant economic indicators
- Created temporal mapping between company quarters and economic months
- Encoded categorical variables (bond ratings, stock ratings, regions, industries)
- Implemented feature engineering to capture financial relationships

#### 2. Feature Engineering
- Created economic indicator averages for each business quarter
- Encoded ordinal variables (bond ratings from AAA-CCC, stock ratings)
- Generated numerical representations of categorical variables

#### 3. Model Development
We implemented a Random Forest Regression model with the following parameters:
- 200 decision trees
- Maximum depth of 10
- Minimum samples split of 5
- Minimum samples leaf of 2

This approach was selected for its:
- Ability to capture non-linear relationships
- Robustness to outliers
- Feature importance quantification
- Strong performance on heterogeneous data

#### 4. Validation Strategy
- 5-fold cross-validation to ensure model robustness
- Mean Absolute Error (MAE) as the evaluation metric
- Analysis of feature importance for business insights

### Results and Findings

#### Model Performance
- **Cross-validation MAE**: 1428.07 ± 120.12
- **Training MAE**: 743.70

#### Feature Importance
The model revealed that company financial metrics were the strongest predictors:

| Feature | Importance (%) |
|---------|---------------|
| QuickRatio | 31.65% |
| InventoryRatio | 22.30% |
| RevenueGrowth | 9.26% |
| Bond Rating | 8.67% |
| Stock Rating | 8.45% |
| Marketshare | 7.18% |
| Money Supply | 2.26% |
| North Economic Activity Index | 2.24% |
| PMI | 1.91% |
| Interest Rate | 1.64% |

#### Key Insights
1. **Liquidity is Paramount**: A company's QuickRatio (ability to pay short-term obligations) is the strongest predictor of future sales, suggesting that financially healthy companies maintain stronger sales performance.

2. **Inventory Management Matters**: InventoryRatio has a substantial impact on sales outcomes, highlighting the importance of efficient supply chain operations.

3. **Growth Signals**: RevenueGrowth projections correlate significantly with actual sales, confirming the value of growth metrics in forecasting.

4. **Financial Ratings**: Combined bond and stock ratings account for approximately 17% of predictive power, demonstrating that external financial evaluations provide valuable signals.

5. **Economic Environment**: While company-specific metrics dominate, macroeconomic indicators still contribute meaningfully to the model, particularly Money Supply and regional Economic Activity Indices.

### Business Implications

1. **Credit Risk Assessment**: The strong correlation between QuickRatio and sales suggests that credit departments should prioritize this metric when evaluating customer relationships.

2. **Supply Chain Optimization**: Given the importance of InventoryRatio, sales forecasting should be integrated with inventory management systems.

3. **Portfolio Management**: Sales resources could be allocated more efficiently by prioritizing customers with stronger financial health indicators.

4. **Economic Monitoring**: While company-specific metrics are most important, sales teams should still track key economic indicators that influence customer purchasing behavior.

### Limitations and Future Work

**Current Limitations:**
- Limited historical data (only 9 quarters)
- Potential for model overfitting as indicated by the gap between training and cross-validation MAE
- Missing potential lead/lag relationships between economic indicators and sales

**Future Enhancements:**
1. **Time Series Modeling**: Explore specialized time series methods to better capture temporal patterns
2. **Ensemble Approaches**: Combine multiple model types to improve prediction accuracy
3. **Customer Segmentation**: Develop segment-specific models for different industries or regions
4. **Additional Features**: Incorporate market-specific factors like steel price indices and competitor activities
5. **Hyperparameter Tuning**: Further optimize model parameters through grid search

### Conclusion

The developed sales forecasting model demonstrates strong predictive capability with an MAE of 1428.07, identifying company financial health and inventory management as the primary sales drivers in the B2B steel manufacturing context.

By quantifying the impact of various factors on sales performance, this model provides actionable insights for optimizing customer relationship management, inventory planning, and sales resource allocation. The findings suggest that prioritizing financially healthy customers and efficient inventory management could significantly improve sales outcomes.

---

### Appendix: Technical Implementation

The solution was implemented in Python using the following libraries:
- pandas and numpy for data manipulation
- scikit-learn for machine learning models
- matplotlib and seaborn for visualization

The full implementation is available in the `sales_forecasting_solution.py` file, which includes data preprocessing, model training, evaluation, and prediction generation.
