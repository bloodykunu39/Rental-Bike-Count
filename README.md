# Rental-Bike-Count

A/B Testing
- Conducted one tailed t-test to check if the mean of hourly bike rentals reduces if "Snowfall" is non zero in winter season.
- Conducted 1 way ANOVA test to check if the mean hourly count of bike rentals in the four quartiles are different .
- Conducted Chi-squared goodness of fit test to identify if the two distributions are different using Chi-squared test.

# XGBoost Count Prediction
## Data preprocessing
### Feature engineering
  * Checked for null and duplicates in the data.
  * Added weekend data, month data.
  * Identified the datatypes.
  * Checked whether all int/float datatypes falls into numerical feature category.
  * Seperated features into numerical and categorical.

## Data visualization and encoding  
1. Visualized how the rented bike count varies hourly for different categorical features.
2. Visualized the rented bike count outliers for each categorical features.
3. Viualtized the variation in the data distribution for each numerical features. Also, mark/show the mean and median of the distibution in the plot.
4. Visualized the outliers in each numerical feature data.
5. created a regression plot to know relation between dependent and independent nuumerical variables.
6. Visualized the correlation between different numerical features using heat map. Find and remove correlated features for a threshold value of 0.7 (correlation>0.7).
7. Encoding categorical features (**Use of pipelines**).
  * One-hot encoding for seasons.
  * Numerical encoding (1 or 0) for categorical features with 2 unique values.
  * Used numerical identifier for other categorical features (eg. month: January-1, march-3)
8. Deleted non-relevant feautes from the dataframe and comment.
9. Visualized the dependent variable data-distribution and check for skewness. The regression assumes that the dependent variable has a nearly normal distribution, therefore, to meet this assumption, make some measures to normalize the distribution.

## Build, train, test and optimize the model
1. Seperate target and features from the data.
2. Do 80:20 train test split.
3. Train the XGBoost regression model with default parameters.
4. Tune the hyperparameters of the model using Optuna. Save the best hyperparameters.

