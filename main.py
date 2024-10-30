# Case Study with Linear Regression
from sys import prefix

# Causal Analysis
# Predictive Analytics

# What are the features that define the californian house values

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', 1000)  # Increase width to fit all columns

file_path = "../../datasets/housing.csv"

data = pd.read_csv(file_path)

data

data.columns

data.head(10)

# Data Exploration
data.info()

data["ocean_proximity"].unique()

# Identifying and Removing Missing Data
missing_values = data.isnull().sum()

missing_percentage = (missing_values / len(data)) * 100

print("Missing Values in Each Column:\n", missing_values)
print("Percentage of Missing Data:\n", missing_percentage)

# Remove rows with missing values
data_cleaned = data.dropna()

# Verify that missing values have been removed
print("Missing values in each column after removal:\n",data_cleaned.isnull().sum())

# Descriptive Statistics
data.describe()

# Data Visualization - Histogram
sns.set(style="whitegrid")      # setting background
plt.figure(figsize=(12,6))
sns.histplot(data_cleaned["median_house_value"], color="forestgreen", kde=True)
plt.title("Distribution of Median House Values")
plt.xlabel("Median House Value")
plt.ylabel("Frequency")
plt.show()

# Inter-Quantile-Range [IQR] for removing Outliers
# Assuming "data" is your DataFrame and "median_house_value" is the column of interest
Q1 = data_cleaned["median_house_value"].quantile(0.25)
print(Q1)
Q3 = data_cleaned["median_house_value"].quantile(0.75)
print(Q3)
IQR = Q3 - Q1

# Define the bounds for the outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Remove outliers
data_no_outliers_1 = data_cleaned[(data_cleaned["median_house_value"] >= lower_bound) & (data_cleaned["median_house_value"] <= upper_bound)]

# Check the shape of the data before after removal of outliers
print("Original data shape:", data_cleaned.shape)
print("New data shape without outliers:", data_no_outliers_1.shape)

# BoxPlot for removing Outliers
plt.figure(figsize=(10,6))
sns.boxplot(x=data_no_outliers_1["median_income"], color="purple")
plt.title("Outlier Analysis in Median Income")
plt.xlabel("Median Income")
plt.show()

# Calculate Q1 and Q3
Q1 = data_no_outliers_1["median_income"].quantile(0.25)
Q3 = data_no_outliers_1["median_income"].quantile(0.75)
IQR = Q3 - Q1

# Define the bounds for the outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Remove outliers
data_no_outliers_2 = data_no_outliers_1[(data_no_outliers_1["median_income"] >= lower_bound) & (data_no_outliers_1["median_income"] <= upper_bound)]

# Check the shape of the data before after removal of outliers
print("Original data shape:", data_no_outliers_1.shape)
print("New data shape without outliers:", data_no_outliers_2.shape)

data = data_no_outliers_2

# Drop non-numeric columns for correlation analysis
data = data.select_dtypes(include=[np.number])  # Select only numeric columns

# Correlation Analysis - HeatMap
plt.figure(figsize=(12,8))
sns.heatmap(data.corr(), annot=True, cmap="Greens")
plt.title("Correlation Heatmap of Housing Data")
plt.show()

# String Data Categorization to Dummy Variables
string_data = data_no_outliers_2

for column in ["ocean_proximity"]: # add other categorical columns if any
    print(f"Unique values in {column}:", string_data[column].unique())

ocean_proximity_dummies = pd.get_dummies(string_data["ocean_proximity"], prefix="ocean_proximity")
string_data = pd.concat([string_data.drop("ocean_proximity", axis=1), ocean_proximity_dummies], axis=1)
print(ocean_proximity_dummies)

print(string_data.columns)

data = string_data.drop(columns="ocean_proximity_ISLAND")

data[['ocean_proximity_<1H OCEAN', 'ocean_proximity_INLAND',
      'ocean_proximity_NEAR BAY', 'ocean_proximity_NEAR OCEAN']] = data[['ocean_proximity_<1H OCEAN', 'ocean_proximity_INLAND',
                                                                         'ocean_proximity_NEAR BAY', 'ocean_proximity_NEAR OCEAN']].astype(int)
print(data.columns)
# Splitting Data to Train and Test
# Define your features (independent variables) and target (dependent variable)
features = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
            'total_bedrooms', 'population', 'households', 'median_income',
            'median_house_value', 'ocean_proximity_<1H OCEAN', 'ocean_proximity_INLAND',
            'ocean_proximity_NEAR BAY', 'ocean_proximity_NEAR OCEAN']
target = ["median_house_value"]

X = data[features]
y = data[target]

# Split the data into a training set and a testing set
# test_size specifies the proportion of the data to be included in the test split
# random_state ensures reproducibility of you split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1111) # 80 train 20 test

# Check the size of the splits
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")

# Training & Testing Linear Regression Model
# Adding a constant to the predictors because statsmodels' OLS doesn't include it by default
X_train_const = sm.add_constant(X_train)
print(X_train_const)

# Fit the 0LS model
model_fitted = sm.OLS(y_train, X_train_const).fit()

# Printing Summary
print(model_fitted.summary())

# Prediction/Testing
# Adding a constant to the test predictors
X_test_const = sm.add_constant(X_test)
print(X_test_const)

# Making predictions on the test set
test_predictions = model_fitted.predict(X_test_const)
print(test_predictions)


# Checking 0LS Assumtions
# Assumtion 1: Linearity
# Scatter plott for observed vs predicted values on test data
plt.scatter(y_test, test_predictions, color = "forestgreen")
plt.xlabel('Observed Values')
plt.ylabel('Predicted Values')
plt.title('Observed vs Predicted Values on Test Data')
plt.plot(y_test, y_test, color='darkred')  # line for perfect prediction (true values)
plt.show()

"""
Positive Linear Relationship: The red line (which represents a perfect prediction line) 
and the distribution of the data points suggest there's a positive linear relationship 
between the observed and predicted values. This means that as the actual values increase, 
the predicted values also increase, which is a good sign for linearity.
"""


# Assumtion 2: Random Sample
# Calculate the mean of the residuals
mean_residuals = np.mean(model_fitted.resid)
print(f"The mean of the residuals is {np.round(mean_residuals,2)}")

"""
While we cannot directly observe the true errors in the model, we can work with the residuals, 
which are the differences between the observed values and the predicted values from the model. 
If the model is well-fitted, the residuals should be randomly scattered around zero without any systematic patterns.
"""

# Plotting the residuals
plt.scatter(model_fitted.fittedvalues, model_fitted.resid, color = "forestgreen")
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted Values')
plt.show()

"""
In this plot, we don't see any discernible patterns. The residuals are thus randomly 
distributed around the horizontal line at zero, with no clear shape or trend. If there's a pattern, 
or if the residuals show a systematic deviation from zero, it could suggest issues such as model 
misspecification, non-linearity, or omitted variable bias.
"""

# Assumtion 3: Exogeneity
# Calculate the residuals
residuals = model_fitted.resid

# Check for correlation between residuals and each predictor
for column in X_train.columns:
    corr_coefficient = np.corrcoef(X_train[column], residuals)[0, 1]
    print(f'Correlation between residuals and {column}: {np.round(corr_coefficient,2)}')

"""
Durbin-Wu-Hausman Test: For a more formal statistical test, use the Durbin-Wu-Hausman test. 
This involves comparing your model with one that includes an instrumental variable. 
This test checks whether the coefficients of the model change significantly when the potentially 
endogenous variables are instrumented. This test is a more advanced, econometrical approach and requires 
identification of suitable instruments, which is not always straightforward.
"""

# Assumtion 4: Homoskedasticty
# Plotting the residuals
plt.scatter(model_fitted.fittedvalues, model_fitted.resid, color = "forestgreen")
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted Values')
plt.show()

"""
Random Scatter: If plot shows a random scatter of residuals around the horizontal line at zero, 
it supports the homoscedasticity assumption.

Pattern or Shape: If residuals display a pattern (such as a curve) or form a funnel shape where the 
spread increases with fitted values, this would suggest heteroscedasticity, meaning variance of 
residuals changes with the level of the independent variables.
"""

# Training & Testing Linear Regression Model

# Scaling the Data
from sklearn.preprocessing import StandardScaler

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler on the training data and transform it
X_train_scaled = scaler.fit_transform(X_train)

# Apply the same transformation to the test data
X_test_scaled = scaler.transform(X_test)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import sqrt

# Create and fit the model
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

# Make predictions on the scaled test data
y_pred = lr.predict(X_test_scaled)

# Calculate MSE and RMSE
mse = mean_squared_error(y_test, y_pred)
rmse = sqrt(mse)

# Output the performance metrics
print(f'MSE on Test Set: {mse}')
print(f'RMSE on Test Set: {rmse}')

print(y_pred)

print(y_test)
