#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 10:15:43 2023

@author: nima.db
"""
# %% - Step 0 - Configuration
import os
os.system('clear')      # Clear console
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from pandas.plotting import scatter_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.cm import get_cmap
import copy

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier  # For classification
from sklearn.ensemble import RandomForestClassifier  # For classification
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from warnings import filterwarnings
filterwarnings('ignore')

plt.close('all')        # Close all figures
def print_asterisk_line():
    print('*' * 60)
print_asterisk_line()

# %% - Step 1a) - Data Import
print("Importing csv data file as Data Frame")   # Status message
df = pd.read_csv('Project 1 Data.csv')                      # Read data
print("Imported")                                 # Status message

print("\nData:", df, "\n")                        # Print data
print("Data Frame info:\n", df.info())            # Print data info 

print("\nChecking for null values in rows/columns")           # Status message
null_col = df.isna().any(axis=0).sum()                      # Check for columns with empty values
null_row = df.isna().any(axis=1).sum()                      # Check for rows with empty values
if null_col > 0 or null_row > 0:                            # If any emptry values
    print("Number of columns with NA values:\n", null_col)    # Print how many columns with empty values
    print("Number of rows with NA values:\n", null_row)       # Print how many rows with empty values
    df = df.dropna()                                        # Drop empty rows
    df = df.reset_index(drop=True)                          # Reset the row indices
else:
    print("No null rows/columns detected")                    # Status message if no emptry values

df_counts = df['Step'].value_counts().reset_index()
df_counts.columns = ['Step', 'Count']
df_counts = df_counts.sort_values(by='Step')
print(df_counts)
print_asterisk_line()

# %% - Step 1b) - Data Splitting
from sklearn.model_selection import train_test_split
# Load your dataset (Assuming df contains your data)
X = df[['X', 'Y', 'Z']]
y = df['Step']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train = X_train.copy()
train['Step'] = y_train
train = train.reset_index(drop=True)

test = X_test.copy()
test['Step'] = y_test
test = test.reset_index(drop=True)

print_asterisk_line()

# %% - Step 2 - Visualization and Data Analysis
print("Visualization and analysis of entire data set")
# Scatter mattrix of whole data 
fig = plt.figure()
sns.pairplot(train) 

# Create a 3D plot for the entire dataset
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Extract X, Y, and Z coordinates for the entire dataset
x = train['X']
y = train['Y']
z = train['Z']
step = train['Step']

# Create a dictionary to map 'Step' values to specific colors
step_colors = {
    1: 'red',
    2: 'blue',
    3: 'green',
    4: 'orange',
    5: 'purple',
    6: 'cyan',
    7: 'magenta',
    8: 'yellow',
    9: 'black',
    10: 'gray',
    11: 'pink',
    12: 'brown',
    13: 'lime',
}
# Get colors for each data point based on the 'Step' values
colors = [step_colors[step_val] for step_val in step]
# Create the 3D scatter plot with individual colors
scatter = ax.scatter(x, y, z, c=colors, marker='o', label='Entire Dataset')
# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('Color-Coded 3D Plot of the Entire Dataset')
# Create a custom legend to display colors for each step
custom_legend = [plt.Line2D([0], [0], marker='o', color='w', label=f'Step {step_val}', markerfacecolor=step_colors[step_val]) for step_val in sorted(step_colors.keys())]
ax.legend(handles=custom_legend, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# Descriptive statistics
descriptive_stats = train.describe()
print("Descriptive Statistics:\n", descriptive_stats)

print_asterisk_line()
print("Visualization and analysis of data set grouped by 'Step'")
# Group the data by the "Step" column
grouped_data = train.groupby('Step')

for step, group in grouped_data:
    print(f"\nStep {step}:")
    
    # You can perform separate statistical analysis or visualizations for each step here
    # For example, to get descriptive statistics for each step
    descriptive_stats = group.describe()
    print("Descriptive Statistics:")
    print(descriptive_stats)

    # Extract X, Y, and Z coordinates for the current step
    x = group['X']
    y = group['Y']
    z = group['Z']
    
    # Create a 3D plot for the current step's data
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Create the 3D scatter plot for the current step, color-coded
    scatter = ax.scatter(x, y, z, c=step_colors[step], marker='o', label=f'Step {step}')
    
    # Set labels and title for the 3D plot
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title(f'3D Plot for Step {step}')
    plt.legend()
    plt.show()
    
    # Create pair plots for the current step's data (not color-coded)
    sns.set(style="whitegrid")
    sns.pairplot(group[['X', 'Y', 'Z']], diag_kind="kde")
    plt.title(f'Pairplot for Step {step}')
    plt.show()
print_asterisk_line() 

# %% - Step 3 - Correlation Analysis
# Create lists to store the correlations
correlations_x = []
correlations_y = []
correlations_z = []
# Iterate through each group (each step)
for step, group in grouped_data:
    # Calculate the correlation between 'X' and 'Step'
    correlation_x, _ = stats.pearsonr(group['X'], group['Step'])
    correlations_x.append(correlation_x)
    
    # Calculate the correlation between 'Y' and 'Step'
    correlation_y, _ = stats.pearsonr(group['Y'], group['Step'])
    correlations_y.append(correlation_y)
    
    # Calculate the correlation between 'Z' and 'Step'
    correlation_z, _ = stats.pearsonr(group['Z'], group['Step'])
    correlations_z.append(correlation_z)

    print(f"Step {step}:")
    print(f"Correlation X and Step: {correlation_x:.2f}")
    print(f"Correlation Y and Step: {correlation_y:.2f}")
    print(f"Correlation Z and Step: {correlation_z:.2f}")
    print("\n")

# If needed, you can also access the correlations for all steps
print("Correlations for X:", correlations_x)
print("Correlations for Y:", correlations_y)
print("Correlations for Z:", correlations_z)
print("This makes sense that ther is no correlation with grouped data.")

correlation_matrix = df[['X', 'Y', 'Z', 'Step']].corr(method='pearson')
print(correlation_matrix)

print_asterisk_line() 
print("New grouping Strategy")
# Create a new column 'StepGroup' based on your grouping criteria
grouping_criteria = {
    1: 'Group 1-6',
    2: 'Group 1-6',
    3: 'Group 1-6',
    4: 'Group 1-6',
    5: 'Group 1-6',
    6: 'Group 1-6',
    7: 'Group 7',
    8: 'Group 8',
    9: 'Group 9',
    10: 'Group 10-13',
    11: 'Group 10-13',
    12: 'Group 10-13',
    13: 'Group 10-13',
}

train['StepGroup'] = train['Step'].map(grouping_criteria)

# Verify the new 'StepGroup' column
print(train[['Step', 'StepGroup']])
new_grouped_data = train.groupby('StepGroup')
new_correlations_x = []
new_correlations_y = []
new_correlations_z = []
for step, group in new_grouped_data:
    # Calculate the correlation between 'X' and 'Step'
    new_correlation_x, _ = stats.pearsonr(group['X'], group['Step'])
    new_correlations_x.append(new_correlation_x)
    
    # Calculate the correlation between 'Y' and 'Step'
    new_correlation_y, _ = stats.pearsonr(group['Y'], group['Step'])
    new_correlations_y.append(new_correlation_y)
    
    # Calculate the correlation between 'Z' and 'Step'
    new_correlation_z, _ = stats.pearsonr(group['Z'], group['Step'])
    new_correlations_z.append(new_correlation_z)

    print(f"Step {step}:")
    print(f"Correlation X and Step: {new_correlation_x:.2f}")
    print(f"Correlation Y and Step: {new_correlation_y:.2f}")
    print(f"Correlation Z and Step: {new_correlation_z:.2f}")
    print("\n")
print_asterisk_line() 

# %% Step 4/5 - Classification and Model Development/Engineering
# %% - Step 4/5a) - LogisticRegression Model
lr_param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga']
}

# Create the GridSearchCV object
lr_grid_search = GridSearchCV(LogisticRegression(max_iter=1000), lr_param_grid, cv=5, scoring='accuracy')
# Train the model on the training data
lr_grid_search.fit(X_train, y_train)
# Get the best hyperparameters
lr_best_params = lr_grid_search.best_params_
print("Best Hyperparameters for Logistic Regression:", lr_best_params)

# Train a new Logistic Regression model with the best hyperparameters
lr_best_model = LogisticRegression(**lr_best_params, max_iter=1000)
lr_best_model.fit(X_train, y_train)

# Use the trained model to make predictions on the test data
lr_y_pred = lr_best_model.predict(X_test)
lr_results = X_test.copy()
lr_results['Step'] = lr_y_pred
lr_results = lr_results.reset_index(drop=True)

# LR Model Evaluation
# Calculate accuracy
lr_accuracy = accuracy_score(y_test, lr_y_pred)
print(f'lr_Accuracy: {lr_accuracy:.2f}')
# Generate a classification report
lr_report = classification_report(y_test, lr_y_pred)
print(lr_report)
# Create a confusion matrix
lr_cm = confusion_matrix(y_test, lr_y_pred)
print('Confusion Matrix:')
print(lr_cm)

lr_results['Actual_Step'] = test['Step']
lr_results['Difference'] = lr_results['Step'] - test['Step']
print(lr_results)
lr_non_zero_rows = lr_results[lr_results['Difference'] != 0]
print(lr_non_zero_rows)
lr_num_non_zero_rows = lr_non_zero_rows.shape[0]
lr_num_rows_lr_results = lr_results.shape[0]
print("Total number of rows with non-zero values:", lr_num_non_zero_rows, "out of", lr_num_rows_lr_results)
print_asterisk_line()

# %% - Step 4/5b) - Decision Tree Model
# Define the hyperparameter grid
dt_param_grid = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create the GridSearchCV object
dt_grid_search = GridSearchCV(DecisionTreeClassifier(), dt_param_grid, cv=5, scoring='accuracy')

# Fit the model to find the best hyperparameters
dt_grid_search.fit(X_train, y_train)

# Get the best hyperparameters
dt_best_params = dt_grid_search.best_params_
print("Best Hyperparameters for Decision Tree:", dt_best_params)

# Train a new Decision Tree model with the best hyperparameters
dt_best_model = DecisionTreeClassifier(**dt_best_params)
dt_best_model.fit(X_train, y_train)

dt_y_pred = dt_best_model.predict(X_test)

dt_results = X_test.copy()
dt_results['Step'] = dt_y_pred
dt_results = dt_results.reset_index(drop=True)

# clf Model Evaluation
dt_accuracy = accuracy_score(y_test, dt_y_pred)
print(f"dt_Accuracy: {dt_accuracy}")
# Generate the classification report
dt_report = classification_report(y_test, dt_y_pred)
print("Classification Report:\n", dt_report)
# Generate the confusion matrix
dt_cm = confusion_matrix(y_test, dt_y_pred)
print("Confusion Matrix:\n", dt_cm)

dt_results['Actual_Step'] = test['Step']
dt_results['Difference'] = dt_results['Step'] - test['Step']
print(dt_results)
dt_non_zero_rows = dt_results[dt_results['Difference'] != 0]
print(dt_non_zero_rows)
dt_num_non_zero_rows = dt_non_zero_rows.shape[0]
dt_num_rows_dt_results = dt_results.shape[0]
print("Total number of rows with non-zero values:", dt_num_non_zero_rows, "out of", dt_num_rows_dt_results)
print_asterisk_line()

# %% - Step 4/5c) - Random Forest Model
# Define the hyperparameter grid
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Create the GridSearchCV object
rf_grid_search = GridSearchCV(RandomForestClassifier(), rf_param_grid, cv=5, scoring='accuracy')

# Fit the model to find the best hyperparameters
rf_grid_search.fit(X_train, y_train)

# Get the best hyperparameters
rf_best_params = rf_grid_search.best_params_
print("Best Hyperparameters for Random Forest:", rf_best_params)

# Train a new Random Forest model with the best hyperparameters
rf_best_model = RandomForestClassifier(**rf_best_params)
rf_best_model.fit(X_train, y_train)

rf_y_pred = rf_best_model.predict(X_test)

rf_results = X_test.copy()
rf_results['Step'] = rf_y_pred
rf_results = rf_results.reset_index(drop=True)

# rf Model Evaluation
rf_accuracy = accuracy_score(y_test, rf_y_pred)
print(f"rf_Accuracy: {rf_accuracy}")
# Generate the classification report
rf_report = classification_report(y_test, rf_y_pred)
print("Classification Report:\n", rf_report)
# Generate the confusion matrix
rf_cm = confusion_matrix(y_test, rf_y_pred)
print("Confusion Matrix:\n", rf_cm)

rf_results['Actual_Step'] = test['Step']
rf_results['Difference'] = rf_results['Step'] - test['Step']
print(rf_results)
rf_non_zero_rows = rf_results[rf_results['Difference'] != 0]
print(rf_non_zero_rows)
rf_num_non_zero_rows = rf_non_zero_rows.shape[0]
rf_num_rows_rf_results = rf_results.shape[0]
print("Total number of rows with non-zero values:", rf_num_non_zero_rows, "out of", rf_num_rows_rf_results)
print_asterisk_line()

# %% - Step 6 - Model Evaluation
import joblib
# # Save the Logistic Regression model
# joblib.dump(lr_best_model, 'logistic_regression_model.joblib')

# # Save the Decision Tree model
# joblib.dump(dt_best_model, 'decision_tree_model.joblib')

# # Save the Random Forest model
# joblib.dump(rf_best_model, 'random_forest_model.joblib')

# Load the Logistic Regression model
lr_model = joblib.load('logistic_regression_model.joblib')

# Load the Decision Tree model
dt_model = joblib.load('decision_tree_model.joblib')

# Load the Random Forest model
rf_model = joblib.load('random_forest_model.joblib')

data = [[9.375,3.0625,1.51], [6.995,5.125,0.3875], [0,3.0625,1.93], [9.4,3,1.8], [9.4,3,1.3]]
new_data_column_names = ['X', 'Y', 'Z']
new_data = pd.DataFrame(data, columns=new_data_column_names)

lr_predictions = lr_model.predict(new_data)
dt_predictions = dt_model.predict(new_data)
rf_predictions = rf_model.predict(new_data)
print('Logistic Regression Model Step Prediction:', lr_predictions)
print('Decision Tree Model Step Prediction:', dt_predictions)
print('Random Forest Model Step Prediction:', rf_predictions)