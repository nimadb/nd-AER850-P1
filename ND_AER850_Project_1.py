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

plt.close('all')        # Close all figures
def print_asterisk_line():
    print('*' * 60)
print_asterisk_line()

# %% - Step 1 - Data Import
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

# # %% - Step 2 - Visualization and Data Analysis
# print("Visualization and analysis of entire data set")
# # Scatter mattrix of whole data 
# fig = plt.figure()
# sns.pairplot(df) 

# # Create a 3D plot for the entire dataset
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# # Extract X, Y, and Z coordinates for the entire dataset
# x = df['X']
# y = df['Y']
# z = df['Z']
# step = df['Step']

# # Create a dictionary to map 'Step' values to specific colors
# step_colors = {
#     1: 'red',
#     2: 'blue',
#     3: 'green',
#     4: 'orange',
#     5: 'purple',
#     6: 'cyan',
#     7: 'magenta',
#     8: 'yellow',
#     9: 'black',
#     10: 'gray',
#     11: 'pink',
#     12: 'brown',
#     13: 'lime',
# }
# # Get colors for each data point based on the 'Step' values
# colors = [step_colors[step_val] for step_val in step]
# # Create the 3D scatter plot with individual colors
# scatter = ax.scatter(x, y, z, c=colors, marker='o', label='Entire Dataset')
# # Set labels and title
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# plt.title('Color-Coded 3D Plot of the Entire Dataset')
# # Create a custom legend to display colors for each step
# custom_legend = [plt.Line2D([0], [0], marker='o', color='w', label=f'Step {step_val}', markerfacecolor=step_colors[step_val]) for step_val in sorted(step_colors.keys())]
# ax.legend(handles=custom_legend, bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.show()

# # Descriptive statistics
# descriptive_stats = df.describe()
# print("Descriptive Statistics:\n", descriptive_stats)

# print_asterisk_line()
# print("Visualization and analysis of data set grouped by 'Step'")
# # Group the data by the "Step" column
# grouped_data = df.groupby('Step')

# for step, group in grouped_data:
#     print(f"\nStep {step}:")
    
#     # You can perform separate statistical analysis or visualizations for each step here
#     # For example, to get descriptive statistics for each step
#     descriptive_stats = group.describe()
#     print("Descriptive Statistics:")
#     print(descriptive_stats)

#     # Extract X, Y, and Z coordinates for the current step
#     x = group['X']
#     y = group['Y']
#     z = group['Z']
    
#     # Create a 3D plot for the current step's data
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
    
#     # Create the 3D scatter plot for the current step, color-coded
#     scatter = ax.scatter(x, y, z, c=step_colors[step], marker='o', label=f'Step {step}')
    
#     # Set labels and title for the 3D plot
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     plt.title(f'3D Plot for Step {step}')
#     plt.legend()
#     plt.show()
    
#     # Create pair plots for the current step's data (not color-coded)
#     sns.set(style="whitegrid")
#     sns.pairplot(group[['X', 'Y', 'Z']], diag_kind="kde")
#     plt.title(f'Pairplot for Step {step}')
#     plt.show()
# print_asterisk_line() 
# # %% - Step 3 - Correlation Analysis
# # Create lists to store the correlations
# correlations_x = []
# correlations_y = []
# correlations_z = []
# # Iterate through each group (each step)
# for step, group in grouped_data:
#     # Calculate the correlation between 'X' and 'Step'
#     correlation_x, _ = stats.pearsonr(group['X'], group['Step'])
#     correlations_x.append(correlation_x)
    
#     # Calculate the correlation between 'Y' and 'Step'
#     correlation_y, _ = stats.pearsonr(group['Y'], group['Step'])
#     correlations_y.append(correlation_y)
    
#     # Calculate the correlation between 'Z' and 'Step'
#     correlation_z, _ = stats.pearsonr(group['Z'], group['Step'])
#     correlations_z.append(correlation_z)

#     print(f"Step {step}:")
#     print(f"Correlation X and Step: {correlation_x:.2f}")
#     print(f"Correlation Y and Step: {correlation_y:.2f}")
#     print(f"Correlation Z and Step: {correlation_z:.2f}")
#     print("\n")

# # If needed, you can also access the correlations for all steps
# print("Correlations for X:", correlations_x)
# print("Correlations for Y:", correlations_y)
# print("Correlations for Z:", correlations_z)
# print("This makes sense that ther is no correlation with grouped data.")

# correlation_matrix = df[['X', 'Y', 'Z', 'Step']].corr(method='pearson')
# print(correlation_matrix)

# print_asterisk_line() 
# print("New grouping Strategy")
# # Create a new column 'StepGroup' based on your grouping criteria
# grouping_criteria = {
#     1: 'Group 1-6',
#     2: 'Group 1-6',
#     3: 'Group 1-6',
#     4: 'Group 1-6',
#     5: 'Group 1-6',
#     6: 'Group 1-6',
#     7: 'Group 7',
#     8: 'Group 8',
#     9: 'Group 9',
#     10: 'Group 10-13',
#     11: 'Group 10-13',
#     12: 'Group 10-13',
#     13: 'Group 10-13',
# }

# df['StepGroup'] = df['Step'].map(grouping_criteria)

# # Verify the new 'StepGroup' column
# print(df[['Step', 'StepGroup']])
# new_grouped_data = df.groupby('StepGroup')
# new_correlations_x = []
# new_correlations_y = []
# new_correlations_z = []
# for step, group in new_grouped_data:
#     # Calculate the correlation between 'X' and 'Step'
#     new_correlation_x, _ = stats.pearsonr(group['X'], group['Step'])
#     new_correlations_x.append(new_correlation_x)
    
#     # Calculate the correlation between 'Y' and 'Step'
#     new_correlation_y, _ = stats.pearsonr(group['Y'], group['Step'])
#     new_correlations_y.append(new_correlation_y)
    
#     # Calculate the correlation between 'Z' and 'Step'
#     new_correlation_z, _ = stats.pearsonr(group['Z'], group['Step'])
#     new_correlations_z.append(new_correlation_z)

#     print(f"Step {step}:")
#     print(f"Correlation X and Step: {new_correlation_x:.2f}")
#     print(f"Correlation Y and Step: {new_correlation_y:.2f}")
#     print(f"Correlation Z and Step: {new_correlation_z:.2f}")
#     print("\n")
print_asterisk_line() 

# %% Step 4 - Classification and Model Development/Engineering
#stratified sampling
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=20231005)
for train_index, test_index in split.split(df, df["Step"]):
    strat_train_set = df.loc[train_index].reset_index(drop=True)
    strat_test_set = df.loc[test_index].reset_index(drop=True)

train_data = strat_train_set[['X', 'Y', 'Z', 'Step']]
print(train_data.info())
print(train_data.describe())
print(train_data)

train_step_counts = train_data['Step'].value_counts().reset_index()
train_step_counts.columns = ['Step', 'Count']
train_step_counts = train_step_counts.sort_values(by='Step')
print(train_step_counts)


