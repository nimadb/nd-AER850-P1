#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 10:15:43 2023

@author: nima.db
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pandas.plotting import scatter_matrix
plt.close('all')

# Step 1 - Import data from csv file into a data frame
print("Importing csv data file as Data Frame.")
df = pd.read_csv('Project 1 Data.csv')
print("Imported.")

print("\nData:", df, "\n")

print("Data Frame info:")
df.info()

print("\nChecking for null values in rows/columns.")
null_col = df.isna().any(axis=0).sum()
null_row = df.isna().any(axis=1).sum()

if null_col > 0 or null_row > 0:
    print("Number of columns with NA values:\n", null_col)  #how many columns have missing values?
    print("Number of rows with NA values:\n", null_row) #how many rows have missing values?
    df = df.dropna()
    df = df.reset_index(drop=True)
else:
    print("No null rows/columns detected.")


# Step 2 - Visualize data
# pd.plotting.scatter_matrix(df)
# plt.figure()
# sns.set_theme()
# sns.pairplot(df)

# Group the data by the "step" column
grouped = df.groupby('Step')

# Iterate through the groups and create scatter matrix plots
for step, group in grouped:
    # Select the columns you want to include in the scatter matrix
    columns_to_plot = ['X', 'Y', 'Z']
    
    # Create a scatter matrix plot for the current group
    scatter_matrix(group[columns_to_plot], diagonal='hist')
    
    # Add titles and labels
    plt.suptitle(f'Scatter Matrix Plot - Step {step}')
    plt.show()

for step, group in grouped:
    # Select the columns you want to include in the correlation calculation
    columns_to_correlate = ['X', 'Y', 'Z']
    
    # Calculate Pearson correlation within the current group
    correlations = group[columns_to_correlate].corr(method='pearson')
    
    # Print the correlation matrix for the current group
    print(f'Pearson Correlation Matrix - Step {step}:\n{correlations}\n')

# plt.figure()
# sns.heatmap(np.abs(corr_matrix))