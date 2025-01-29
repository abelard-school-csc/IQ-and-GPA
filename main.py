import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import ttest_ind, linregress

data = pd.read_csv("data.csv")

sns.histplot(data['iq'], kde=True, bins=20)
plt.savefig('images/iq_distribution.png')
plt.close()

sns.histplot(data['gpa'], kde=True, bins=20)
plt.savefig('images/gpa_distribution.png')
plt.close()

correlations = data[['gpa', 'iq']].corr()
sns.heatmap(correlations, annot=True, cmap='coolwarm')
plt.savefig('images/correlation_heatmap.png')
plt.close()

slope, intercept, r_value, p_value, std_err = linregress(data['iq'], data['gpa'])
plt.scatter(data['iq'], data['gpa'], label='Data')
plt.plot(data['iq'], slope * data['iq'] + intercept, color='red', label='Linear Fit')
plt.legend()
plt.savefig('images/iq_gpa_regression.png')
plt.close()

t_stat, p_val = ttest_ind(data['iq'], data['gpa'])
