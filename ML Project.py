#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# In[2]:


# Read data
data = pd.read_csv("Cleaned Global Eco Inc.csv")
data


# In[3]:


data.describe()


# In[4]:


# Check for missing values
print(data.isnull().sum())


# In[5]:


data.dtypes


# In[6]:


# Change categorical variables to "category"
data['Country'] = data['Country'].astype('category')


# In[7]:


data.dtypes


# In[8]:


# Create dummy variables
data = pd.get_dummies(data, columns=['Country'], drop_first=True)
data


# In[9]:


data = data.drop('Unnamed: 0', axis=1)


# In[10]:


data = data[np.isfinite(data).all(axis=1)]
data.describe()


# In[11]:


numeric_cols = data.select_dtypes(include=[np.number]).columns
numeric_data = data[numeric_cols]

# Calculate the correlation matrix for the numeric data
correlation_matrix = numeric_data.corr()

# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# Displaying correlation pairs that are highly correlated (above 0.90)
high_corr_pairs = correlation_matrix.abs().unstack().sort_values(kind="quicksort", ascending=False)
high_corr_pairs = high_corr_pairs[high_corr_pairs >= 0.9]
high_corr_pairs = high_corr_pairs[high_corr_pairs < 1]  # remove self-correlation
print(high_corr_pairs)


# Simple Linear Regression

# In[12]:


# Define the features and target variable
X = data.drop('GDP Growth Rate (%)', axis=1)
y = data['GDP Growth Rate (%)']

# Splitting the dataset into training and testing sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Display the sizes of the train and test sets
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[13]:


import statsmodels.api as sm
# Prepare the data
X = data['Gross National Income(GNI) in USD']  # Predictor
y = data['GDP Growth Rate (%)']  # Target variable
X = sm.add_constant(X)  # Adding a constant for the intercept

# Fit the OLS model
model = sm.OLS(y, X).fit()

# Print out the statistics
model_summary = model.summary()
print(model_summary)


# In[14]:


from sklearn.linear_model import LinearRegression

X_train_GDP = X_train[['Gross National Income(GNI) in USD']]
X_test_GDP = X_test[['Gross National Income(GNI) in USD']]

# Creating and training the linear regression model
linear_model = LinearRegression()
linear_model.fit(X_train_GDP, y_train)

# Predicting the quality on the training set
y_train_pred = linear_model.predict(X_train_GDP)

# Computing the R-squared value on the training set
r_squared_train = linear_model.score(X_train_GDP, y_train)


plt.figure(figsize=(8, 6))
plt.scatter(X_train_GDP, y_train, color='blue', label='Actual GDP Growth Rate', alpha=0.5)
plt.plot(X_train_GDP, y_train_pred, color='red', linewidth=2, label='Fitted Line')
plt.xlabel('Gross National Income(GNI) in USD')
plt.ylabel('GDP Growth Rate (%)')
plt.title('Relationship between Gross National Income(GNI) in USD and GDP Growth Rate (%) with Regression Line')
plt.legend()
plt.show()

r_squared_train


# In[15]:


from sklearn.metrics import mean_absolute_error, mean_squared_error

y_test_pred = linear_model.predict(X_test_GDP)

# Calculating MAE and RMSE
mae = mean_absolute_error(y_test, y_test_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

print("MAE:", mae, "RMSE:", rmse)


# Multiple Linear Regression

# In[16]:


features = ['Agriculture, hunting, forestry, fishing (ISIC A-B)', 'Construction (ISIC F)', 'Exports of goods and services', 'Imports of goods and services','Manufacturing (ISIC D)','Transport, storage and communication (ISIC I)','Wholesale, retail trade, restaurants and hotels (ISIC G-H)']  # Multiple predictors
X = data[features]  # Predictor matrix
y = data['GDP Growth Rate (%)']  # Target variable
X = sm.add_constant(X)  # Adding a constant for the intercept

# Fit the OLS model
model = sm.OLS(y, X).fit()

# Print out the statistics
model_summary = model.summary()
print(model_summary)


# In[17]:


# Selecting multiple features for the predictors
features = ['Agriculture, hunting, forestry, fishing (ISIC A-B)', 'Construction (ISIC F)', 'Exports of goods and services', 'Imports of goods and services','Manufacturing (ISIC D)','Transport, storage and communication (ISIC I)','Wholesale, retail trade, restaurants and hotels (ISIC G-H)']  # Multiple predictors
X_train_multi = X_train[features]
X_test_multi = X_test[features]

# Creating and training the multiple linear regression model
multi_linear_model = LinearRegression()
multi_linear_model.fit(X_train_multi, y_train)

# Predicting the quality on the training set
y_train_pred_multi = multi_linear_model.predict(X_train_multi)

# Computing the R-squared value on the training set
r_squared_train_multi = multi_linear_model.score(X_train_multi, y_train)

# Getting the coefficients of the model
coefficients = pd.DataFrame(multi_linear_model.coef_, index=features, columns=['Coefficient'])

# Displaying the coefficients and the R-squared value
coefficients, r_squared_train_multi


# In[18]:


y_test_pred_multi = multi_linear_model.predict(X_test_multi)

# Calculating MAE and RMSE for the test set predictions
mae_multi = mean_absolute_error(y_test, y_test_pred_multi)
rmse_multi = np.sqrt(mean_squared_error(y_test, y_test_pred_multi))

print("MAE:", mae_multi, "RMSE:", rmse_multi)


# In[19]:


# Visualizing the relationship between Predicted Quality and Actual Quality
plt.figure(figsize=(8, 6))
plt.scatter(y_train, y_train_pred_multi, color='blue', label='Predicted GDP Growth Rate', alpha=0.5)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=2, label='Ideal Fit')
plt.xlabel('Actual GDP Growth Rate')
plt.ylabel('Predicted GDP Growth Rate')
plt.title('Multiple Linear Regression: Actual vs. Predicted GDP Growth Rate')
plt.legend()
plt.show()


# In[23]:


features = ['Imports of goods and services','Transport, storage and communication (ISIC I)']  # Multiple predictors
X = data[features]  # Predictor matrix
y = data['GDP Growth Rate (%)']  # Target variable
X = sm.add_constant(X)  # Adding a constant for the intercept

# Fit the OLS model
model = sm.OLS(y, X).fit()

# Print out the statistics
model_summary = model.summary()
print(model_summary)


# In[25]:


# Selecting multiple features for the predictors
features = ['Imports of goods and services','Transport, storage and communication (ISIC I)']  # Multiple predictors
X_train_multi = X_train[features]
X_test_multi = X_test[features]

# Creating and training the multiple linear regression model
multi_linear_model = LinearRegression()
multi_linear_model.fit(X_train_multi, y_train)

# Predicting the quality on the training set
y_train_pred_multi = multi_linear_model.predict(X_train_multi)

# Computing the R-squared value on the training set
r_squared_train_multi = multi_linear_model.score(X_train_multi, y_train)

# Getting the coefficients of the model
coefficients = pd.DataFrame(multi_linear_model.coef_, index=features, columns=['Coefficient'])

# Displaying the coefficients and the R-squared value
coefficients, r_squared_train_multi


# In[26]:


y_test_pred_multi = multi_linear_model.predict(X_test_multi)

# Calculating MAE and RMSE for the test set predictions
mae_multi = mean_absolute_error(y_test, y_test_pred_multi)
rmse_multi = np.sqrt(mean_squared_error(y_test, y_test_pred_multi))

print("MAE:", mae_multi, "RMSE:", rmse_multi)


# In[27]:


# Visualizing the relationship between Predicted Quality and Actual Quality
plt.figure(figsize=(8, 6))
plt.scatter(y_train, y_train_pred_multi, color='blue', label='Predicted GDP Growth Rate', alpha=0.5)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=2, label='Ideal Fit')
plt.xlabel('Actual GDP Growth Rate')
plt.ylabel('Predicted GDP Growth Rate')
plt.title('Multiple Linear Regression: Actual vs. Predicted GDP Growth Rate')
plt.legend()
plt.show()


# In[20]:


from sklearn.tree import DecisionTreeRegressor, plot_tree

# Creating the regression tree model
tree_model = DecisionTreeRegressor(max_depth=13, random_state=1)
tree_model.fit(X_train, y_train)


# In[21]:


from sklearn.metrics import mean_squared_error, mean_absolute_error

# Making predictions on the test set
y_test_pred = tree_model.predict(X_test)

# Calculating MAE and RMSE
mae = mean_absolute_error(y_test, y_test_pred)
rmse = mean_squared_error(y_test, y_test_pred, squared=False)

# Output the evaluation metrics
mae, rmse


# In[59]:


# Experimenting with different depths
depths = [3, 4, 5, 6, 7, 9, 11, 13, 14]
results = []

for depth in depths:
    # Create and fit the tree model
    model = DecisionTreeRegressor(max_depth=depth, random_state=1)
    model.fit(X_train, y_train)

    # Predicting on the test set
    y_pred = model.predict(X_test)

    # Calculating metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    # Storing results
    results.append((depth, mae, rmse))

# Display results as a DataFrame for easy comparison
results_df = pd.DataFrame(results, columns=['Max Depth', 'MAE', 'RMSE'])
results_df


# In[31]:


from sklearn.svm import SVR

# Creating and training the SVR model with default parameters
svr_model = SVR()
svr_model.fit(X_train, y_train)

# Making predictions on the test set
y_test_pred_svr = svr_model.predict(X_test)

# Calculating MAE and RMSE for the SVR model
mae_svr = mean_absolute_error(y_test, y_test_pred_svr)
rmse_svr = mean_squared_error(y_test, y_test_pred_svr, squared=False)

mae_svr, rmse_svr


# In[32]:


# Visualization: Plot of predicted vs actual GDP Growth Rate
plt.figure(figsize=(10, 8))
plt.scatter(y_test, y_test_pred_svr, alpha=0.5, color='blue', label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', linewidth=2, label='Ideal Prediction Line')
plt.xlabel('Actual GDP Growth Rate')
plt.ylabel('Predicted GDP Growth Rate')
plt.title('Predicted vs Actual GDP Growth Rate (SVR Model)')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:





# In[22]:


from sklearn.neural_network import MLPRegressor

# Creating and training the MLP model with a basic architecture
mlp_model = MLPRegressor(hidden_layer_sizes=(10,), max_iter=100, random_state=1)
mlp_model.fit(X_train, y_train)

# Making predictions on the test set
y_test_pred_mlp = mlp_model.predict(X_test)

# Calculating MAE and RMSE for the MLP model
mae_mlp = mean_absolute_error(y_test, y_test_pred_mlp)
rmse_mlp = mean_squared_error(y_test, y_test_pred_mlp, squared=False)

mae_mlp, rmse_mlp


# In[33]:


plt.figure(figsize=(10, 8))
plt.scatter(y_test, y_test_pred_mlp, alpha=0.5, color='blue', label='Predicted vs Actual (MLP)')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', linewidth=2, label='Ideal Prediction Line')
plt.xlabel('Actual GDP Growth Rate')
plt.ylabel('Predicted GDP Growth Rate')
plt.title('Predicted vs Actual GDP Growth Rate (MLP Model)')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:




