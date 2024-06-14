#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# In[33]:


# Read data
data = pd.read_csv("Cleaned Global Eco Inc.csv")
data


# In[34]:


data.describe()


# In[35]:


# Check for missing values
print(data.isnull().sum())


# In[36]:


data.dtypes


# In[37]:


# Change categorical variables to "category"
data['Country'] = data['Country'].astype('category')


# In[38]:


data.dtypes


# In[39]:


# Create dummy variables
data = pd.get_dummies(data, columns=['Country'], drop_first=True)
data


# In[40]:


data = data.drop('Unnamed: 0', axis=1)
data.dtypes


# In[41]:


data = data[np.isfinite(data).all(axis=1)]
data


# In[42]:


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

# In[43]:


# Define the features and target variable
X = data.drop('GDP Growth Rate (%)', axis=1)
y = data['GDP Growth Rate (%)']

# Splitting the dataset into training and testing sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Display the sizes of the train and test sets
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[14]:


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


# In[15]:


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


# In[16]:


from sklearn.metrics import mean_absolute_error, mean_squared_error

y_test_pred = linear_model.predict(X_test_GDP)

# Calculating MAE and RMSE
mae = mean_absolute_error(y_test, y_test_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

print("MAE:", mae, "RMSE:", rmse)


# Multiple Linear Regression

# In[17]:


features = ['Agriculture, hunting, forestry, fishing (ISIC A-B)', 'Construction (ISIC F)', 'Exports of goods and services', 'Imports of goods and services','Manufacturing (ISIC D)','Transport, storage and communication (ISIC I)','Wholesale, retail trade, restaurants and hotels (ISIC G-H)']  # Multiple predictors
X = data[features]  # Predictor matrix
y = data['GDP Growth Rate (%)']  # Target variable
X = sm.add_constant(X)  # Adding a constant for the intercept

# Fit the OLS model
model = sm.OLS(y, X).fit()

# Print out the statistics
model_summary = model.summary()
print(model_summary)


# In[18]:


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


# In[19]:


y_test_pred_multi = multi_linear_model.predict(X_test_multi)

# Calculating MAE and RMSE for the test set predictions
mae_multi = mean_absolute_error(y_test, y_test_pred_multi)
rmse_multi = np.sqrt(mean_squared_error(y_test, y_test_pred_multi))

print("MAE:", mae_multi, "RMSE:", rmse_multi)


# In[20]:


# Visualizing the relationship between Predicted Quality and Actual Quality
plt.figure(figsize=(8, 6))
plt.scatter(y_train, y_train_pred_multi, color='blue', label='Predicted GDP Growth Rate', alpha=0.5)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=2, label='Ideal Fit')
plt.xlabel('Actual GDP Growth Rate')
plt.ylabel('Predicted GDP Growth Rate')
plt.title('Multiple Linear Regression: Actual vs. Predicted GDP Growth Rate')
plt.legend()
plt.show()


# In[21]:


features = ['Imports of goods and services','Transport, storage and communication (ISIC I)']  # Multiple predictors
X = data[features]  # Predictor matrix
y = data['GDP Growth Rate (%)']  # Target variable
X = sm.add_constant(X)  # Adding a constant for the intercept

# Fit the OLS model
model = sm.OLS(y, X).fit()

# Print out the statistics
model_summary = model.summary()
print(model_summary)


# In[22]:


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


# In[23]:


y_test_pred_multi = multi_linear_model.predict(X_test_multi)

# Calculating MAE and RMSE for the test set predictions
mae_multi = mean_absolute_error(y_test, y_test_pred_multi)
rmse_multi = np.sqrt(mean_squared_error(y_test, y_test_pred_multi))

print("MAE:", mae_multi, "RMSE:", rmse_multi)


# In[24]:


# Visualizing the relationship between Predicted Quality and Actual Quality
plt.figure(figsize=(8, 6))
plt.scatter(y_train, y_train_pred_multi, color='blue', label='Predicted GDP Growth Rate', alpha=0.5)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=2, label='Ideal Fit')
plt.xlabel('Actual GDP Growth Rate')
plt.ylabel('Predicted GDP Growth Rate')
plt.title('Multiple Linear Regression: Actual vs. Predicted GDP Growth Rate')
plt.legend()
plt.show()


# In[25]:


from sklearn.tree import DecisionTreeRegressor, plot_tree

# Creating the regression tree model
tree_model = DecisionTreeRegressor(max_depth=13, random_state=1)
tree_model.fit(X_train, y_train)


# In[26]:


from sklearn.metrics import mean_squared_error, mean_absolute_error

# Making predictions on the test set
y_test_pred = tree_model.predict(X_test)

# Calculating MAE and RMSE
mae = mean_absolute_error(y_test, y_test_pred)
rmse = mean_squared_error(y_test, y_test_pred, squared=False)

# Output the evaluation metrics
mae, rmse


# In[27]:


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


# In[28]:


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


# In[29]:


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


# In[61]:


from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
# Creating and training the MLP model with a basic architecture
mlp_model = MLPRegressor(hidden_layer_sizes=(100,), max_iter=100, random_state=1)
mlp_model.fit(X_train, y_train)

# Making predictions on the test set
y_test_pred_mlp = mlp_model.predict(X_test)

# Calculating MAE and RMSE for the MLP model
mae_mlp = mean_absolute_error(y_test, y_test_pred_mlp)
rmse_mlp = mean_squared_error(y_test, y_test_pred_mlp, squared=False)

mae_mlp, rmse_mlp


# In[62]:


plt.figure(figsize=(10, 8))
plt.scatter(y_test, y_test_pred_mlp, alpha=0.5, color='blue', label='Predicted vs Actual (MLP)')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', linewidth=2, label='Ideal Prediction Line')
plt.xlabel('Actual GDP Growth Rate')
plt.ylabel('Predicted GDP Growth Rate')
plt.title('Predicted vs Actual GDP Growth Rate (MLP Model)')
plt.legend()
plt.grid(True)
plt.show()


# In[44]:


from sklearn.ensemble import RandomForestRegressor
# Create and train the Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=1)
rf_model.fit(X_train, y_train)

# Predict the target variable on the test set
y_test_pred = rf_model.predict(X_test)

# Calculate evaluation metrics
mae_rf = mean_absolute_error(y_test, y_test_pred)
rmse_rf = mean_squared_error(y_test, y_test_pred, squared=False)
print(f"Random Forest - MAE: {mae_rf:.2f}, RMSE: {rmse_rf:.2f}")

# Feature Importance
feature_importances = rf_model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False).head(20)

# Plot Top 20 Feature Importance
plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Top 20 Feature Importance (Random Forest)')
plt.tight_layout()
plt.show()

# Visualization: Plot of Predicted vs Actual GDP Growth Rate
plt.figure(figsize=(10, 8))
plt.scatter(y_test, y_test_pred, alpha=0.5, color='blue', label='Predicted vs Actual (RF)')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', linewidth=2, label='Ideal Prediction Line')
plt.xlabel('Actual GDP Growth Rate')
plt.ylabel('Predicted GDP Growth Rate')
plt.title('Predicted vs Actual GDP Growth Rate (Random Forest Model)')
plt.legend()
plt.grid(True)
plt.show()


# In[45]:


from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.preprocessing import PolynomialFeatures

# Feature Engineering - Polynomial Features (Degree 2)
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X)
poly_feature_names = poly.get_feature_names_out(X.columns)
X_poly_df = pd.DataFrame(X_poly, columns=poly_feature_names)

# Combine original and polynomial features
X_combined = pd.concat([pd.DataFrame(X, columns=X.columns), X_poly_df], axis=1)

# Ensure all features are numerical and consistent
X_combined = X_combined.apply(pd.to_numeric, errors='coerce').dropna()
y = y[X_combined.index]

# Reset index to ensure consistency
X_combined = X_combined.reset_index(drop=True)
y = y.reset_index(drop=True)


# In[47]:


# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.3, random_state=1)

# Feature Selection with SelectFromModel
rf_model = RandomForestRegressor(n_estimators=10, random_state=1, n_jobs=-1)
sfm = SelectFromModel(estimator=rf_model, threshold='mean')
sfm.fit(X_train, y_train)
X_train_sfm = sfm.transform(X_train)
X_test_sfm = sfm.transform(X_test)


# In[49]:


# Train a RandomForest model on the selected features
rf_model.fit(X_train_sfm, y_train)
y_test_pred_sfm = rf_model.predict(X_test_sfm)

# Calculate evaluation metrics
mae_rf_sfm = mean_absolute_error(y_test, y_test_pred_sfm)
rmse_rf_sfm = mean_squared_error(y_test, y_test_pred_sfm, squared=False)
print(f"Random Forest with SelectFromModel - MAE: {mae_rf_sfm:.2f}, RMSE: {rmse_rf_sfm:.2f}")

# Visualization: Plot of Predicted vs Actual GDP Growth Rate (SelectFromModel)
plt.figure(figsize=(10, 8))
plt.scatter(y_test, y_test_pred_sfm, alpha=0.5, color='green', label='Predicted vs Actual (RF with SelectFromModel)')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', linewidth=2, label='Ideal Prediction Line')
plt.xlabel('Actual GDP Growth Rate')
plt.ylabel('Predicted GDP Growth Rate')
plt.title('Predicted vs Actual GDP Growth Rate (Random Forest with SelectFromModel)')
plt.legend()
plt.grid(True)
plt.show()


# In[60]:


# Feature Engineering - Polynomial Features (Degree 2)
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X)
poly_feature_names = poly.get_feature_names_out(X.columns)
X_poly_df = pd.DataFrame(X_poly, columns=poly_feature_names)

# Combine original and polynomial features
X_combined = pd.concat([pd.DataFrame(X, columns=X.columns), X_poly_df], axis=1)

# Ensure all features are numerical and consistent
X_combined = X_combined.apply(pd.to_numeric, errors='coerce').dropna()
y = y.reset_index(drop=True)
X_combined = X_combined.reset_index(drop=True)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.3, random_state=1)

# Feature Selection with RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=10, random_state=1, n_jobs=-1)
rf_model.fit(X_train, y_train)

# Get feature importances and sort them in descending order
feature_importances = rf_model.feature_importances_
sorted_indices = np.argsort(feature_importances)[::-1]

# Select only the top 20 features
top_n = 20
top_indices = sorted_indices[:top_n]
top_feature_names = X_combined.columns[top_indices].tolist()
print("Top 20 Features after SelectFromModel:")
print(top_feature_names)

# Train a RandomForest model using only the top 20 features
X_train_top = X_train.iloc[:, top_indices]
X_test_top = X_test.iloc[:, top_indices]
rf_model.fit(X_train_top, y_train)
y_test_pred_top = rf_model.predict(X_test_top)

# Calculate evaluation metrics
mae_rf_top = mean_absolute_error(y_test, y_test_pred_top)
rmse_rf_top = mean_squared_error(y_test, y_test_pred_top, squared=False)
print(f"Random Forest with Top 20 Features - MAE: {mae_rf_top:.2f}, RMSE: {rmse_rf_top:.2f}")

# Visualization: Plot of Predicted vs Actual GDP Growth Rate (Top 10 Features)
plt.figure(figsize=(10, 8))
plt.scatter(y_test, y_test_pred_top, alpha=0.5, color='blue', label='Predicted vs Actual (RF with Top 20 Features)')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', linewidth=2, label='Ideal Prediction Line')
plt.xlabel('Actual GDP Growth Rate')
plt.ylabel('Predicted GDP Growth Rate')
plt.title('Predicted vs Actual GDP Growth Rate (Random Forest with Top 20 Features)')
plt.legend()
plt.grid(True)
plt.show()


# In[52]:


from sklearn.decomposition import PCA
# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.3, random_state=1)

# Feature Selection with PCA
pca = PCA(n_components=20)  # Adjust the number of components accordingly
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Train a RandomForest model on the PCA features
rf_model = RandomForestRegressor(n_estimators=10, random_state=1, n_jobs=-1)
rf_model.fit(X_train_pca, y_train)
y_test_pred_pca = rf_model.predict(X_test_pca)

# Calculate evaluation metrics
mae_rf_pca = mean_absolute_error(y_test, y_test_pred_pca)
rmse_rf_pca = mean_squared_error(y_test, y_test_pred_pca, squared=False)
print(f"Random Forest with PCA - MAE: {mae_rf_pca:.2f}, RMSE: {rmse_rf_pca:.2f}")

# Visualization: Plot of Predicted vs Actual GDP Growth Rate (PCA)
plt.figure(figsize=(10, 8))
plt.scatter(y_test, y_test_pred_pca, alpha=0.5, color='purple', label='Predicted vs Actual (RF with PCA)')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', linewidth=2, label='Ideal Prediction Line')
plt.xlabel('Actual GDP Growth Rate')
plt.ylabel('Predicted GDP Growth Rate')
plt.title('Predicted vs Actual GDP Growth Rate (Random Forest with PCA)')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:




