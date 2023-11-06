import pandas as pd
import numpy as np

# Load the dataset
file_path = 'Data/Fulldataset_withoutCountryYear - GVC_sorted.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataframe
data.head()


import seaborn as sns
import matplotlib.pyplot as plt

# Plotting the distributions of the features and the target variable
plt.figure(figsize=(15, 10))

for i, column in enumerate(data.columns, 1):
    plt.subplot(2, 3, i)
    sns.histplot(data[column], kde=True)
    plt.title(f'Distribution of {column}')

plt.tight_layout()
plt.show()


# Compute the correlation matrix
correlation_matrix = data.corr()

# Visualize the correlation matrix with a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()


# For the purpose of the analysis, we will assume that GVC, FVA, and DVX should be ratios or percentages
# Thus, we will scale them down assuming they are currently represented in a larger unit (e.g., a factor of 10,000 or 100,000)

# Scale down by a factor (for example 100,000 if they are in tens of thousands or more)
scaling_factor = 100000

# Apply scaling
data_scaled = data.copy()
data_scaled['GVC'] = data_scaled['GVC'] / scaling_factor
data_scaled['FVA'] = data_scaled['FVA'] / scaling_factor
data_scaled['DVX'] = data_scaled['DVX'] / scaling_factor

# Now let's check the new summary statistics and distributions
data_scaled[['GVC', 'FVA', 'DVX']].describe()


# Apply log transformation to ENS to normalize its distribution
data_scaled['ENS_log'] = np.log(data_scaled['ENS'])

# Check the new summary statistics for ENS_log
data_scaled['ENS_log'].describe()


# Plotting the distributions of the scaled and transformed variables
plt.figure(figsize=(15, 10))

# Plotting the original ENS distribution
plt.subplot(2, 3, 1)
sns.histplot(data['ENS'], kde=True)
plt.title('Original Distribution of ENS')

# Plotting the transformed ENS distribution
plt.subplot(2, 3, 2)
sns.histplot(data_scaled['ENS_log'], kde=True)
plt.title('Log-transformed Distribution of ENS')

# Plotting the scaled GVC distribution
plt.subplot(2, 3, 3)
sns.histplot(data_scaled['GVC'], kde=True)
plt.title('Scaled Distribution of GVC')

# Plotting the scaled FVA distribution
plt.subplot(2, 3, 4)
sns.histplot(data_scaled['FVA'], kde=True)
plt.title('Scaled Distribution of FVA')

# Plotting the scaled DVX distribution
plt.subplot(2, 3, 5)
sns.histplot(data_scaled['DVX'], kde=True)
plt.title('Scaled Distribution of DVX')

plt.tight_layout()
plt.show()


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectFromModel
import numpy as np

# Selecting the features and target variable
X = data_scaled[['GVC', 'FVA', 'DVX', 'URPOP']]
y = data_scaled['ENS_log']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initializing the Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Fitting the model
rf.fit(X_train, y_train)

# Predicting on the test set
y_pred = rf.predict(X_test)

# Calculating the RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Feature selection using SelectFromModel
selector = SelectFromModel(rf, prefit=True)
X_important_train = selector.transform(X_train)
X_important_test = selector.transform(X_test)

# Training the model again only on important features
rf_important = RandomForestRegressor(n_estimators=100, random_state=42)
rf_important.fit(X_important_train, y_train)

# Predicting on the test set with only important features
y_important_pred = rf_important.predict(X_important_test)

# Calculating the RMSE for the model with selected features
important_rmse = np.sqrt(mean_squared_error(y_test, y_important_pred))

# Get the names of the important features
important_features = X.columns[selector.get_support()]



# Get feature importances from the random forest model
feature_importances = rf.feature_importances_

# Convert the importances into a dataframe
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
})

# Sort the dataframe by importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plotting the feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importances from Random Forest')
plt.xlabel('Relative Importance')
plt.ylabel('Feature')
plt.show()

print(f'RMSE with all features: {rmse}\n RMSE with {important_features.tolist()}: {important_rmse}')