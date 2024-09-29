
#Calling all the required libraries:
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pickle


#Loading the dataset:
zomato_data_nw = pd.read_csv("./data/zomato_df_final_data.csv")

# Checking for total NA values in the dataset
total_na_count = zomato_data_nw.isna().sum()

# Checking for any empty cells (cells with empty strings) in the dataset
empty_cell = (zomato_data_nw == '').sum()

# Since we will be making "rating_number as our target variable, lets remove the rows having NA or Null values for rating number.

# Removing the rows where 'rating_number' has NA or NaN values
zomato_new_cleaned = zomato_data_nw.dropna(subset=['rating_number', 'cost'])

# Rechecking the number of NA or NaN values in each column
na_count_nw = zomato_new_cleaned.isna().sum()
na_columns = na_count_nw[na_count_nw > 0]

# Adding a new column 'suburb' that extracts only the suburb names in the 'subzone' column
zomato_new_cleaned.loc[:, 'suburb'] = zomato_new_cleaned['subzone'].apply(lambda x: x.split(',')[-1].strip())

# Selecting only numerical columns
numerical_columns = zomato_new_cleaned.select_dtypes(include=['float64', 'int64'])

# Calculating correlation between 'rating_number' and other numerical columns
correlation = numerical_columns.corr()['rating_number'].sort_values(ascending=False)

# Checking the data types of each column
column_types = zomato_new_cleaned.dtypes


# #### Label/Feature Encoding:

# #### Lets convert the non numeric columns using Label encoder to find out the correlation for all features.

# Using Label Encoding for non-numeric columns
from sklearn.preprocessing import LabelEncoder

# Selecting only the non-numeric columns
non_numeric_columns = zomato_new_cleaned.select_dtypes(include=['object']).columns

# Initializing LabelEncoder
lebel_encoder = LabelEncoder()

# Creating a copy of the dataframe to avoid modifying the original one
zomato_encoded = zomato_new_cleaned.copy()

# Applying Label Encoding for non-numeric columns
for col in non_numeric_columns:
    zomato_encoded[col] = zomato_encoded[col].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
    zomato_encoded[col] = lebel_encoder.fit_transform(zomato_encoded[col].astype(str))

# Calculating the correlation matrix
correlation = zomato_encoded.corr()['rating_number'].sort_values(ascending=False)

# #### From the output we can see, features having high correlation are: "rating_number", "rating_text", "votes ", "cost", "cost_2", "cuisine" and "suburb". 
# Also, the "rating_text is same as the "rating_number", we have to exclude this during modelling. So we will be considering only the remaining highly correlated features for our modeling.

# Selecting only the specified features that determine our target variable.
selected_features = ['votes', 'cost', 'cost_2', 'cuisine', 'suburb']

# Preparing data with the selected features
X = zomato_encoded[selected_features]  # Use selected features
y = zomato_encoded['rating_number']

# Handling missing values if any, by filling with median of each column
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

# Spliting the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=0)

# Standardizing the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model 1: Linear Regression
linear_model_1 = LinearRegression()
linear_model_1.fit(X_train_scaled, y_train)

# Predicting and calculating MSE for model 1
y_pred_1 = linear_model_1.predict(X_test_scaled)
mse_model_1 = mean_squared_error(y_test, y_pred_1)

# Model 2: Linear Regression with Gradient Descent (SGDRegressor)
linear_model_2 = SGDRegressor(max_iter=1000, tol=1e-3, random_state=0)
linear_model_2.fit(X_train_scaled, y_train)

# Predicting and calculating MSE for model 2
y_pred_2 = linear_model_2.predict(X_test_scaled)
mse_model_2 = mean_squared_error(y_test, y_pred_2)

mse_model_1, mse_model_2

print("mse for liner model is:", mse_model_1)
print("mse for SGD Regressor: ", mse_model_2)

# Saving the linear model:
with open('LinearModel1.pkl', 'wb') as f:
        pickle.dump(linear_model_1, f)

with open('LinearModel2.pkl', 'wb') as f:
        pickle.dump(linear_model_2, f)



# ### III 7. Build a logistic regression model (model_classification_3) for the simplified data, with 80% training data and 20% test data.

# Binary Classes: ***Class 1: 'Poor' and 'Average'*** and, *** Class 2: 'Good', 'Very Good', 'Excellent' ***

#Creating a copy
zomato_binary = zomato_new_cleaned.copy()

#Performing the binary classification
zomato_binary['rating_class'] = zomato_binary['rating_text'].map({
    'Poor': 1,
    'Average': 1,
    'Good': 2,
    'Very Good': 2,
    'Excellent': 2
})

# Using Label Encoding for non-numeric columns
from sklearn.preprocessing import LabelEncoder

# Initializing LabelEncoder
lebel_encoder = LabelEncoder()

# Creating a copy of the dataframe
zomato_binary_encoded = zomato_binary.copy()

# Applying Label Encoding for non-numeric columns
for col in non_numeric_columns:
    zomato_binary_encoded[col] = zomato_binary_encoded[col].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
    zomato_binary_encoded[col] = lebel_encoder.fit_transform(zomato_binary_encoded[col].astype(str))

# Selecting only the most relevant features
selected_features = ['votes', 'cost', 'cost_2', 'cuisine', 'suburb']

# Preparing data with the selected features
X = zomato_binary_encoded[selected_features]  # Use selected features
y = zomato_binary_encoded['rating_class']

# Spliting the dataset into training - 80% and testing - 20% sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Building the Logistic Regression Model
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train, y_train)

# Predictions
y_pred_logistic = logistic_model.predict(X_test)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred_logistic)
print("Confusion Matrix:\n", conf_matrix)

# Classification Report
class_report = classification_report(y_test, y_pred_logistic)
print("Classification Report:\n", class_report)

# Saving the logistic model:
with open('LogisticModel.pkl', 'wb') as f:
        pickle.dump(logistic_model, f)


# Selecting only the most relevant features
selected_features = ['votes', 'cost', 'cost_2', 'cuisine', 'suburb']

# Preparing data with the selected features
X = zomato_binary_encoded[selected_features]  # Use selected features
y = zomato_binary_encoded['rating_class']

# Spliting the dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Model 1: Random Forest Classifier
randomforest_model = RandomForestClassifier(random_state=0)
randomforest_model.fit(X_train, y_train)

# Model 2: Support Vector Classifier (SVC)
svc_model = SVC(random_state=0)
svc_model.fit(X_train, y_train)

# Model 3: K-Nearest Neighbors (KNN)
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)

# Predictions for each model
y_pred_randomforest = randomforest_model.predict(X_test)
y_pred_svc = svc_model.predict(X_test)
y_pred_knn = knn_model.predict(X_test)

# Evaluation of Random Forest Classifier
print("Random Forest Classifier Performance")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_randomforest))
print("Classification Report:\n", classification_report(y_test, y_pred_randomforest))

# Evaluation of Support Vector Classifier (SVC)
print("\nSupport Vector Classifier Performance")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svc))
print("Classification Report:\n", classification_report(y_test, y_pred_svc))

# Evaluation of K-Nearest Neighbors (KNN)
print("\nK-Nearest Neighbors Performance")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_knn))
print("Classification Report:\n", classification_report(y_test, y_pred_knn))

# Saving the models:
with open('RandomForestModel.pkl', 'wb') as f:
        pickle.dump(randomforest_model, f)

with open('SVCmodel.pkl', 'wb') as f:
        pickle.dump(svc_model, f)

with open('KNNmodel.pkl', 'wb') as f:
        pickle.dump(knn_model, f)

# Printing Accuracy:

#Logistic Regression:
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
print("Accuracy for Logistic Regression:", accuracy_logistic)

# Random Forest Classifier
y_pred_randomforest = randomforest_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_randomforest)
print("Accuracy for Random Forest Classifier:", accuracy_rf)

# Support Vector Classifier (SVC)
y_pred_svc = svc_model.predict(X_test)
accuracy_svc = accuracy_score(y_test, y_pred_svc)
print("Accuracy for Support Vector Classifier:", accuracy_svc)

# K-Nearest Neighbors (KNN)
y_pred_knn = knn_model.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print("Accuracy for K-Nearest Neighbors:", accuracy_knn)
