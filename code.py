# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
df = pd.read_csv('/content/drive/MyDrive/Logistic_regression/Social_Network_Ads.csv')

# Display the first few rows of the dataset
df.head()

# Display the shape of the dataset
df.shape

# Display information about the dataset
df.info()

# Drop the 'User ID' column
df = df.drop('User ID', axis=1)

# Display the first few rows after dropping 'User ID'
df.head()

# Visualization of variables
r = 2
c = 2
it = 1
for i in df.columns:
    plt.suptitle("Visualizing all the variables")
    plt.subplot(r, c, it)
    if df[i].dtype == 'object':
        sns.countplot(x=df[i])
    else:
        sns.kdeplot(df[i])
        plt.grid()
    it += 1
plt.tight_layout()
plt.show()

# Descriptive statistics for 'Purchased' equal to 0
df[df['Purchased'] == 0].describe()

# Descriptive statistics for 'Purchased' equal to 1
df[df['Purchased'] == 1].describe()

# Convert categorical variable 'Gender' to dummy variables
df = pd.get_dummies(df, columns=['Gender'], drop_first=True)
df['Gender_Male'] = df['Gender_Male'].replace({True: 1, False: 0})

# Boxplot for numeric variables 'Age' and 'EstimatedSalary'
df[['Age', 'EstimatedSalary']].boxplot()

# Scale numeric variables using Min-Max scaling
mmax = MinMaxScaler()
df[['Age', 'EstimatedSalary']] = mmax.fit_transform(df[['Age', 'EstimatedSalary']])

# Boxplot after scaling
df.boxplot()
plt.show()

# Define target variable 'y' and features 'x'
y = df['Purchased']
x = df.drop('Purchased', axis=1)

# Add a constant to the features for Statsmodels logistic regression
xc = sm.add_constant(x)

# Fit logistic regression model using Statsmodels
model = sm.Logit(y, xc).fit()
print(model.summary())

# Split the data into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.30, random_state=42)

# Fit logistic regression model using scikit-learn
lr = LogisticRegression()
lr_model = lr.fit(xtrain, ytrain)

# Predictions on the training and testing sets
y_pred_train = lr_model.predict(xtrain)
y_pred_test = lr_model.predict(xtest)

# Evaluate the model
accuracy_score_train = accuracy_score(ytrain, y_pred_train)
accuracy_score_test = accuracy_score(ytest, y_pred_test)

precision_score_train = precision_score(ytrain, y_pred_train)
precision_score_test = precision_score(ytest, y_pred_test)

recall_score_train = recall_score(ytrain, y_pred_train)
recall_score_test = recall_score(ytest, y_pred_test)

f1_score_train = f1_score(ytrain, y_pred_train)
f1_score_test = f1_score(ytest, y_pred_test)

# Print evaluation metrics
print("Training Set:")
print(f"Accuracy: {accuracy_score_train:.2f}")
print(f"Precision: {precision_score_train:.2f}")
print(f"Recall: {recall_score_train:.2f}")
print(f"F1 Score: {f1_score_train:.2f}")

print("\nTest Set:")
print(f"Accuracy: {accuracy_score_test:.2f}")
print(f"Precision: {precision_score_test:.2f}")
print(f"Recall: {recall_score_test:.2f}")
print(f"F1 Score: {f1_score_test:.2f}")
