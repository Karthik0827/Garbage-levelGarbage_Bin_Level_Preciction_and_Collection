#!/usr/bin/env python
# coding: utf-8

# # Importing required libraries

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import mean_squared_error


# # Suppress warnings

# In[2]:


import warnings
warnings.filterwarnings('ignore')


# # Load data

# In[3]:


df=pd.read_csv("C:/Users/geeth/Downloads/trash data PS-2 (5).csv")
df


# In[4]:


df.info()


# In[5]:


df.isnull().sum()


# In[6]:


df.nunique()


# # Data Exploration

# ## Check columns

# In[7]:


cat_cols=df.select_dtypes(include=['object']).columns

num_cols = df.select_dtypes (include=np.number).columns.tolist()

print("Categorical Variables:")

print(cat_cols)

print("Numerical Variables:")

print(num_cols)


# ## Explore numerical variables

# In[8]:


for col in num_cols:
    print(col)
    print('Skew:', round(df[col].skew(), 2))
    plt.figure(figsize = (15, 4))
    plt.subplot(1, 2, 1)
    df[col].hist(grid=False)
    plt.ylabel('count')
    plt.subplot(1, 2, 2)
    sns.boxplot(x=df[col])
    plt.show()


# ## Explore categorical variables
# 

# In[9]:


fig, axes = plt.subplots (5, 2, figsize = (18, 18))

fig.suptitle('Bar plot for all categorical variables in the dataset', fontsize=24)
sns.countplot(ax = axes[0, 0], x = 'BIN ID', data = df, color = 'blue', order = df['BIN ID'].value_counts().index);
sns.countplot(ax = axes[0, 1], x = 'Date', data = df, color = 'blue', order = df['Date'].value_counts().index);
sns.countplot(ax = axes[1, 0], x = 'TIME', data = df, color = 'blue', order = df['TIME'].value_counts().index);
sns.countplot(ax = axes[1, 1], x = 'FILL PERCENTAGE', data = df, color = 'blue', order = df['FILL PERCENTAGE'].value_counts().index);
sns.countplot(ax = axes[2, 0], x = 'LOCATION ', data = df, color = 'blue', order = df['LOCATION '].value_counts().index);
sns.countplot(ax = axes[2, 1], x = 'LATITUDE', data = df, color = 'blue', order = df['LATITUDE'].value_counts().index);
sns.countplot(ax = axes[3, 0], x = 'LONGITUDE', data = df, color = 'blue', order = df['LONGITUDE'].value_counts().index);
sns.countplot(ax = axes[3, 1], x = 'TEMPERATURE( IN ⁰C)' , data = df, color = 'blue', order = df['TEMPERATURE( IN ⁰C)'].value_counts().index);
sns.countplot(ax = axes[4, 0], x = 'BATTERY LEVEL ', data = df, color = 'blue', order = df['BATTERY LEVEL '].value_counts().index);


# # Pairplot
# 

# In[10]:


plt.figure(figsize=(15,20))

sns.pairplot(df)

plt.show()


# # Grouped bar plots
# 

# In[11]:


fig, axarr= plt.subplots(5, 2, figsize=(12, 18))

df.groupby('BIN ID')['FILL LEVEL INDICATOR(Above 550)'].mean().sort_values(ascending=False).plot.bar(ax=axarr[0][0], fontsize=12)
axarr[0][0].set_title("BIN ID Vs FILL LEVEL INDICATOR(Above 550)", fontsize=14)

df.groupby('Date')['FILL LEVEL INDICATOR(Above 550)'].mean().sort_values(ascending=False).plot.bar(ax=axarr[0][1], fontsize=12)
axarr[0][1].set_title("Date Vs FILL LEVEL INDICATOR(Above 550)", fontsize=14)

df.groupby('TIME')['FILL LEVEL INDICATOR(Above 550)'].mean().sort_values(ascending=False).plot.bar(ax=axarr[1][0], fontsize=12)
axarr[1][0].set_title("TIME Vs FILL LEVEL INDICATOR(Above 550)", fontsize=14)

df.groupby('FILL PERCENTAGE')['FILL LEVEL INDICATOR(Above 550)'].mean().sort_values(ascending=False).plot.bar(ax=axarr[1][1], fontsize=12)
axarr[1][1].set_title("FILL PERCENTAGE Vs FILL LEVEL INDICATOR(Above 550)", fontsize=14)

df.groupby('LOCATION ')['FILL LEVEL INDICATOR(Above 550)'].mean().sort_values(ascending=False).head(10).plot.bar(ax=axarr[2][0], fontsize=12)
axarr[2][0].set_title("LOCATION Vs FILL LEVEL INDICATOR(Above 550)", fontsize=14)

df.groupby('LATITUDE')['FILL LEVEL INDICATOR(Above 550)'].mean().sort_values(ascending=False).plot.bar(ax=axarr[2][1], fontsize=12)
axarr[2][1].set_title("LATITUDE Vs FILL LEVEL INDICATOR(Above 550)", fontsize=14)

df.groupby('LONGITUDE')['FILL LEVEL INDICATOR(Above 550)'].mean().sort_values(ascending=False).plot.bar(ax=axarr[3][0], fontsize=12)
axarr[3][0].set_title("LONGITUDE Vs FILL LEVEL INDICATOR(Above 550)", fontsize=14)

df.groupby('TEMPERATURE( IN ⁰C)')['FILL LEVEL INDICATOR(Above 550)'].mean().sort_values(ascending=False).plot.bar(ax=axarr[3][1], fontsize=12)
axarr[3][1].set_title("TEMPERATURE( IN ⁰C) Vs FILL LEVEL INDICATOR(Above 550)", fontsize=14)

df.groupby('BATTERY LEVEL ')['FILL LEVEL INDICATOR(Above 550)'].mean().sort_values(ascending=False).plot.bar(ax=axarr[4][0], fontsize=12)
axarr[4][0].set_title("BATTERY LEVEL  Vs FILL LEVEL INDICATOR(Above 550)", fontsize=14)

plt.subplots_adjust(hspace=1.2)

plt.subplots_adjust(wspace=.3)

sns.despine()


# # Encode categorical variables
# 

# In[12]:


# Define the columns to encode
columns = ['BIN ID','Date','TIME', 'LOCATION ', 'LATITUDE', 'LONGITUDE','FILL LEVEL INDICATOR(Above 550)','FILL PERCENTAGE','BATTERY LEVEL ','TEMPERATURE( IN ⁰C)']

# Initialize the LabelEncoder
encoder = LabelEncoder()

# Encode each column separately
for column in columns:
    df[column] = encoder.fit_transform(df[column])

# Now df contains the encoded values for each column
df


# # Count of unique values for categorical columns
# 

# In[13]:


categorical_columns = ['BIN ID','WEEK NO','TOTAL(LITRES)','LOCATION ','LATITUDE','LONGITUDE','FILL LEVEL INDICATOR(Above 550)']
for column in categorical_columns:
    plt.figure(figsize=(8, 6))
    sns.countplot(df[column])
    plt.title(f'Count of unique values in {column}')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()


# In[14]:


df.describe().T


# # Heatmap of correlations
# 

# In[15]:


fig, ax = plt.subplots(figsize=(20,16))
sns.heatmap(df.corr(), annot=True, cmap='Greens',annot_kws={"fontsize":16})


# # Modeling

# ## Split data

# In[16]:


x=df.drop(['BIN ID','Date','TIME','WEEK NO','TOTAL(LITRES)','BATTERY LEVEL ','FILL LEVEL INDICATOR(Above 550)'],axis=1)
y=df['FILL LEVEL INDICATOR(Above 550)']
x


# In[17]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)


# ## Logistic Regression
# 

# In[18]:


lr=LogisticRegression(max_iter=1000)
lr.fit(x_train,y_train)
pred_1=lr.predict(x_test)
acc_1=accuracy_score(y_test,pred_1)
mse_lr = mean_squared_error(y_test, pred_1)
print("Accuracy:", acc_1)
print("Mean Squared Error:", mse_lr)


# ## Random Forest
# 

# In[19]:


rfc=RandomForestClassifier()
rfc.fit(x_train,y_train)
pred_2=rfc.predict(x_test)
acc_2=accuracy_score(y_test,pred_2)
mse_rfc = mean_squared_error(y_test, pred_2)
print("Accuracy:", acc_2)
print("Mean Squared Error:", mse_rfc)


# ## KNN
# 

# In[20]:


best_k = None
best_acc_knn = 0
for i in range(1,21):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train,y_train)
    preds=knn.predict(x_test)
    acc_3=accuracy_score(y_test,preds)
    mse_knn = mean_squared_error(y_test,preds)
    if acc_3 > best_acc_knn:
        best_acc_knn = acc_3
        best_k = i
    print("\nK-Nearest Neighbors (k =", i, "):")
    print("Accuracy:", acc_3)
    print("Mean Squared Error:", mse_knn)


# ## Custom Linear Regression

# In[21]:


import numpy as np
from sklearn.metrics import mean_squared_error

class CustomLinearRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.theta = None
    
    def fit(self, x, y):
        # Add bias term to input features
        ones = np.ones((x.shape[0], 1))
        x = np.concatenate((ones, x), axis=1)
        
        # Initialize parameters randomly
        np.random.seed(0)
        self.theta = np.random.rand(x.shape[1])
        
        # Gradient descent
        for _ in range(self.num_iterations):
            # Compute gradients
            gradients = np.dot(x.T, np.dot(x, self.theta) - y) / x.shape[0]
            
            # Update parameters
            self.theta -= self.learning_rate * gradients
    
    def predict(self, x):
        # Add bias term to input features
        ones = np.ones((x.shape[0], 1))
        x = np.concatenate((ones, x), axis=1)
        
        # Predict target variable
        return np.dot(x, self.theta)
    def score(self, X, y):
        # Calculate R^2 score or any other metric you want to use for scoring
        predictions = self.predict(X)
        return your_custom_scoring_function(y, predictions)
    def get_params(self, deep=True):
        return {'learning_rate': self.learning_rate, 'num_iterations': self.num_iterations}

# Example usage:
# Assuming x_train, y_train, x_test, y_test are your training and testing data
custom_lr = CustomLinearRegression(learning_rate=0.01, num_iterations=1000)
custom_lr.fit(x_train, y_train)
predictions = custom_lr.predict(x_test)

# Handle NaN values
predictions = np.nan_to_num(predictions)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error (MSE):", mse)

# Calculate Accuracy Score
threshold = 0.5  # Example threshold
predicted_classes = (predictions >= threshold).astype(int)
true_classes = (y_test >= threshold).astype(int)
accuracy = (predicted_classes == true_classes).mean()
print("Accuracy Score:", accuracy)


# # Cross-validation

# In[22]:


from sklearn.model_selection import cross_val_score
lr_cv_scores = cross_val_score(lr, x, y, cv=5)
print("Logistic Regression Cross-Validation Accuracy:", np.mean(lr_cv_scores))

rfc_cv_scores = cross_val_score(rfc, x, y, cv=5)
print("Random Forest Cross-Validation Accuracy:", np.mean(rfc_cv_scores))

knn_cv_scores = cross_val_score(knn, x, y, cv=5)
print(f"KNN Cross-Validation Accuracy with {best_k} neighbors:", np.mean(knn_cv_scores))

# Cross-validation for Custom Linear Regression
custom_lr_cv_scores = cross_val_score(custom_lr, x, y, cv=5)
print("Custom Linear Regression Cross-Validation Accuracy:", np.mean(custom_lr_cv_scores))


# # Metrics and Evaluation

# In[23]:


from sklearn import metrics
confusion_matrix = metrics.confusion_matrix(y_test, preds)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
cm_display.plot()
plt.show()


# In[24]:


f1 = f1_score(y_test, preds)
print("F1 score:", f1)


# In[25]:


f1 = f1_score(y_test, preds)
print("F1 score:", f1)


# In[26]:


recall = recall_score(y_test, preds)
print("Recall: {:.2f}".format(recall))


# In[27]:


new_bin_data = [[5, 58, 1, 1, 1, 25]]
prediction = knn.predict(new_bin_data)
if prediction[0]== 0:
    print("The bin didn't fill yet")
else:
    print("The bin is full")

