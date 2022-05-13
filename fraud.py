# importing all libraries
import math
import scipy
import sklearn
import numpy as np
import pandas as pd
from sklearn import *
import seaborn as sns
from pylab import rcParams
from sklearn.svm import SVC
rcParams['figure.figsize']=14,8
RANDOM_SEED=42
LABELS=["Normal", "Fraud"]
from matplotlib import style
from matplotlib import pyplot as plt
from ssl import create_default_context
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# Get the data
credit_data=pd.read_csv('creditcard.csv') #last column in data is "class"  which lets us know if it is fraudulent

#284807 rows and 31 columns
print(credit_data.info)

x=credit_data.iloc[:,1:30].values
y=credit_data.iloc[:,30].values # gets all rows of 30th column

print("Input Range: ", x.shape)
print("Output Range: ", y.shape)

# most transactions are valid since data will output most 0
print("Class Labels: ", y)

# looks for null value in out data
if (credit_data.isnull().values.any()):
  print("Null data value found")
else:
  print("No null dataset found")


# find class label's count
transactions=pd.value_counts(credit_data['Class'], sort=True)
#  number of 0: normal transactions vs 1: frauds 
plt.figure(figsize=(10,6))
transactions.plot(kind='bar', rot=0, color=['green', 'blue'], width=0.4)
plt.title("Class Distribution of Transaction")
plt.xticks(range(2), LABELS)
plt.xlabel("Classes")
plt.ylabel("Number of Occurances")
plt.show()


# seperate normal vs fraud data set
normal_data=credit_data[credit_data['Class']==0]
fraud_data=credit_data[credit_data['Class']==1]
print("Number of Real Transactions: ", len(normal_data))
print("Number of Fraud Transactions : ", len(fraud_data))
#statistical summary of both datasets
normal_data.Amount.describe()
fraud_data.Amount.describe()


# get correlation of each features in dataset
corrmat= credit_data.corr()
top_corr_features =corrmat.index
# plot heat map
g=sns.heatmap(credit_data[top_corr_features].corr(), annot=True,cmap="RdYlGn", linewidth=0.2)
plt.title("Heat Map")
plt.show()

# or show 2d heat map
def heatmap2d(arr: np.ndarray):
  plt.title("2D Heat Map")
  plt.figure(figsize=(20,20))
  plt.imshow(arr, cmap='viridis')
  plt.colorbar()
  plt.show()
heatmap2d(credit_data[top_corr_features].corr())

# time column is not needed
credit_data.drop('Time', axis=1, inplace=True)
# independent columns except for class
X=credit_data.drop(['Class'], axis=1)
# our target: class-dependent variable
Y=credit_data['Class']


# randomness should be there so Logistic regression will be applied
x_train, x_test, y_train, y_test=train_test_split(x,y, train_size=0.75,  test_size=0.25, random_state=100) #training set=75%     
print("Xtrain: ", x_train.shape)
print("Xtest: ", x_test.shape)
print("Ytrain: ", y_train.shape)
print("Ytest: ", y_test.shape)

# transform training and test to unifrom distribution  
stdsc= StandardScaler()
# values converted between standard deviation (+-)
x_train=stdsc.fit_transform(x_train)
x_test=stdsc.transform(x_test) 
print("Training Set after Standardised: \n",x_train[0])
print("Testing Set after Standardised: \n",x_test[0])

# check if proper label is being predicted or not 
clsf_dt = DecisionTreeClassifier(criterion = 'entropy', random_state = 0) #entropy=major quality spiltting
clsf_dt.fit(x_train, y_train) #associate trains into dt
# check if outside data will give accurate result 
predict_dt = clsf_dt.predict(x_test)
# shows 1=we have data with fraud or 0=no fraud
print("Decision Tree Prediction : \n", predict_dt)

# compares accurate and predicted data 
conf_matrix = confusion_matrix(y_test, predict_dt)
print("Confusion Matrix : \n", conf_matrix)

# check doc for matrix work  
# (0,0) True Negative=Predicted Nonfraudlent correctly compared to actual data
# and (1,1) True Positive= predicted fraud correctly 
accuracy = ((conf_matrix[0][0] + conf_matrix[1][1]) / conf_matrix.sum()) *100
# (0,1) False Positive and (1,0) False Negative 
error_rate= ((conf_matrix[0][1] + conf_matrix[1][0]) / conf_matrix.sum()) *100
# Accurate Fake Rate: TP/Predicted Yes
precision_specificity = (conf_matrix[1][1] / (conf_matrix[1][1] + conf_matrix[0][1])) *100
# Genuine Rate = TN/Predicted No
sensitivity = (conf_matrix[0][0] / (conf_matrix[0][0] + conf_matrix[1][0])) *100

# accuracy = (TP + TN) / Total
print("Accuracy    : ", accuracy)
print("Error Rate  : ", error_rate)
print("Precision   : ", precision_specificity)
print("Sensitivity : ", sensitivity)

# convert low dimensional features to high dimensional features- seperate easily
svc_clsf = SVC(kernel = 'rbf', random_state =0)
svc_clsf.fit(x_train, y_train)

# predict unseen data 
predict_clsf = svc_clsf.predict(x_test)
print("Predict Random Unseen Forest : \n", predict_clsf)

conf_matrix2 = confusion_matrix(y_test, predict_clsf)
print("Confusion Matrix 2 : \n", conf_matrix2)

# Prediction Validation
accuracy = ((conf_matrix2[0][0] + conf_matrix2[1][1]) / conf_matrix2.sum()) *100
error_rate = ((conf_matrix2[0][1] + conf_matrix2[1][0]) / conf_matrix2.sum()) *100
precision_specificity= (conf_matrix2[1][1] / (conf_matrix2[1][1] + conf_matrix2[0][1])) *100
sensitivity= (conf_matrix2[0][0] / (conf_matrix2[0][0] + conf_matrix2[1][0])) *100

print("Support Vector Classifier's Prediction")
print("Accuracy    : ", accuracy)
print("Error Rate  : ", error_rate)
print("Precision   : ", precision_specificity)
print("Sensitivity : ", sensitivity)


print("Hence proved SVC model is better choice for credit fraud prediction")