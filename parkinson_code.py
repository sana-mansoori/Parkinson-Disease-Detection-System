import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score

import warnings
warnings.filterwarnings('ignore')

df=pd.read_csv("parkinsons.data")

df
df['status'].value_counts()
sb.countplot(x='status',data=df)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows',None)
df

df.head()
df.info()
df.describe()
df.shape
df.isnull().sum()
df['status'].value_counts()
sb.countplot(df.status)

X=df.drop(columns=['name', 'status'],axis=1)

Y=df['status']

print(X)
print(Y)
X_train, X_test, Y_train, Y_test= train_test_split(X,Y, test_size=0.2, random_state=2)

print(X.shape, X_train.shape, X_test.shape)
ss=StandardScaler()
ss.fit(X_train)

X_train= ss.transform(X_train)
X_test= ss.transform(X_test)

print(X_train)

print(X_test)

rfc = RandomForestClassifier(random_state=2)

params = {'n_estimators': [100, 200, 300, 400, 500], 'max_depth': [1, 5, 10, 15, 20]}
grid_search = GridSearchCV(rfc, params, cv=5)

grid_search.fit(X_train, Y_train)
y_pred = grid_search.predict(X_test)

confusion = confusion_matrix(Y_test, y_pred)
print("Confusion matrix:\n", confusion)
report = classification_report(Y_test, y_pred)
print("Classification report:\n", report)
accuracy = grid_search.score(X_test, Y_test)
print(f"Accuracy: {accuracy:.2%}")

sb.heatmap(confusion, cmap='Blues', annot=True, fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')

model= svm.SVC(kernel= 'linear')
model.fit(X_train, Y_train)

X_train_pred=model.predict(X_train)
train_data_acc=accuracy_score(Y_train, X_train_pred)

print("Accuracy of training dat :", train_data_acc)

X_test_pred =model.predict(X_test)
test_data_acc= accuracy_score(Y_test, X_test_pred)

print("Accuracy of testing data :",test_data_acc)

input_data=(107.33200,113.84000,104.31500,0.00290,0.00003,0.00144,0.00182,0.00431,0.01567,0.13400,0.00829,0.00946,0.01256,0.02487,0.00344,26.89200,0.637420,0.763262,-6.167603,0.183721,2.064693,0.163755)
input_data_np = np.asarray(input_data)
input_data_re = input_data_np.reshape(1 , -1)
s_data = ss.transform(input_data_re)
pred=model.predict(s_data)

print(pred)

if(pred[0]==0):
    print("Negative, No Parkinson's Disease found")
else:
    print("Positive, Parkinson's Disease found")