import numpy as np
from tkinter import filedialog
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import seaborn as sns
from xgboost import XGBClassifier

dataset = pd.read_csv('enc_heart.csv')
dataset.fillna(0, inplace = True)

dataset = dataset.values
cols = dataset.shape[1]-1
X = dataset[:,0:cols]
Y = dataset[:,cols]
X = normalize(X)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,random_state=0)

classifier = XGBClassifier() #object of extreme gradient boosting
classifier.fit(X_train, y_train)#training xgb on daatset
predict = classifier.predict(X_test)
gbc_acc = accuracy_score(predict, y_test)
print("Graident Boosting Diabetes Prediction Accuracy : "+str(gbc_acc)+"\n\n")
print(y_test)
print(predict)

test = pd.read_csv('enc_testData.csv')
test = test.values
test = normalize(test)    
predict = classifier.predict(test)
print(predict)
