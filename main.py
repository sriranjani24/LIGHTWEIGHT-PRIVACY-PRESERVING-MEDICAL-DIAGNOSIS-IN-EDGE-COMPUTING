import os
from flask import Flask, render_template, request, redirect, Response
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
import pickle
from LPME import privacyPreservingTrain,privacyPreservingTest
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix

app = Flask(__name__)
app.secret_key = 'dropboxapp1234'
global classifier

@app.route("/Encrypt")
def Encrypt():
    privacyPreservingTrain('Dataset/heart.csv','EncryptedData/enc_heart.csv')
    plain = pd.read_csv('Dataset/heart.csv')
    encrypted = pd.read_csv('EncryptedData/enc_heart.csv')
    output = '<table border="1" align="center">'
    output+='<tr><th>Plain Dataset</th><th>Encrypted Datast</th></tr>'

    color = '<font size="" color="black">'
    output+='<tr><td>'+color+str(plain.head())+'</td><td>'+color+str(encrypted.head())+'</td></tr>'
    return render_template("AdminScreen.html",error=output)
    

@app.route("/TrainML")
def TrainML():
    global classifier
    #reading encrtypted data
    dataset = pd.read_csv('EncryptedData/enc_heart.csv')
    dataset.fillna(0, inplace = True)
    dataset = dataset.values
    cols = dataset.shape[1]-1
    #getting X and Y values from dataset
    X = dataset[:,0:cols]
    Y = dataset[:,cols]
    X = normalize(X)
    #dividing dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,random_state=0)

    classifier = XGBClassifier() #object of extreme gradient boosting
    classifier.fit(X_train, y_train)#training xgb on train data
    predict = classifier.predict(X_test) #predicting on test data
    gbc_acc = accuracy_score(predict, y_test) * 100#calculating accuracy of test data
    cm = confusion_matrix(predict, y_test)
    total=sum(sum(cm))
    specificity = cm[1,1]/(cm[1,0]+cm[1,1]) * 100 #recall calculation
    recall = recall_score(y_test,predict,average='macro') * 100 #specificity calculation

    output = '<table border="1" align="center">'
    output+='<tr><th>Algorithm Name</th><th>Accuracy</th><th>Recall</th><th>Specificity</th></tr>'

    color = '<font size="" color="black">'
    output+='<tr><td>XGBoost Algorithm</td><td>'+color+str(gbc_acc)+'</td><td>'+color+str(recall)+'</td><td>'+color+str(specificity)+'</td></tr>'
    output+='</table><br/><br/><br/><br/>'
    return render_template("AdminScreen.html",error=output)


@app.route('/Predict', methods =['GET', 'POST'])
def Predict():
    if request.method == 'GET':
        global classifier
        privacyPreservingTest('Dataset/testData.csv','EncryptedData/enc_testData.csv')
        test = pd.read_csv('EncryptedData/enc_testData.csv')
        print(test)
        test = test.values
        test = normalize(test)    
        predict = classifier.predict(test)
        output = '<table border="1" align="center">'
        output+='<tr><th>Heart Disease Test Data</th><th>Diagnosis Result</th></tr>'
        color = '<font size="" color="black">'
        for i in range(len(predict)):
            if predict[i] == 0:
                output+='<tr><td>'+color+str(test[i])+'</td><td>'+color+"No Heart Disease Predicted</td></tr>"
            else:
                output+='<tr><td>'+color+str(test[i])+'</td><td>'+color+"Heart Disease Predicted</td></tr>"
        output+='</table><br/><br/><br/><br/>'
        print(output)
        return render_template("AdminScreen.html",error=output)    
         
@app.route('/SignupAction', methods =['GET', 'POST'])
def SignupAction():
    if request.method == 'POST':
        username = request.form['t1']
        password = request.form['t2']
        contact = request.form['t3']
        email = request.form['t4']
        address = request.form['t5']
        users = []
        option = 0
        if os.path.exists('static/user.db'):
            with open('static/user.db', 'rb') as file:
                users = pickle.load(file)
            file.close()
        else:
            users.append(username+","+password+","+contact+","+email+","+address)
            with open('static/user.db', 'wb') as file:
                pickle.dump(users, file)
            file.close()
            option = 1
        if option == 0:
            for i in range(len(users)):
                arr = users[i].split(",")
                if arr[0] == username:
                    option = 2
                    break
        if option == 0 or option == 1:
            users.append(username+","+password+","+contact+","+email+","+address)
            with open('static/user.db', 'wb') as file:
                pickle.dump(users, file)
            file.close()
            return render_template("Signup.html",error='Signup process completed')
        else:
            return render_template("Signup.html",error=username+' username already exists')



@app.route("/index")
def index():
    return render_template("index.html")

@app.route("/Signup")
def Signup():
    return render_template("Signup.html")

@app.route("/Login")
def Login():
    return render_template("Login.html")

@app.route('/UserLogin', methods =['GET', 'POST'])
def UserLogin():
    if request.method == 'POST':
        username = request.form['t1']
        password = request.form['t2']
        users = []
        option = 0
        if os.path.exists('static/user.db'):
            with open('static/user.db', 'rb') as file:
                users = pickle.load(file)
            file.close()
        
        if option == 0:
            for i in range(len(users)):
                arr = users[i].split(",")
                if arr[0] == username and arr[1] == password:
                    option = 2
                    break
        if option == 2:
            return render_template("AdminScreen.html",error='Welcome '+username)
        else:
            return render_template("Login.html",error='Invalid Login')
            

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)
