'''
A gradient descent algorithm using one-vs-all classification
from scratch only using numpy
'''
import math
import numpy as np
import matplotlib.pyplot as plt
import costFunction
from tensorflow.keras.datasets import mnist

#load all of the data ->
(trainX, trainy), (testX, testy) = mnist.load_data()

#train the data ->
eta = 0.1
arrOfWeight = []
arrOfIts = []
arrOfeIn = []
maxIts = 1000

#one vs all 0-9
for i in range(10):
    wInit = np.zeros(len(trainX[0]))
    numIts, w, e_in = costFunction.logisticReg(trainX,i,wInit,maxIts,eta)
    arrOfWeight.append(w)
    arrOfeIn.append(e_in)
    arrOfIts.append(numIts)

#prediction algorithm: if it is not labeled as the item then we move onto another label
#display the first four classification and print results:








'''
numpyTrain = clevelandTrainData.to_numpy()
yTrain = numpyTrain[:,len(numpyTrain[0])-1]

for i in range(len(yTrain)):
    if yTrain[i] == 0:
        yTrain[i]= -1
#print(yTrain)
xTrain = np.delete(numpyTrain, len(numpyTrain[0])-1, axis=1)

numpyTest = clevelandTestData.to_numpy()
yTest = numpyTest[:,len(numpyTest[0])-1]
for i in range(len(yTest)):
    if yTest[i] == 0:
        yTest[i]= -1
#print(yTest)
xTest = np.delete(numpyTest, len(numpyTest[0])-1, axis=1)

def find_cross_entropy(X, y, w):
    x = np.ones((len(X), len(X[0]) + 1))
    x[: , :-1] = X
    X = x
    e_in = 0
    for i in range(len(X)):
        e_in = e_in+np.log(1+math.exp(-y[i]*np.matmul(np.transpose(w),X[i])))
    e_in = e_in/len(X)
    return e_in

#constraints for logistical 
eta = .00001
wInit = np.zeros(len(numpyTrain[0]))
maxIts = 10000
firstItTime = time.time()
t, w, e_in = logistic_reg(xTrain, yTrain, wInit, maxIts, eta)
resTime = time.time()
print("10^4 its:")
print("cross entropy train: "+str(e_in))
print("cross entropy test: " + str(find_cross_entropy(xTest, yTest, w)))
print(w)
print(t)
print("clock time diff: "+ str(resTime-firstItTime))
print("done")
test_err = find_test_errorA(w, xTest, yTest)
print("binary test: " + str(test_err))
train_err = find_test_errorA(w, xTrain, yTrain)
print("binary train: " + str(train_err))


'''