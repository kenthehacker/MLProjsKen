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
    numIts, w, e_in = costFunction.logisticReg(trainX,trainy,wInit,maxIts,eta)
    arrOfWeight.append(w)
    arrOfeIn.append(e_in)
    arrOfIts.append(numIts)

#prediction algorithm: if it is not labeled as the item then we move onto another label
#display the first four classification and print results:
for i in range(5):
    plt.imshow(testX[i])
    plt.show()
    #print the result:
    print("Actual label: "+str(testy[i]))
    print("prediction: ")






