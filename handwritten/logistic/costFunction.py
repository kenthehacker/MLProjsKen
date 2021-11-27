import numpy as np 
import math

#w(t+1) := w(t) - eta*grad(e_in(W))
#grad(e_in(w)) = -1/n * sum((y_n * x_n) / (1+exp(y_n * WT * x_n)))
def logisticReg(X,y,w_init,max_its,eta):
    #x is data matrix without init columns of 1's
    #y data labels
    newX = np.ones((len(X), len(X[0])+1))
    newX[: , :-1] = X
    w = w_init
    e_in = 0
    numIts = 0
    for i in range(max_its):
        numIts+=1
        #find gradient of e_in ->
        gradient = np.zeros(len(w))
        for j in range(len(X)):
            gradient = gradient + (y[j]* X[j])/(1+math.exp(y[j]*np.matmul(np.transpose(w),X[j])))
        w = w+gradient*eta*(1/len(X))
    for i in range(len(X)):
        e_in = e_in+np.log(1+math.exp(-y[i]*np.matmul(np.transpose(w),X[i])))
    e_in = e_in/len(X)  
    return numIts, w, e_in
