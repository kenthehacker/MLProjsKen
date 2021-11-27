import numpy as np 
import math

#w(t+1) := w(t) - eta*grad(e_in(W))
#grad(e_in(w)) = -1/n * sum((y_n * x_n) / (1+exp(y_n * WT * x_n)))
def logisticReg(X,y,w_init,max_its,eta):
    #x is data matrix without init columns of 1's
    #y data labels
    newX = np.ones((len(X), len(X[0])+1))
    newX[: , :-1] = X
    wVec = w_init
    e_in = 0
    numIts = 0
    for i in range(max_its):
        numIts+=1
        #find gradient of e_in ->
        gradient = [] 
        for x in range(10):
            gradient.append(np.zeros(len(wVec[x])))
        
        for j in range(len(X)):
            gradient[y[j]] = gradient[y[j]] + (y[j]* X[j])/(1+math.exp(y[j]*np.matmul(np.transpose(wVec[y[j]]),X[j])))
        
        for n in range(10):
            wVec[n] = wVec[n]+gradient[y[j]]*eta*(1/len(X))
    '''
    for i in range(len(X)):
        e_in = e_in+np.log(1+math.exp(-y[i]*np.matmul(np.transpose(wVec),X[i])))
    e_in = e_in/len(X)/10
    '''
    return numIts, wVec, e_in


'''

TODO: the weight matrix aint right: we're throwing in 6000 weight values, we only want 10!
TODO: ALSO we want to cherry pick which weight we want to be using!!!
TODO: EIN is currently 0 we need to fix that
'''

