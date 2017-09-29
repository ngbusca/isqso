import numpy as np
from numpy import random

def sigmoid(z):
    return 1/(1+np.exp(-z))

def activate(z):
    '''
    RELU
    '''
    return z*(z>0)

def compute_cost(A2,Y):

    m = Y.shape[1]

    logprobs = Y*np.log(A2) + (1-Y)*np.log(1-A2)
    w = np.isnan(logprobs)
    if w.sum()>0:
        stop

    cost = -logprobs.sum()/m

    return cost

def forward_propagation(X,parameters):

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = W1.dot(X)+b1
    A1 = activate(Z1)
    
    Z2 = W2.dot(A1)+b2
    A2 = sigmoid(Z2)

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}

    return A2,cache

def backward_propagation(parameters,cache,X,Y):

    m = X.shape[1]

    W1 = parameters["W1"]
    W2 = parameters["W2"]

    A1 = cache["A1"]
    A2 = cache["A2"]

    dZ2= A2-Y
    dW2 = dZ2.dot(A1.T)/m
    db2 = dZ2.sum(axis=1,keepdims=True)/m

    dZ1 = W2.T.dot(dZ2)*(A2>0)#*(1-A2**2)
    dW1 = dZ1.dot(X.T)/m
    db1 = dZ1.sum(axis=1,keepdims=True)

    grads = {"dW1":dW1,"db1":db1,"dW2":dW2,"db2":db2}
    return grads
    
def initialize_parameters(nx,nn):
    W1 = random.randn(nn,nx)*0.01
    b1 = np.zeros((nn,1))

    W2 = random.randn(1,nn)*0.01
    b2 = np.zeros((1,1))

    parameters = {"W1":W1,"b1":b1,"b2":b2,"W2":W2}

    return parameters

def update_parameters(parameters,grads,learning_rate=1.):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    W1 -= learning_rate*dW1
    b1 -= learning_rate*db1
    W2 -= learning_rate*dW2
    b2 -= learning_rate*db2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    
    return parameters


def nn_model(X,Y,nn=10,nit = 1000,learning_rate=1.):
    nx = X.shape[0]
    parameters = initialize_parameters(nx,nn)

    cost_vs_iter = []

    for i in range(nit):
        A2, cache = forward_propagation(X,parameters)
        cost = compute_cost(A2,Y)
        cost_vs_iter.append(cost)

        grads = backward_propagation(parameters,cache,X,Y)

        parameters = update_parameters(parameters,grads,learning_rate)

        if i %10 == 0:
            print("INFO: iteration {}, cost {}".format(i,cost))


    return parameters,cost_vs_iter


