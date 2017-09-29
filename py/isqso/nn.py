import numpy as np
from numpy import random

def sigmoid(z):
    return 1/(1+np.exp(-z))

def activate(z):
    return np.tanh(z)

def compute_cost(A2,Y):

    m = Y.shape[1]

    logprobs = Y*np.log(A2) + (1-Y)*np.log(1-A2)
    w = np.isnan(logprobs)
    if w.sum()>0:
        stop

    cost = -logprobs.sum()/m

    return cost

def forward_propagation(X,parameters):

    L = parameters["L"]
    cache = {}

    A = X
    for ell in range(1,L):
        W = parameters["W"+str(ell)]
        b = parameters["b"+str(ell)]

        Z = W.dot(A)+b
        A = activate(Z)

        cache["Z"+str(ell)]=Z
        cache["A"+str(ell)]=A
    
    ## last iteration with sigmoid
    W = parameters["W"+str(ell+1)]
    b = parameters["b"+str(ell+1)]
    Z = W.dot(A)+b
    A = sigmoid(Z)
    cache["Z"+str(ell+1)]=Z
    cache["A"+str(ell+1)]=A


    return A,cache

def backward_propagation(parameters,cache,X,Y):

    m = X.shape[1]
    L = parameters["L"]

    A2 = cache["A"+str(L)]
    dZ = A2 - Y

    grads = {}
    for ell in range(L,1,-1):
        A1 = cache["A"+str(ell-1)]
        dW = dZ.dot(A1.T)/m
        db = dZ.sum(axis=1,keepdims=True)/m

        W = parameters["W"+str(ell)]
        dZ = W.T.dot(dZ)*(1-A1**2)
        A2 = A1

        grads["dW"+str(ell)] = dW
        grads["db"+str(ell)] = db

    ## last...
    dW = dZ.dot(X.T)/m
    db = dZ.sum(axis=1,keepdims=True)/m
    grads["dW1"] = dW
    grads["db1"] = db

    return grads
    
def initialize_parameters(nn):
    parameters = {"L":len(nn)-1}
    for ell in range(1,len(nn)):
        W = random.randn(nn[ell],nn[ell-1])*0.01
        b = np.zeros((nn[ell],1))
        parameters["W"+str(ell)]=W
        parameters["b"+str(ell)]=b

    return parameters

def update_parameters(parameters,grads,learning_rate=1.):

    L = parameters["L"]
    pars_out = {"L":L}
    for ell in range(1,L+1):
        W = parameters["W"+str(ell)]
        b = parameters["b"+str(ell)]
        dW = grads["dW"+str(ell)]
        db = grads["db"+str(ell)]
        W -= learning_rate*dW
        db -= learning_rate*db
        pars_out["W"+str(ell)]=W
        pars_out["b"+str(ell)]=b

    return pars_out


def nn_model(X,Y,nn=[10],nit = 1000,learning_rate=1.):
    nx = X.shape[0]
    nn = [nx] + nn
    parameters = initialize_parameters(nn)

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


