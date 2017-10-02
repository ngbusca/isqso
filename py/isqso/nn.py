import numpy as np
from numpy import random
import fitsio

def sigmoid(z):
    return 1/(1+np.exp(-z))

def activate(z):
    return z*(z>0)
    return np.tanh(z)

def compute_cost(A2,Y,parameters,reg_factor):

    m = Y.shape[1]

    logprobs = Y*np.log(A2) + (1-Y)*np.log(1-A2) 
    reg = 0.
    for ell in range(1,parameters["L"]+1):
        reg += np.sum(parameters["W"+str(ell)]**2)

    w = np.isnan(logprobs)
    if w.sum()>0:
        stop

    cost = -logprobs.sum()/m + reg_factor*reg/2/m

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
    W = parameters["W"+str(L)]
    b = parameters["b"+str(L)]
    Z = W.dot(A)+b
    A = sigmoid(Z)
    cache["Z"+str(L)]=Z
    cache["A"+str(L)]=A


    return A,cache

def backward_propagation(parameters,cache,X,Y,reg_factor):

    m = X.shape[1]
    L = parameters["L"]

    AL = cache["A"+str(L)]
    dAL = - (Y/AL) + (1-Y)/(1-AL)
    Z = cache["Z"+str(L)]
    dZ = dAL*(Z>0)

    grads = {}
    for ell in range(L,1,-1):
        A = cache["A"+str(ell-1)]
        dW = dZ.dot(A.T)/m 
        db = dZ.sum(axis=1,keepdims=True)/m

        W = parameters["W"+str(ell)]
        dA = W.T.dot(dZ)
        Z = cache["Z"+str(ell-1)]
        dZ = dA*(Z>0)#*(1-A**2)

        ## add regularization
        dW += reg_factor*W/m

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
        b -= learning_rate*db
        pars_out["W"+str(ell)]=W
        pars_out["b"+str(ell)]=b

    return pars_out


def nn_model(X,Y,X_valid,Y_valid,nn=[10],nit = 1000,max_learning_rate=1,reg_factor=1.,parameters=None,learning_rate_init = 1.,fout=None):
    nx = X.shape[0]
    nn = [nx] + nn
    nn = nn + [Y.shape[0]]
    if parameters is None:
        parameters = initialize_parameters(nn)

    cost = []
    success = []
    cost_valid = []
    success_valid = []


    prev_cost = 1.
    learning_rate = max_learning_rate
    nlow=0
    for i in range(nit):
        A2, cache = forward_propagation(X,parameters)
        c = compute_cost(A2,Y,parameters,reg_factor)
        grads = backward_propagation(parameters,cache,X,Y,reg_factor)

        if c > prev_cost:
            learning_rate /= 1.5
        else:
            nlow +=1
            if nlow == 5 and learning_rate<max_learning_rate:
                learning_rate *= 1.5
                nlow=0
        prev_cost = c

        w = A2 > 0.5
        s=((w*Y).sum() + ((~w)*(1-Y)).sum())*1./Y.shape[1]
        success.append(s)
        cost.append(c)

        A2, _ = forward_propagation(X_valid,parameters)
        w = A2>0.5
        s_valid = ((w*Y_valid).sum()+((~w)*(1-Y_valid)).sum())*1./Y_valid.shape[1]
        success_valid.append(s_valid)
        c_valid = compute_cost(A2,Y_valid,parameters,reg_factor)
        cost_valid.append(c_valid)

        print("INFO: iteration {}, c {},cv {}, s {}, sv {}".format(i,c,c_valid,round(s,2),round(s_valid,2)))


        parameters = update_parameters(parameters,grads,learning_rate)

    return parameters,cost,cost_valid,success,success_valid

def export(fout,data,parameters,cost):
    f = fitsio.FITS(fout,"rw",clobber=True)
    f.write(data[3],extname="DATA")
    f.write(data[4],extname="TRUTH")
    f.write(np.array(data[0]),extname="THINGIDS")
    f.write([data[1],data[2]],names=["MEAN","STD"],extname="MEANSTD")
    f.write(np.array(cost),extname="COST")
    for ell in range(1,parameters["L"]+1):
        W = parameters["W"+str(ell)]
        b = parameters["b"+str(ell)]
        f.write([W,b],names=["W"+str(ell),"b"+str(ell)])
    f.close()

