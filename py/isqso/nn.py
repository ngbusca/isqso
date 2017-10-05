import numpy as np
from numpy import random
import fitsio
import copy

def sigmoid(z):
    return 1/(1+np.exp(-z))

def activate(z):
    return z*(z>0)
    #return np.tanh(z)

def compute_cost(AL,Y,parameters,reg_factor,kind="logistic"):
    if kind=="logistic":
        return cost_logistic(AL,Y,parameters,reg_factor)
    elif kind=="chi2":
        return cost_chi2(AL,Y,parameters,reg_factor)

def cost_logistic(AL,Y,parameters,reg_factor):
    m = Y.shape[1]
    
    logprobs = np.log(AL[0,Y[0]]).sum() + np.log(1-AL[0,~Y[0]]).sum()
    reg = 0.
    for ell in range(1,parameters["L"]+1):
        reg += np.sum(parameters["W"+str(ell)]**2)

    cost = -logprobs/m + reg_factor*reg/2/m

    return cost

def cost_chi2(AL,Y,parameters,reg_factor):
    m = Y.shape[1]
    chi2 = np.sum((AL-Y)**2)/2/m
    reg = 0
    for ell in range(1,parameters["L"]+1):
        reg += np.sum(parameters["W"+str(ell)]**2)

    chi2 += reg_factor*reg/2/m

    return chi2

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

def backward_propagation(parameters,cache,X,Y,reg_factor,kind="logistic"):

    m = X.shape[1]
    L = parameters["L"]

    A = cache["A"+str(L)]
    if kind=="logistic":
        dA = - (Y/A) + (1-Y)/(1-A)
    elif kind=="chi2":
        dA = A-Y
    Z = cache["Z"+str(L)]
    dZ = dA*A*(1-A)

    grads = {}
    for ell in range(L,1,-1):
        A = cache["A"+str(ell-1)]
        dW = dZ.dot(A.T)/m 
        db = dZ.sum(axis=1,keepdims=True)/m

        W = parameters["W"+str(ell)]

        ## add regularization
        dW += reg_factor*W/m

        grads["dW"+str(ell)] = dW
        grads["db"+str(ell)] = db

        ## calculate dZ for next round
        dA = W.T.dot(dZ)
        Z = cache["Z"+str(ell-1)]
        dZ = dA*(Z>0)
        #dZ = dA*(1-A**2)

    ## last...
    dW = dZ.dot(X.T)/m
    db = dZ.sum(axis=1,keepdims=True)/m
    W = parameters["W1"]
    grads["dW1"] = dW + reg_factor*W/m
    grads["db1"] = db

    return grads
    
def initialize_parameters(nn):
    parameters = {"L":len(nn)-1}
    for ell in range(1,len(nn)):
        W = random.randn(nn[ell],nn[ell-1])*np.sqrt(1./nn[ell-1])
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


def nn_model(X,Y,nn=[10],nit = 1000,max_learning_rate=None,reg_factor=1.,parameters=None,learning_rate_init = 1.,fout=None,num_mini_batches=1,momentum=0.9,momentum2=0.999,kind="logistic",verbose=1):

    if max_learning_rate is None:
        max_learning_rate = learning_rate_init

    nx = X.shape[0]
    nn = [nx] + nn
    nn = nn + [Y.shape[0]]
    if parameters is None:
        parameters = initialize_parameters(nn)

    cost = []
    success = []

    learning_rate = learning_rate_init
    nlow=0
    x_mini_batches = np.array_split(X,num_mini_batches,axis=1)
    y_mini_batches = np.array_split(Y,num_mini_batches,axis=1)
    grad0 = {}
    S = {}
    prev_cost = 1
    nprev = 0
    max_learning_rate=learning_rate
    for i in range(nit):
        for x,y in zip(x_mini_batches,y_mini_batches):
            A2, cache = forward_propagation(x,parameters)
            grads = backward_propagation(parameters,cache,x,y,reg_factor,kind=kind)
            if len(grad0) == 0:
                grad0 = copy.deepcopy(grads)
                for p in grad0:
                    S[p]=grad0[p]*0.
            else:
                for p in grads:
                    grad0[p] = momentum*grad0[p] + (1-momentum)*grads[p]
                    S[p] = momentum2*S[p]/(1-momentum2**i)+momentum2*grads[p]**2
                    if p=="db1":
                        print grads[p]
                    if i>10000:
                        grad0[p]/=np.sqrt(S[p])

            parameters = update_parameters(parameters,grad0,learning_rate)


        A2,_ = forward_propagation(X,parameters)
        c = compute_cost(A2,Y,parameters,reg_factor,kind=kind)
        w = A2 > 0.5
        s=((w*Y).sum() + ((~w)*(1-Y)).sum())*1./Y.shape[1]
        success.append(s)
        cost.append(c)
        if c > prev_cost:
            learning_rate /= 1.1
            nprev = 0
        else:
            nprev+=1
            if nprev>=10 and learning_rate<max_learning_rate:
                learning_rate*=1.1
                nprev=0

        prev_cost = c
        if i%verbose==0:
            print("INFO: iteration {}, c {}, s {}".format(i,c,s))

    return parameters,cost,success

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

def test_backprop(X,Y,parameters,epsilon=1e-3,kind="logistic"):
    A,cache = forward_propagation(X,parameters)
    grads = backward_propagation(parameters,cache,X,Y,0.,kind=kind)
    for p in parameters:
        if p =="L":continue
        pars = copy.deepcopy(parameters)
        W = parameters[p]
        i = random.randint(W.shape[0])
        j = random.randint(W.shape[1])
        pars[p][i,j] = W[i,j]+epsilon
        A,_ = forward_propagation(X,pars)
        cp = compute_cost(A,Y,pars,0,kind=kind)
        pars[p][i,j] = W[i,j]-epsilon
        A,_ = forward_propagation(X,pars)
        cm = compute_cost(A,Y,pars,0,kind=kind)
        print("numerical d{}[{},{}]: {}".format(p,i,j,(cp-cm)/(2*epsilon)))
        print("backpropa d{}[{},{}]: {}".format(p,i,j,grads["d"+p][i,j]))


    print("Now again but regularization")
    reg_factor=1.
    grads = backward_propagation(parameters,cache,X,Y,reg_factor,kind=kind)
    for p in parameters:
        if p =="L":continue
        pars = copy.deepcopy(parameters)
        W = parameters[p]
        i = random.randint(W.shape[0])
        j = random.randint(W.shape[1])
        pars[p][i,j] = W[i,j]+epsilon
        A,_ = forward_propagation(X,pars)
        cp = compute_cost(A,Y,pars,reg_factor,kind=kind)
        pars[p][i,j] = W[i,j]-epsilon
        A,_ = forward_propagation(X,pars)
        cm = compute_cost(A,Y,pars,reg_factor,kind=kind)
        print("numerical d{}[{},{}]: {}".format(p,i,j,(cp-cm)/(2*epsilon)))
        print("backpropa d{}[{},{}]: {}".format(p,i,j,grads["d"+p][i,j]))


