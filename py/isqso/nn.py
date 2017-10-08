import numpy as np
from numpy import random
import fitsio
import copy

def sigmoid(z,epsilon=1e-7):
    return epsilon+(1-2*epsilon)/(1+np.exp(-z))

def sigmoid_prim(z,epsilon=1e-7):
    a = sigmoid(z,epsilon)
    return (1-2*epsilon)*a*(1-a)

def activate(z,kind="relu"):
    if kind=="relu":
        return z*(z>0)
    elif kind=="tanh":
        return np.tanh(z)

def activate_prim(z,kind="relu"):
    if kind == "relu":
        return z>0
    elif kind =="tanh":
        a = np.tanh(z)
        return 1-a**2

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
    dZ = dA*sigmoid_prim(Z)

    grads = {}
    for ell in range(L,1,-1):
        A = cache["A"+str(ell-1)]
        dW = dZ.dot(A.T)/m 
        db = dZ.sum(axis=1,keepdims=True)/m

        W = parameters["W"+str(ell)]

        grads["W"+str(ell)] = dW + reg_factor*W/m
        grads["b"+str(ell)] = db

        ## calculate dZ for next round
        dA = W.T.dot(dZ)
        Z = cache["Z"+str(ell-1)]
        dZ = dA*activate_prim(Z)

    ## last...
    dW = dZ.dot(X.T)/m
    db = dZ.sum(axis=1,keepdims=True)/m
    W = parameters["W1"]
    grads["W1"] = dW + reg_factor*W/m
    grads["b1"] = db

    return grads
    
def initialize_parameters(nn):
    parameters = {"L":len(nn)-1}
    for ell in range(1,len(nn)):
        W = random.randn(nn[ell],nn[ell-1])*np.sqrt(1./nn[ell-1])
        b = np.zeros((nn[ell],1))
        parameters["W"+str(ell)]=W
        parameters["b"+str(ell)]=b

    return parameters

def update_parameters(parameters,grads,learning_rate):

    for p in parameters:
        if p == "L":continue
        parameters[p] -= learning_rate*grads[p]

    return parameters


def update_parameters_momentum(parameters,grads,learning_rate,v,beta):

    for p in parameters:
        if p == "L":continue
        v[p] = beta*v[p] + (1-beta)*grads[p]
        parameters[p] -= learning_rate*v[p]

    return parameters,v


def update_parameters_adam(parameters,grads,learning_rate,v,beta1,s,beta2,t,epsilon=1e-7):

    for p in parameters:
        if p == "L":continue
        v[p] = beta1*v[p] + (1-beta1)*grads[p]
        v_corrected = v[p]/(1-beta1**t)

        s[p] = beta2*s[p] + (1-beta2)*grads[p]**2
        s_corrected = s[p]/(1-beta2**t)

        parameters[p] -= learning_rate*v_corrected/(epsilon + np.sqrt(s_corrected))

    return parameters,v,s,t+1

def split_batch(X,Y,mini_batch_size):
    m = X.shape[1]
    if mini_batch_size == m:
        return [X],[Y]

    x_mini_batch = []
    y_mini_batch = []

    a = np.random.rand(m).argsort()
    X_shuffled = X[:,a]
    Y_shuffled = Y[:,a]

    for i in range(m//mini_batch_size):
        x_mini_batch.append(X_shuffled[:,i*mini_batch_size:(i+1)*mini_batch_size])
        y_mini_batch.append(Y_shuffled[:,i*mini_batch_size:(i+1)*mini_batch_size])

    ## last
    if m % mini_batch_size != 0:
        x_mini_batch.append(X_shuffled[:,m-m % mini_batch_size:m])
        y_mini_batch.append(Y_shuffled[:,m-m % mini_batch_size:m])

    return x_mini_batch, y_mini_batch

def nn_model(X,Y,nn=[10],nit = 1000,reg_factor=1.,parameters=None,learning_rate = 1.,mini_batch_size=None,momentum=0.9,momentum2=0.999,kind="logistic",method="gd",beta1=None,beta2=None,verbose=10):

    nx = X.shape[0]
    nn = [nx] + nn
    nn = nn + [Y.shape[0]]
    if parameters is None:
        parameters = initialize_parameters(nn)

    cost = []

    nlow=0
    if mini_batch_size is None:
        mini_batch_size = X.shape[1]

    if method == "momentum" or method == "adam":
        if beta1 is None:
            beta1 = 0.9
        v = {}
        for p in parameters:
            v[p] = parameters[p]*0
    if method == "adam":
        if beta2 == None:
            beta2 = 0.999
        s = {}
        t = 1
        for p in parameters:
            s[p] = parameters[p]*0

    for i in range(nit):
        x_mini_batches, y_mini_batches = split_batch(X,Y,mini_batch_size)
        for imini,(x,y) in enumerate(zip(x_mini_batches,y_mini_batches)):
            A, cache = forward_propagation(x,parameters)
            grads = backward_propagation(parameters,cache,x,y,reg_factor,kind=kind)
            A,_ = forward_propagation(X,parameters)
            c = compute_cost(A,Y,parameters,reg_factor,kind=kind)
            if method == "gd":
                parameters = update_parameters(parameters,grads,learning_rate)
            elif method == "momentum":
                parameters,v = update_parameters_momentum(parameters,grads,learning_rate,v,beta1)
            elif method == "adam":
                parameters,v,s,t = update_parameters_adam(parameters,grads,learning_rate,v,beta1,s,beta2,t)
            cost.append(c)
            
            print("INFO: mini-batch {} iteration {}, c {}".format(imini,i,c))

    return parameters,cost

def export(fout,parameters,arq,cost,mean_data,std_data,alpha,reg_factor,nit):
    f = fitsio.FITS(fout,"rw",clobber=True)
    head={"LAYERS":parameters["L"],"ALPHA":alpha,"REG":reg_factor,"NIT":nit}
    sarq = "logistic"
    if len(arq)>0:
        sarq=""
        for s in arg:
            sarq = sarq+str(arq)+" "

    head["ARQ"] = sarq
    f.write(np.array(cost),extname="COST",header=head)
    f.write([mean_data,std_data],names=["MDATA","SDATA"],extname="MEANSTD")
    for ell in range(1,parameters["L"]+1):
        W = parameters["W"+str(ell)]
        b = parameters["b"+str(ell)]
        f.write([W,b],names=["W"+str(ell),"b"+str(ell)],extname="LAYER{}".format(ell))
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


