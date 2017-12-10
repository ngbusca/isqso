from os.path import dirname

import numpy as np
import fitsio
from isqso import data


def read_data():

    path = dirname(data.__file__)

    tc = data.read_drq_superset(path+"/../../data/Superset_DR12Q.fits")

    h=fitsio.FITS(path+"/../../data/data_training.fits.gz")
    Xtmp = h[0].read().T
    tids = h[1]["TARGETID"][:]
    Xtmp = Xtmp[:,:443]
    w = np.in1d(tids,tc.keys())
    tids = tids[w]
    X_train = np.zeros((w.sum(), Xtmp.shape[1]))
    X_train[:] = Xtmp[w,:]
    del Xtmp
    mdata = X_train.mean(axis=0)
    sdata = X_train.std(axis=0)
    X_train-=mdata
    X_train/=sdata

    Ytmp = np.array([tc[t] for t in tids])
    w=(Ytmp[:,3]==1) & (Ytmp[:,4]==1)
    Ytmp[w,3]=0
    Y_train=np.zeros(Ytmp.shape)
    Y_train[:]=Ytmp
    del Ytmp
    h.close()
    
    h=fitsio.FITS(path+"/../../data/data_validation.fits.gz")
    Xtmp = h[0].read()[:443].T
    tids = h[1]["TARGETID"][:]
    w = np.in1d(tids,tc.keys())
    tids=tids[w]
    X_valid = np.zeros((w.sum(), Xtmp.shape[1]))
    X_valid[:] = Xtmp[w,:]
    del Xtmp
    X_valid-=mdata
    X_valid/=sdata
    h.close()
    Ytmp = np.array([tc[t] for t in tids])
    w=(Ytmp[:,3]==1) & (Ytmp[:,4]==1)
    Ytmp[w,3]=0
    Y_valid=np.zeros(Ytmp.shape)
    Y_valid[:]=Ytmp
    del Ytmp

    return X_train, Y_train, X_valid, Y_valid, mdata, sdata
