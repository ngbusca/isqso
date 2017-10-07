from __future__ import print_function
import fitsio
import numpy as np
from numpy import random
import glob

llmin = np.log10(3600)
llmax = np.log10(10000)
dll = 1e-3

nbins = int((llmax-llmin)/dll)
wave = 10**(llmin + np.arange(nbins)*dll)

def read_spcframe(b_spcframe,r_spcframe,pf2tid,qso_thids):
    data = []
    isqso = []
    tids = []

    hb = fitsio.FITS(b_spcframe)
    hr = fitsio.FITS(r_spcframe)
    target_bits = hb[5]["BOSS_TARGET1"][:]
    w = np.zeros(len(target_bits),dtype=bool)
    mask = [10,11,12,13,14,15,16,17,18,19,40,41,42,43,44]
    for i in mask:
        w = w | (target_bits & 2**i)
    w = w>0
    print("INFO: found {} quasars in file {}".format(w.sum(),b_spcframe))

    plate = hb[0].read_header()["PLATEID"]
    fid = hb[5]["FIBERID"][:]
    fl = np.hstack((hb[0].read(),hr[0].read()))
    iv = np.hstack((hb[1].read()*(hb[2].read()==0),hr[1].read()*(hr[2].read()==0)))
    ll = np.hstack((hb[3].read(),hr[3].read()))

    fid = fid[w]
    fl = fl[w,:]
    iv = iv[w,:]
    ll = ll[w,:]

    for i in range(fl.shape[0]):
        fl_aux = np.zeros(nbins)
        iv_aux = np.zeros(nbins)
        bins = ((ll[i]-llmin)/dll).astype(int)
        wbin = (bins>=0) & (bins<nbins) & (iv[i]>0)
        bins=bins[wbin]
        c = np.bincount(bins,weights=fl[i,wbin]*iv[i,wbin])
        fl_aux[:len(c)]=+c
        c = np.bincount(bins,weights=iv[i,wbin])
        iv_aux[:len(c)]=+c
        data.append(np.hstack((fl_aux,iv_aux)))
        if (plate,fid[i]) in pf2tid:
            t = pf2tid[(plate,fid[i])]
        else:
            t = -1
        isqso.append(t in qso_thids)
        tids.append(t)

        assert ~np.isnan(fl_aux,iv_aux).any()

    if len(data)==0:
        return
    data = np.vstack(data).T
    assert ~np.isnan(data).any()
    ## now normalize coadded fluxes
    norm = data[nbins:,:]
    w = norm==0
    norm[w] = 1.
    data[:nbins,:]/=norm
    isqso = np.array(isqso).reshape((1,len(isqso)))

    assert isqso.shape[1] == data.shape[1]

    assert ~np.isnan(data).any()

    return tids,data,isqso

def read_spall(spall):
    spall = fitsio.FITS(spall)
    plate=spall[1]["PLATE"][:]
    mjd = spall[1]["MJD"][:]
    fid = spall[1]["FIBERID"][:]
    tid = spall[1]["THING_ID"][:]
    pf2tid = {(p,f):t for p,f,t in zip(plate,fid,tid)}
    spall.close()
    return pf2tid

def read_drq(drq):

    drq = fitsio.FITS(drq)
    qso_thids = drq[1]["THING_ID"][:]
    w = qso_thids > 0
    qso_thids = qso_thids[w]
    drq.close()

    return qso_thids

def read_plates(plates,pf2tid,qso_thids,nqso=None):
    fi = open(plates,"r")
    data = []
    isqso = []
    read_plates = 0
    tids = []
    for l in fi:
        l = l.split()
        plate_dir = l[0]
        b1_spcframe = plate_dir+"/"+l[1]
        r1_spcframe = plate_dir+"/"+l[2]
        b2_spcframe = plate_dir+"/"+l[3]
        r2_spcframe = plate_dir+"/"+l[4]
        print("INFO: reading plate {}".format(plate_dir))
        res = read_spcframe(b1_spcframe,r1_spcframe,pf2tid,qso_thids)
        if res is not None:
            tid,plate_data,plate_isqso = res
            data.append(plate_data)
            isqso.append(plate_isqso)
            tids = tids + tid
        res = read_spcframe(b2_spcframe,r2_spcframe,pf2tid,qso_thids)
        if res is not None:
            tid,plate_data,plate_isqso = res
            data.append(plate_data)
            isqso.append(plate_isqso)
            tids = tids + tid

        if nqso is not None:
            if len(data)==nqso:
                break

    data = np.hstack(data)

    ## normalize
    mdata = data.mean(axis=1).reshape((-1,1))
    data -= mdata
    std = data.std(axis=1).reshape((-1,1))
    data /= std
    isqso = np.hstack(isqso)

    return tids,mdata,std,data,isqso

def export(fout,tids,mdata,std,data,isqso):
    h = fitsio.FITS(fout,"rw",clobber=True)
    h.write(np.array(tids),extname="THIDS")
    h.write([mdata,std],names=["MDATA","STD"],extname="MEANSTD")
    h.write(data,extname="DATA")
    h.write(isqso.astype(int),extname="ISQSO")
    h.close()

