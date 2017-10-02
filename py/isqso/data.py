from __future__ import print_function
import fitsio
import numpy as np
from numpy import random
import glob

llmin = np.log10(3600)
llmax = np.log10(10000)
dll = 1e-3
pmf2tid = None
qso_thids = None

nbins = int((llmax-llmin)/dll)
wave = 10**(llmin + np.arange(nbins)*dll)

def read_plate(indir,spall,drq):
    global pmf2tid,qso_thids
    if pmf2tid is None:
        spall = fitsio.FITS(spall)
        plate=spall[1]["PLATE"][:]
        mjd = spall[1]["MJD"][:]
        fid = spall[1]["FIBERID"][:]
        tid = spall[1]["THING_ID"][:]
        pmf2tid = {(p,m,f):t for p,m,f,t in zip(plate,mjd,fid,tid)}
        spall.close()

    if qso_thids is None :
        drq = fitsio.FITS(drq)
        qso_thids = drq[1]["THING_ID"][:]
        w = qso_thids > 0
        qso_thids = qso_thids[w]
        drq.close()

    fi = glob.glob(indir+"/spCFrame-b1*")
    if len(fi)==0:
        print("INFO: no spCFrames found in {}".format(indir))
        return
    ## choose a random exposure
    a = random.uniform(size=len(fi))
    infile = fi[a.argsort()[0]]
    ## this is the data: rebinned fluxes and ivars
    ## for red and blue
    data = []
    isqso = []
    tids = []

    for spectro in ['1','2']:
        try:
            thefile = infile.replace("b1","b{}".format(spectro))
            h=fitsio.FITS(thefile)
        except:
            continue
        target_bits = h[5]["BOSS_TARGET1"][:]
        w = np.zeros(len(target_bits),dtype=bool)
        mask = [10,11,12,13,14,15,16,17,18,19,40,41,42,43,44]
        for i in mask:
            w = w | (target_bits & 2**i)
        w = w>0
        print("INFO: found {} quasars in file {}".format(w.sum(),thefile))
        h.close()

        try:
            hb=fitsio.FITS(infile.replace("b1","b{}".format(spectro)))
            hr=fitsio.FITS(infile.replace("b1","r{}".format(spectro)))
        except:
            continue

        plate = hb[0].read_header()["PLATEID"]
        mjd = hb[0].read_header()["MJD"]
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
            wbin = (bins>=0) & (bins<nbins)
            bins=bins[wbin]
            c = np.bincount(bins,weights=fl[i,wbin]*iv[i,wbin])
            fl_aux[:len(c)]=+c
            c = np.bincount(bins,weights=iv[i,wbin])
            iv_aux[:len(c)]=+c
            data.append(np.hstack((fl_aux,iv_aux)))
            if (plate,mjd,fid[i]) in pmf2tid:
                t = pmf2tid[(plate,mjd,fid[i])]
            else:
                t = -1
            isqso.append(t in qso_thids)
            tids.append(t)

    if len(data)==0:
        return
    data = np.vstack(data).T
    ## now normalize fluxes
    norm = data[nbins:,:]
    w = norm==0
    norm[w] = 1.
    data[:nbins,:]/=norm
    isqso = np.array(isqso).reshape((1,len(isqso)))

    assert isqso.shape[1] == data.shape[1]

    return tids,data,isqso


def read_plates(plate_dir,spall,drq,nplates=None):
    fi = glob.glob(plate_dir+"/????/")
    if nplates is not None:
        ifi = random.uniform(size=len(fi))
        fi = [fi[i] for i in ifi.argsort()]
    data = []
    isqso = []
    read_plates = 0
    tids = []
    for d in fi:
        print("INFO: reading plate {}".format(d))
        res = read_plate(d,spall,drq)
        if res is None:
            continue
        read_plates += 1
        tid,plate_data,plate_isqso = res
        data.append(plate_data)
        isqso.append(plate_isqso)
        tids = tids + tid
        if nplates is not None:
            if read_plates == nplates:
                break

    data = np.hstack(data)
    mdata = data.mean(axis=1).reshape((-1,1))
    data -= mdata
    std = data.std(axis=1).reshape((-1,1))
    data /= std
    isqso = np.hstack(isqso)

    return tid,mdata,std,data,isqso
