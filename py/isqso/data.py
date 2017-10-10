from __future__ import print_function
import fitsio
import numpy as np
from numpy import random
import glob

llmin = np.log10(3600)
llmax = np.log10(10000)
dll = 1e-3

nbins = int((llmax-llmin)/dll)
nmasked_max = nbins/10
wave = 10**(llmin + np.arange(nbins)*dll)

def read_spcframe(b_spcframe,r_spcframe,pf2tid):
    data = []
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
        if (plate,fid[i]) in pf2tid:
            t = pf2tid[(plate,fid[i])]
        else:
            print("DEBUG: ({},{}) not found in spall".format(plate,fid[i]))
            continue

        fl_aux = np.zeros(nbins)
        iv_aux = np.zeros(nbins)
        bins = ((ll[i]-llmin)/dll).astype(int)
        wbin = (bins>=0) & (bins<nbins) & (iv[i]>0)
        bins=bins[wbin]
        c = np.bincount(bins,weights=fl[i,wbin]*iv[i,wbin])
        fl_aux[:len(c)]=+c
        c = np.bincount(bins,weights=iv[i,wbin])
        iv_aux[:len(c)]=+c
        nmasked = (iv_aux==0).sum()
        if nmasked >= nmasked_max :
            print("INFO: skipping specrum {} with too many masked pixels {}".format(t,nmasked))
            continue
        data.append(np.hstack((fl_aux,iv_aux)))
        tids.append(t)

        assert ~np.isnan(fl_aux,iv_aux).any()

    if len(data)==0:
        return
    data = np.vstack(data).T
    assert ~np.isnan(data).any()
    ## now normalize coadded fluxes
    norm = data[nbins:,:]*1.
    w = norm==0
    norm[w] = 1.
    data[:nbins,:]/=norm

    assert ~np.isnan(data).any()

    return tids,data

def read_spall(spall):
    spall = fitsio.FITS(spall)
    plate=spall[1]["PLATE"][:]
    mjd = spall[1]["MJD"][:]
    fid = spall[1]["FIBERID"][:]
    tid = spall[1]["THING_ID"][:]
    specprim=spall[1]["SPECPRIMARY"][:]
    pf2tid = {(p,f):t for p,f,t,s in zip(plate,fid,tid,specprim) if s==1}
    spall.close()
    return pf2tid

def read_drq_superset(drq_sup,high_z = 2.1):
    ##from https://arxiv.org/pdf/1311.4870.pdf
    ##      only return targets with z_conf_person == 3:
    ##      class person: 1 (Star), 3 (QSO), 4 (Galaxy), 30 (QSO_BAL)
    ##      my class_person: 1 (Star), 3 (QSO and z < hz), 4 (Galaxy), 5 (QSO and z>=hz), 30 (QSO_BAL)

    drq = fitsio.FITS(drq_sup)
    qso_thids = drq[1]["THING_ID"][:]
    class_person = drq[1]["CLASS_PERSON"][:]
    z_conf = drq[1]["Z_CONF_PERSON"][:]
    z = drq[1]["Z_VI"][:]

    ## select objects with good classification
    w = (qso_thids > 0) & (z_conf==3)
    qso_thids = qso_thids[w]
    class_person = class_person[w]
    my_class_person = class_person*1
    z = z[w]

    ## STARS
    w = class_person == 1
    my_class_person[w] = 0

    ## GALAXIES
    w = class_person == 4
    my_class_person[w] = 1

    ## QSO_LOWZ, include BAL
    w = ((class_person==3) | (class_person == 30)) & (z<high_z)
    my_class_person[w] = 2

    ## QSO_HIGHZ, include BAL
    w = ((class_person==3) | (class_person == 30)) & (z>=high_z)
    my_class_person[w] = 3

    drq_classes = ["STAR","GALAXY","QSO_LOWZ","QSO_HIGHZ","BAL"]
    Y = np.zeros((len(class_person),len(drq_classes)))
    for i in range(Y.shape[0]):
        Y[i,my_class_person[i]]=1

    ## add BAL flag
    w = class_person == 30
    Y[w,drq_classes.index("BAL")]=1

    ## 
    target_class = {tid:y for tid,y in zip(qso_thids,Y)}
    
    drq.close()

    return target_class

def read_plates(plates,pf2tid,nplates=None):
    fi = open(plates,"r")
    data = []
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

        ## read spectro 1
        res = read_spcframe(b1_spcframe,r1_spcframe,pf2tid)
        if res is not None:
            plate_tid,plate_data = res
            data.append(plate_data)
            tids = tids + plate_tid

        ## read spectro 2
        res = read_spcframe(b2_spcframe,r2_spcframe,pf2tid)
        if res is not None:
            plate_tid,plate_data = res
            data.append(plate_data)
            tids = tids + plate_tid

        if nplates is not None:
            if len(data)//2==nplates:
                break

    data = np.hstack(data)

    return tids,data

def export(fout,tids,data):
    h = fitsio.FITS(fout,"rw",clobber=True)
    h.write(data,extname="DATA")
    tids = np.array(tids)
    h.write([tids],names=["TARGETID"],extname="METADATA")
    h.close()

