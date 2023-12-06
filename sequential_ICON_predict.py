#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 21 14:49:20 2021

@author: arndt
uses trained RF to predict on ESACCI
"""

import numpy as np
import torch
import timeit

import os 
import sys

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from prefetch_generator import BackgroundGenerator

import joblib
from datetime import datetime
from src.loader import ICONDataset, ICON_collate, get_avg_all
from tqdm import tqdm
import multiprocessing as mp
import signal


def days_since(df):
    start = datetime.fromisoformat("1970-01-01")
    df.time -= start
    df.time = df.time.apply(lambda x: x.days)
    return df


def timeconv(val):
    val = str(val)
    y = int(val[:4])
    m = int(val[4:6])
    d = int(val[6:8])
    f = int(val[9:])
    if m==1:
        assert d<32,val
    elif m==2:
        assert d<30, val
    elif m==3:
        assert d<32,val
    elif m==4:
        assert d<31,val
    elif m==5:
        assert d<32,val
    elif m==6:
        assert d<31,val
    elif m==7:
        assert d<32,val
    elif m==8:
        assert d<32,val
    elif m==9:
        assert d<31,val
    elif m==10:
        assert d<32,val
    elif m==11:
        assert d<31,val
    elif m==12:
        assert d<32,val
    return datetime(y, m, d, microsecond=f)
                    


def bgsave(x,fn,outpath):
    """call in subprocess to save in background"""
    pq.write_to_dataset(x,outpath)
    np.save(outpath.replace(".parquet","_fn.npy"),np.array(fn))
    return

def signal_handler(signum, handler):
    raise KeyboardInterrupt("signal")

import warnings

if __name__=="__main__":
    abs_start = timeit.default_timer()
    
    
    signal.signal(signal.SIGTERM, signal_handler)
    
    ctx = mp.get_context("forkserver")
    
    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
    
    
    clouds =["clear" , "Ci", "As", "Ac", "St", "Sc", "Cu", "Ns", "Dc"]
    properties = ["cwp","lwp", "iwp","cerl", "ceri","cod","ptop","cth",  "ctt","cee","tsurf"]
    
    # switches
    classes = 9
    pim = False
    batch_size =15 
    
    resolution = 100
        
    rel_or_abs = "sep"
    
   
    num_files = 10000
    warn = 0
    create_new=0
    
    scratch = os.environ["SCR"]
    work= os.environ["WORK"]
    model_dir = os.path.join(work, "models")
    variables = np.array([0,1,2,3,4,5,9])
    varstring = ""
    for i in variables: varstring+=str(i)
    if 0 in variables and 1 in variables:
        variables = np.hstack(([0], variables +1))
    variables_ds = np.array(properties)[variables]
    with open("experiment_log.txt", "a") as of:
        print("predict_ESACCI.py("+str(datetime.today())+") : "+sys.argv[1], file=of)
        
    folder = os.path.join(scratch, "ICON_output/threshcodp2/numpy")
    te_ds = ICONDataset(root_dir=folder,grid = "r360x180", normalizer=None, 
                      indices=None,  variables = np.hstack((variables_ds,["clt"])),
                       output_tiles=False )
    
    outpath =os.path.join(work,"frames/parquets",
            "ICONframe_threshcodp2{}_{}_{}_{}.parquet".format(resolution, 
                                                        num_files,
                                                        te_ds.grid,
                                                        varstring
                                                        ))
    
    try:
        filenames = list(np.load(outpath.replace(".parquet","_fn.npy"), allow_pickle=True))
        filenames = [[os.path.basename(y) for y in x] for x in filenames]
        for fn in filenames:
            for single in fn:
                single = os.path.join(folder,single)
                te_ds.file_paths.remove(single)
    except FileNotFoundError:
        if os.path.exists(outpath):
            os.system("rm -r {}".format(outpath))
        filenames = []
    if len(filenames)>0:
        batch_size=len(filenames[0])
    
    
    loaderstart = timeit.default_timer()
    
    
    print(te_ds.variables, len(te_ds))
    sampler = get_avg_all(te_ds, random=False)
    testloader = torch.utils.data.DataLoader(te_ds, batch_size=batch_size,
                                          sampler=sampler,collate_fn=ICON_collate,
                                          num_workers=batch_size, pin_memory=pim)
    
    model = joblib.load(os.path.join(work,"models",
                                         "viforest{}_{}_{}_{}.pkl".format(resolution, 
                                                                num_files,
                                                                rel_or_abs,
                                                                varstring)))
    
    model.n_jobs=-1
       
    gen = tqdm(enumerate(BackgroundGenerator(testloader)), total=len(testloader),file=sys.stdout)
    
    properties=list(np.array(properties)[variables])
    baseline = np.arange(-90,90,0.1)
    try:
        for j,out in gen:
            fn,i,locs = out
            fn=[os.path.basename(x) for x in np.unique(fn)]
            
            assert len(i)==len(locs),(i.shape,locs.shape)
            if j<len(testloader)-2:
                if len(filenames)>0:
                    assert len(fn)==len(filenames[0]),(fn,filenames[0])
            filenames.append(fn)
            interesting = torch.ones(len(i),dtype=bool)
            """
            if regional:
                #interesting *= locs[:,1]>90
                #interesting *= locs[:,1]<135
                interesting *= locs[:,0]>5
                interesting *= locs[:,0]<15
            """
            interesting *= torch.any(i>0,1)
            #limit cod to reasonable values
            interesting *= i[:,5]<200
            #make sure there is a ptop
            interesting *= i[:,6]>50
            if torch.sum(interesting)==0:
                continue
            
            assert not torch.any(torch.isnan(locs)) and not torch.any(torch.isinf(locs))
            x_np = i.numpy()[interesting][:]
            
            t=model.predict(x_np[:,:len(varstring)+1])
            
            locations = locs.numpy()[interesting][:len(x_np)]
           
            inout= pd.DataFrame(np.hstack((x_np,locations,t)),columns=properties+["clt","lat","lon","time"]+clouds)
            
            inout.time = inout.time.apply(timeconv)
            inout = days_since(inout)
            inout = pa.Table.from_pandas(inout)
            """
            #commenting this out because MPI is an asshole
            try:
                proc.join()
            except Exception:
                pass
            proc =ctx.Process(target=bgsave,args=(inout,filenames, outpath))
            proc.start()
            """
            bgsave(inout,filenames, outpath)
            gen.set_description("Dataloader, prediction and saving")
            if timeit.default_timer()-abs_start>7.5*3600:
                raise KeyboardInterrupt
    except KeyboardInterrupt:
        print("nothing to see here")
        #proc.join()
    print("done")
        
