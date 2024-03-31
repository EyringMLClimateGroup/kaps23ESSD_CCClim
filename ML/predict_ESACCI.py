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
import glob

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from prefetch_generator import BackgroundGenerator

import joblib
from datetime import datetime
from tqdm import tqdm
import traceback
import multiprocessing as mp
import signal

def bgsave(x,fn,outpath):
    """call in subprocess to save in background"""
    pq.write_to_dataset(x,outpath)
    print([len(x) for x in fn])
    np.save(outpath.replace(".parquet","_fn.npy"),np.array(fn))
    return

def signal_handler(signum, handler):
    raise KeyboardInterrupt("signal")

import warnings

if __name__=="__main__":
    abs_start = timeit.default_timer()
    
    #import here because i dont need to import when i launch subprocesses
    from src.loader import ESACCIDataset, ESACCI_collate
    from src.utils import Normalizer, get_chunky_sampler, get_avg_tiles, get_avg_all
    from src.utils import get_dataset_statistics, get_ds_labels_stats
    signal.signal(signal.SIGTERM, signal_handler)
    
    ctx = mp.get_context("forkserver")
    
    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
    
    
    clouds =["clear" , "Ci", "As", "Ac", "St", "Sc", "Cu", "Ns", "Dc"]
    properties = ["lwp", "iwp","cerl", "ceri","cod","ptop","htop",  "ttop","cee","tsurf",
                   "rlut","rsut","rsdt","rsutcs","rlutcs","clt"]
    """
    dtypes = {"clear": "float32" , "Ci", "As", "Ac", "St", "Sc", "Cu", "Ns", "Dc","lwp",
              "iwp","cer_liq", "cer_ice","cot","ctp","cth",  "ctt","cee","stemp_cloudy"
                   ,"cfc"}
    """
    # switches
    classes = 9
    
    
    pim = False
    batch_size = 5
    
    resolution = 100
    
    rel_or_abs = ""
    
    regional = 0
    if regional:
        reg_code = "new"
    else:
        reg_code =""
    hyper = 0
    num_files = 10000
    warn = 0
    create_new=0
    
    
    work= os.environ["WORK"]
    #work = "/mnt/work/"
    model_dir = os.path.join(work, "models")
    variables = np.array([0,1,2,3,4,5,9])
    variables_ds = np.array([0,1,2,3,4,5,6,10])
    ancillary = np.array([11,12,14,15,16]) #rlut,rsut,rstcs,rlutcs,clt
    assert np.max(ancillary)<=len(properties),(ancillary,len(properties))
    varstring = ""
    for i in variables: varstring+=str(i)
    time = "de" #mn monthlymean, dn dailymean, de daily instnataneous
    print(sys.argv)
    with open("experiment_log.txt", "a") as of:
        print("predict_ESACCI.py("+str(datetime.today())+") : "+str(sys.argv), file=of)
    try:
        year = sys.argv[2]
        reg_code += year
    except IndexError:
        year=None
    folder = os.path.join(os.environ["SCR"], "ESACCI/npz_daily")
    if "d" in time:
        assert "daily" in folder
    elif "m" in time:
        assert "monthly" in folder
    if year is not None:
        sorted_f = glob.glob(os.path.join(folder,"*.npz"))
        sorted_f.sort()
        years = np.array([os.path.basename(x)[2:6] for x in sorted_f])
        assert year in years,(year,np.unique(years)[:5])
        idx_choice = np.where(years==year)[0]
        print(len(idx_choice))
    else:
        idx_choice=None

    te_ds = ESACCIDataset(root_dir=folder, normalizer=None, 
                      indices=idx_choice,  variables = variables_ds,ancillary = ancillary,
                      chunksize = 10 , output_tiles=False, time_type = time,
                      subsample=None, overlap=False)#int(3e6))
    #if this says notemp instead of nooverlap i am using data not meant for relase
    outpath =os.path.join(os.environ["SCR"],"ESACCI/parquets/cropped",  
                                 "ESACCIf_{}frame{}_{}_{}_{}.parquet".format(time[0]+time[-1]+reg_code,
                                                                    resolution, 
                                                        num_files,
                                                        te_ds.chunksize,
                                                        varstring
                                                        ))
    
    _path,_name = os.path.split(outpath)
    if os.path.exists(os.path.join(_path,"new"+_name)):
        raise Exception("dont need to do this")

    if not os.path.exists(_path):
        os.makedirs(_path)
    try:
        
        filenames = list(np.load(outpath.replace(".parquet","_fn.npy"), allow_pickle=True))
        filenames = [[os.path.basename(y) for y in x] for x in filenames]
        print(filenames[:10], outpath.replace(".parquet","_fn.npy") )
        if not os.path.exists(outpath):
            filenames = []  
            os.remove(outpath.replace(".parquet","_fn.npy"))
            raise FileNotFoundError
        for fn in filenames:
            for single in fn:
                single = os.path.join(folder,single)
                te_ds.file_paths.remove(single)
    except FileNotFoundError:
        if os.path.exists(outpath):
            os.system("rm -r {}".format(outpath))
        filenames = []
    loaderstart = timeit.default_timer()
    
    
    print(te_ds.variables, len(te_ds))
    sampler = get_avg_all(te_ds, random=False)
    testloader = torch.utils.data.DataLoader(te_ds, batch_size=batch_size,
                                          sampler=sampler,collate_fn=ESACCI_collate,
                                          num_workers=batch_size*2, pin_memory=0)
    
    model = joblib.load(os.path.join(work,"models",
                                         "viforest{}_{}_{}_{}.pkl".format(resolution, 
                                                                num_files,
                                                                rel_or_abs,
                                                                varstring)))
   
    print("viforest{}_{}_{}_{}.pkl".format(resolution,num_files,rel_or_abs, varstring))                                     
    model.n_jobs=27
    print(type(model))
    
    
    gen = tqdm(enumerate(BackgroundGenerator(testloader)), total=len(testloader),
                file=sys.stdout, leave=True, position=0)
    
    props=list(np.array(properties)[variables])
    baseline = np.arange(-90,90,0.1)
    include = []
    maxsample=0
    total=0
    try:
        for j,out in gen:
            fn,i,locs,anc = out
            print(i.shape, locs.shape)
            fn=[os.path.basename(x) for x in np.unique(fn)]
            assert len(i)==len(locs),(i.shape,locs.shape)
            if j<len(testloader)-2:
                if len(filenames)>0:
                    assert len(fn)==len(filenames[0]),(fn,filenames[0])
            filenames.append(fn)
            interesting = torch.ones(len(i),dtype=bool)
            
            for day in locs[0].round().unique():
                if torch.any(i[locs[:,-1] == day].std(dim = 0)<=0):
                    print(day," is shit, i.e. only one measurement here")
                    interesting *= ~(locs[:,-1] == day)
            bad = ((i[:,-1]<10)|(i[:,-2]<10))
            bad += torch.any(torch.isnan(i),1)
            interesting[bad]=0
            assert not torch.any(torch.isnan(locs)) and not torch.any(torch.isinf(locs))
            locs = locs.numpy()[interesting]

            """
            if regional:
                #interesting *= locs[:,1]>90
                #interesting *= locs[:,1]<135
                interesting *= locs[:,0]>5
                interesting *= locs[:,0]<15
            """
            include.append(interesting.sum())
            maxsample = max(maxsample,len(i))
            total = total+len(i)# maxsample*(j+1)
            if torch.sum(interesting)==0:
                continue
            
            
            x_np = i.numpy()[interesting][:]
            print("before",anc[:,-1].mean())
            if anc is not None:
                anc = anc.numpy()[interesting][:]
            t=model.predict(x_np)
            print("after",anc[:,-1].mean(),(1-t[:,0]).mean())
            
            locations = locs[:len(x_np)] 
            
            cols = ["twp"]+props+["lat","lon","time"]+clouds
            if anc is not None:
                stack=np.hstack((x_np,locations,t,anc))
                cols += [properties[x-1] for x in ancillary]
            else:
                stack = np.hstack((x_np,locations,t))
            inout= pd.DataFrame(stack,columns=cols)
            #inout = inout.round({"lat":0, "lon":0, "time":0})
            #inout_gpby = inout.groupby(["lat","lon","time"]).mean()
            #inout = inout_gpby.reset_index()
            inout = pa.Table.from_pandas(inout)
            
            try:
                proc.join()
            except Exception:
                pass
            proc =ctx.Process(target=bgsave,args=(inout,filenames, outpath))
            proc.start()
            gen.set_description("Dataloader, prediction and saving Percentage: {}, lost: {}".format(np.sum(include)/total,total-np.sum(include)))
            gen.update(1)
            if timeit.default_timer()-abs_start>22*3600:
            
                raise KeyboardInterrupt
    except KeyboardInterrupt:
        print("breaking")
        proc.join()
    print("done")
        
