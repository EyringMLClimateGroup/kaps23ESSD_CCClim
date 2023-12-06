#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 21 11:38:44 2021
uses the raw ESA Cloud_cci L3U nc files to make npz files that are more useful here
@author: arndt
"""

import netCDF4 as nc4
import numpy as np
import sys
from datetime import datetime
import os 
import glob
import multiprocessing as mlp
import traceback

coordinates = ['lat', 'lon']
properties = ["lwp", "iwp","cer_liq", "cer_ice","cot","ctp","cth",  "ctt","cee","stemp_cloudy"
               ,"cfc"]
properties_daily_asc = ["cwp_asc",  "cer_asc", "cot_asc", "ctp_asc", "cth_asc", 
                    "ctt_asc", "cot_desc", "stemp_asc", "cph_asc"]
properties_daily_desc = ["cwp_desc",  "cer_desc", "cot_desc", "ctp_desc", "cth_desc", 
                    "ctt_desc", "cot_asc", "stemp_desc", "cph_desc"]

def read_daily(nc_file):
    """Extracts the ascending and descinding orbits and rescales them and puts them in the order the RF expects them to be in

    Args:
        nc_file (string): path of file to load

    Returns:
        list of two np.ndarray: array for each orbit. each contains the 2D fields of all properties,
                                 spatiotemporal locations and radiative fluxes and cloud cover
    """    
    data = nc4.Dataset(nc_file, "r", format="NETCDF4")
    fname = os.path.basename(nc_file)
    #cloud cover is in a different file for some reason
    cltds = nc4.Dataset(os.path.join(os.environ["SCR"],
                    "ESACCI/MASK",fname.replace("PRODUCTS","MASKTYPE")),"r")
    clt_asc = cltds.variables["cmask_asc"]
    clt_desc = cltds.variables["cmask_desc"]
    del cltds
    #radiative fluxes are also in another file
    radds = nc4.Dataset(os.path.join(os.environ["SCR"],"ESACCI/RAD",
                        fname.replace("CLD","RAD")),"r")
    toa_asc = np.vstack([radds.variables[ x][...] for x in ["toa_lwup_asc","toa_swup_asc","toa_swdn_asc",
                                    "toa_swup_clr_asc", "toa_lwup_clr_asc" ]])
    toa_desc = np.vstack([radds.variables[x][...] for x in ["toa_lwup_desc","toa_swup_desc","toa_swdn_desc",
                                    "toa_swup_clr_desc","toa_lwup_clr_desc"] ])
    del radds
    #other physical variables
    f_properties_asc = np.vstack([data.variables[name][:] for name in properties_daily_asc])
    #replace nonsense with missing
    f_properties_asc = np.where(f_properties_asc<0, np.nan, f_properties_asc)
    #rescale height to meters
    f_properties_asc[4]*=1000
    #make 2D fiels for the locations
    lat = data.variables["lat"][:]
    lon = data.variables["lon"][:]
    timepoint = data.variables["time"][:]
    latlat,lonlon = np.meshgrid(lat,lon)
    latlat=latlat.transpose()[np.newaxis,...]
    lonlon=lonlon.transpose()[np.newaxis,...]
    timepoint = np.ones(latlat.shape)*timepoint
    assert np.all(np.unique(latlat)==np.unique(lat))
    assert np.all(np.unique(lonlon)==np.unique(lon))
    
    f_loc = np.vstack((latlat,lonlon,timepoint))
    #phase descision
    isice = f_properties_asc[-1]==2
    isliquid = f_properties_asc[-1]==1
    #phase water path
    lwp=np.copy(f_properties_asc[0])
    iwp=np.copy(f_properties_asc[0])
    lwp[isice]=0
    iwp[isliquid]=0
    lwp[~(isice+isliquid)]/=2
    iwp[~(isice+isliquid)]/=2
    test = ~np.isnan(f_properties_asc[0])
    #make sure ice and water add up to total everywhere
    assert np.all(lwp[test]+iwp[test]==f_properties_asc[0][test])
    #phase effective radius
    ceri=np.copy(f_properties_asc[1])
    cerl=np.copy(f_properties_asc[1])
    ceri[isliquid]=0
    cerl[isice]=0
    ceri[~(isice+isliquid)]/=2
    cerl[~(isice+isliquid)]/=2
    test = ~np.isnan(f_properties_asc[1])
    assert np.all(ceri[test]+cerl[test]==f_properties_asc[1][test])
    #put everything in the expected order
    out1=[f_properties_asc[0],lwp,iwp, cerl,ceri]
    out1.extend(f_properties_asc[2:-1])
    out1.extend(toa_asc)
    out1.extend(clt_asc)
    out1.extend(f_loc)
    out1=np.stack(out1)
    #same thing for the descending orbits
    f_properties_desc = np.vstack([data.variables[name][:] for name in properties_daily_desc])
    
    f_properties_desc = np.where(f_properties_desc<0, np.nan, f_properties_desc)

    f_properties_desc[4]*=1000
    #phase descision
    isice = f_properties_desc[-1]==2
    isliquid = f_properties_desc[-1]==1
    #phase water path
    lwp=np.copy(f_properties_desc[0])
    iwp=np.copy(f_properties_desc[0])
    lwp[isice]=0
    iwp[isliquid]=0
    lwp[~(isice+isliquid)]/=2
    iwp[~(isice+isliquid)]/=2
    test = ~np.isnan(f_properties_desc[0])
    assert np.all(lwp[test]+iwp[test]==f_properties_desc[0][test])
    #phase effective radius
    ceri=np.copy(f_properties_desc[1])
    cerl=np.copy(f_properties_desc[1])
    ceri[isliquid]=0
    cerl[isice]=0
    ceri[~(isice+isliquid)]/=2
    cerl[~(isice+isliquid)]/=2
    test = ~np.isnan(f_properties_desc[1])
    assert np.all(ceri[test]+cerl[test]==f_properties_desc[1][test])
    out2=[f_properties_desc[0],lwp,iwp, cerl,ceri]
    out2.extend(f_properties_desc[2:-1])
    out2.extend(toa_desc)
    out2.extend(clt_desc)
    timepoint = data.variables["time"][:]
    timepoint = np.ones(latlat.shape)*timepoint
    f_loc = np.vstack((latlat,lonlon,timepoint))
    out2.extend(f_loc)
    
    out2=np.stack(out2)
    return [out1, out2]
    

    
if __name__=="__main__":        
    work = os.environ["WORK"]
    scratch = os.environ["SCR"]
    folder = os.path.join(scratch, "ESACCI/CLD",)
    outfolder = os.path.join(folder,"../npz_daily")
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    files = glob.glob(os.path.join(folder, "*.nc"))

    with open("experiment_log.txt", "a+") as of:
        print("extract_ESACCI.py("+str(datetime.today())+") : "+sys.argv[1], file=of)
        
    assert len(files)>0, folder
    def convert(file):
        if "L3U" in file:
            read = read_daily
            time="daily"
        else:
            raise ValueError("file makes no sense?")
        if time=="daily": 
            if not os.path.exists(os.path.join(outfolder,
                                            "1_"+os.path.basename(file).replace(".nc", ".npz"))):
                #only processes if it doesnt already exist
                print("{} doesnt exist".format(os.path.join(outfolder,
                                                
                                            "1_"+os.path.basename(file).replace(".nc", ".npz"))))
                p=read(file)
                assert len(p[0].shape)>2, p[0].shape
                np.savez_compressed(os.path.join(outfolder, "0_"+os.path.basename(file).replace(".nc",".npz")),
                                    props= p[0][:-3].data.astype(np.float32),
                                    locs = p[0][-3:].data.astype(np.float32))
                
                np.savez_compressed(os.path.join(outfolder, "1_"+os.path.basename(file).replace(".nc",".npz")
                                    ), props=p[1][:-3].data.astype(np.float32),
                                    locs = p[1][-3:].data.astype(np.float32))
                os.remove(file)
            
        #sanity check and removal of input files    
        fname = os.path.basename(file)
        if os.path.exists(os.path.join(outfolder,
                                            "1_"+os.path.basename(file).replace(".nc", ".npz"))):
            try:
                p=np.load(os.path.join(outfolder,
                                        "1_"+os.path.basename(file).replace(".nc", ".npz")))["props"]
                assert np.all(p.shape==(17,3600,7200)),"shape not good"
            except Exception:
                traceback.print_exc()
                return
            try:
                os.remove(os.path.join(os.environ["SCR"],
                                "ESACCI/MASK",fname.replace("PRODUCTS","MASKTYPE")))
            except Exception:
                traceback.print_exc()
            try:
                os.remove(os.path.join(os.environ["SCR"],
                                "ESACCI/RAD",fname.replace("CLD","RAD")))
            except Exception:
                traceback.print_exc()
            try:
                os.remove(file)
            except Exception:
                traceback.print_exc()


                
    p=mlp.Pool(20)
    print(len(files))
    p.map(convert, files)

