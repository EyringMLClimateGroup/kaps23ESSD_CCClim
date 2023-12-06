#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 09:53:43 2021
highly parallellized way of regridding the ICON output and constructing npz files
Be careful, if the number of processes is not limited it will launch as many as it can
@author: arndt
"""
import numpy as np
import os
import sys
import glob
from datetime import datetime
import netCDF4 as nc4
from itertools import product
import subprocess
import time
import multiprocessing as mlp


def remap(size,nc,outname):
    """regrids from the native ICON grid to 1° regular in a subprocess so it doesnt block
        requires files specifying the new grid and the interpolation weights"""
    #this thing here can cause everything to explode if the wrong parallelism specs are used. should be ok this way
    proc = subprocess.Popen(["cdo","-P","2", "remap,newgrid_{}.txt,weightfile_{}_full.nc".format(size,size), nc,outname])    
    return

def extract_and_convert(nc):
    """Extracts the interesting variables from the ICON output and makes them into a numpy array of the desired order and shape

    Args:
        nc (string): path to a single ICON 2D output files. THe corresponding 3D name is inferred

    Returns:
        tuple of 2 np.ndarray: first entry is the properties, 2nd is the coordinates and timestamp
    """    
    properties, coords = [],[]
    file = nc4.Dataset(nc, 'r', format='NETCDF4')
    file_3d = nc4.Dataset(nc.replace("2d","3d"), 'r' ,format='NETCDF4')
    for name in variables:
        try:
            x=file.variables[name][:]
            x = np.where(x.mask,0,x)
        except KeyError:
            #the 3d variables require different ways of making them 2D
            if "tau" in name:
                x=get_integral(file_3d,name)
            elif "wap" in name:
                x = get_at_500(file_3d,name)
            else:
                x = get_at_cloud_top(file_3d,name)
            
        if name == "topmax":
            try:
                rtype=file.variables["rtype"][:]
                ignore = np.where((rtype==0)*(x>=99999))
                x[ignore]=np.nan
            except KeyError:
                pass
        #some of these values are nonsense so we replace them
        if name in ["re_drop","re_cryst","tau_sw_232","tau_sw_205"]:
            x = np.where(x>2e6,0,x)
        else:
            x = np.where(x>2e6,np.nan,x)

        properties.append(x)
        
    lat=file.variables["lat"][:]
    lon = file.variables["lon"][:]
    time = file.variables["time"][:]
    time= time.data[0]
    latlat,lonlon = np.meshgrid(lat,lon)
    latlat=latlat.transpose()
    lonlon=lonlon.transpose()
    time = np.ones(latlat.shape)*time
    coords = np.stack((latlat,lonlon,time))
    return np.vstack(properties), coords

def get_at_500(file, name):
    """returns the variable at the pressure level that is closest to 500hPa"""
    data = file.variables[name][:].squeeze()#(height,lat,lon)
    press = file.variables["pfull"][:].squeeze()
    level = np.argmin(np.abs(500-press),0)              
    at_500 = np.take_along_axis(data,np.expand_dims(level,0),0)
    return at_500

def get_at_cloud_top(file, name):
    """find the cloud top by a cumulative optical depth threshold and returns the variables in question at the corresponding level"""
    cf = file.variables["cl"][:]
    cod = file.variables["tau_sw_232"][:]
    cumsum = np.cumsum(cod,1)
    assert np.any(cumsum>0.001)
    toplayer = np.argmax(cumsum>0.2,1)[0]
    data = file.variables[name][:]
    data_attop = np.zeros((data.shape[0], data.shape[2], data.shape[3]))
    #this is probably slower than it needs to be
    for lon,lat in product(np.arange(data.shape[2]), np.arange(data.shape[3])):
        notcloudy = cf[:,toplayer[lon,lat],lon,lat]==0 
        if np.all(notcloudy) or toplayer[lon,lat]==0:
            data_attop[:,lon,lat]=np.nan
        else:
            data_attop[:,lon,lat]=data[:,toplayer[lon,lat],lon,lat]
        
    return data_attop


def nc_to_npz(nc):
    """sanity check for existing files, others are extraceted and saved as npz"""
    inname =  nc
    outname = os.path.join(outfolder_np, os.path.basename(inname).replace(".nc", ".npz").replace("netcdf", "numpy"))
    if os.path.exists(outname):
        try:
            a=np.load(outname)
        except ValueError:
            print(outname,"Not happening")
            os.remove(outname)
        p=a["properties"]
        if np.all(np.nanmean(p,(1,2))>0):
            return

    if "_2d_" in inname:
        try:
            props ,locs= extract_and_convert(inname)
            
            np.savez_compressed(outname,
                            properties = props, locations = locs)
        except KeyError as err:
            print("KeyError",err, inname)
        except TypeError as err:
            print("TypeError",err,inname)
    
    

def get_integral(file, name):
    """column 'integrated' value"""
    data = file.variables[name][:]
    integral = np.sum(data,(1))
    assert np.any(integral>0)
    return integral


def crawl(tup):
    """creates a regridded file and returns its path if its successful"""
    i,nc = tup
    outname =  os.path.join(outfolder_nc, size+"_"+os.path.basename(nc))
    to_test_if_good_file = nc4.Dataset(nc,"r",format="NETCDF4")
    if "2d" in nc:
        assert "cllvi" in to_test_if_good_file.variables,to_test_if_good_file.variables
    elif "3d" in nc:
        assert "cl" in to_test_if_good_file.variables, to_test_if_good_file.variables
    if not os.path.exists(outname):
        remap(size,nc,outname)
        return outname
    elif "2d" in outname:
        data = nc4.Dataset(outname,"r",format="NETCDF4")
        ts = data.variables["ts"][:]
        if np.nanmean(ts.data)<150:
            os.remove(outname)
            remap(size,nc,outname)
            return outname
        elif np.all(ts.mask):
            os.remove(outname)
            remap(size,nc,outname)
            return outname
        else:
            return None
    else:
        return

if __name__=="__main__":

    with open("../pvl8dagans-master/experiment_log.txt", "a+") as of:
        print("ICON_pipeline.py(" + str(datetime.today())+") : "+sys.argv[1], file=of)
    work = os.environ["WORK"]
    scratch = os.environ["SCR"]
    ICON_raw = "/mnt/lustre02/work/bd1179/experiments/icon-2.6.1_atm_amip-cldclass_R2B5_r1v0i2p1l1f1/"
    
    outfolder_nc = os.path.join(scratch, "ICON_output", "full", "netcdf")
    if not os.path.exists(outfolder_nc):
        os.makedirs(outfolder_nc)
    #variables (in order) that we want to have in the output
    variables = [ "cllvi" ,"clivi", "re_drop", "re_cryst"
                ,"pfull", "ts", "tau_sw_232", "tau_sw_205",
                "rlut","rsut","rsdt","rsutcs", "rlutcs", "clt","wap"]    
    #defines wheter a var is 2D or 3D
    variables_2d = ["cllvi","clivi","ts","rlut","rsut","rsdt","rsutcs", "rlutcs","clt"]
    variables_3d = ["re_drop","re_cryst","pfull","tau_sw_232","tau_sw_205","wap"]
    locations = ["lat","lon", "time"]
    #name because we use a cloud optical depth threshold
    outfolder_np = os.path.join(outfolder_nc,"../..","threshcodp2" ,"numpy")
    if not os.path.exists(outfolder_np):
        os.makedirs(outfolder_np)

    size = "r360x180"
    pool = mlp.Pool(20)
    #if you dont have the regridded files:
    #new_nc_files = pool.starmap(crawl, enumerate(glob.glob(ICON_raw+"*nc")))
    #if you do have the files
    new_nc_files= glob.glob(os.path.join(outfolder_nc , "*2d*nc"))
    while None in new_nc_files:#
        new_nc_files.remove(None) 
    for file in new_nc_files:
        file3 = os.path.basename(file).replace("2d","3d")
        if not os.path.exists(os.path.join(outfolder_nc,file3)):
            new_nc_files.remove(file)
    pool.map(nc_to_npz, new_nc_files)
        
