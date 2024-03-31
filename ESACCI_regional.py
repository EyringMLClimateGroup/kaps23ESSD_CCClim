#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 11:20:03 2021

@author: arndt
"""
import time
from distributed import Client, progress, as_completed
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300
from matplotlib.colors import ListedColormap, Normalize
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy.crs as ccrs
import sys
from distributed.client import futures_of
from datetime import datetime,timedelta
import os
import traceback
import dask.array as da
import glob
import dask.dataframe as dd
from scipy.stats import spearmanr,pearsonr
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import global_land_mask as globe





def FI(arr,val):
    """
    Finds the first instance of arr being equal to val
    Parameters
    ----------
    arr : np array
        array 
    val : float
        class

    Returns
    -------
    int
        first time arr is val

    """
    return np.argmin(np.abs(arr-val))


def hist_heights(x,gp,bindict):
    """
    given a preset array of bins, find the corresponding histogram values

    Parameters
    ----------
    x : np array
        partition of dataframe in this case
    gp : str
        name of column to look at
    bindict : dict
         bins to use

    Returns
    -------
    a : np.ndarray
        histogram

    """
    a,_= np.histogram(x,bins=bindict[gp])
    return a


def customize(ax):
    """
    
    makes a good looking world map plot
    Parameters
    ----------
    ax : matplotlib axis
        axis to customize

    Returns
    -------
    None.

    """
    ax.set_extent([-180,180,-90,90], crs=ccrs.PlateCarree())
    ax.set_xticks(range(-160, 200, 40), crs=ccrs.PlateCarree())
    ax.set_yticks(range(-80, 120, 40), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.tick_params(labelsize=14)
    ax.coastlines()

def map_remote(df,gp,bindict):
    """
    in the dask cluster, compute histograms of the columns of a dask dataframe

    Parameters
    ----------
    df : dask.dataframae.DataFrame
        a dataframe
    gp : str
        name of column.
    bindict : dict
        dict of numpy arrays

    Returns
    -------
    out : np array
        the histograms
    gp : same as what goes in
    bindict : same as what goes in

    """
    df=df.loc[:,gp]
    if gp=="time":
        df-=df.min()
        with open("timemax.txt","w+") as file:
            ma=df.max().compute()
            mi=df.min().compute()
            print(ma, mi, file=file)
            
            bindict[gp]=np.arange(mi,ma)
    out = df.map_partitions(hist_heights,gp,bindict)
    if gp=="time":
        with open("timemax.txt","a+") as file:
            print(bindict["time"], file=file)
    return out,gp,bindict
    

def get_max_fraction_simple(df):
    """
    finds dominatn cloud type per lat/lon grid cell
    i.e. type that most often has the largest fraction

    Parameters
    ----------
    df : dask.dataframe.DataFrame
        cloud type fraction lat/lon/time

    Returns
    -------
    Sum : dataframe
        index of the cloud type where the fraction per lat/lon is largest

    """
    Sum = df.groupby(["lat", "lon"]).sum(split_out=int(df_label.npartitions/10))
    Sum=Sum.loc[:,ctnames]
    Sum.columns=range(len(Sum.columns))
    Sum = Sum.idxmax(axis=1).to_frame("mc")
    
    
    return Sum


if __name__=="__main__":
    
    scratch = os.environ["SCR"]
    work = os.environ["WORK"]
    cMap= ListedColormap(['gold', 'green', 'blue','red', "cyan",
                          "lime", "black", "magenta"])#aquamarine
    print(sys.argv)

    if len(sys.argv)>=4:
        path = sys.argv[3]
    else:
        path = os.path.join(scratch, "ESACCI/parquets/cropped")
    #"""
    tried = 0
    while True:
        try:
            SCHEDULER_FILE = glob.glob(os.path.join(scratch,"scheduler*.json"))[0]
            
            if SCHEDULER_FILE and os.path.isfile(SCHEDULER_FILE):
                client = Client(scheduler_file=SCHEDULER_FILE)
            break
        except IndexError:
            if tried:
                raise Exception("no scheduler up")
            else:
                tried=0
            time.sleep(10)
    #"""
    #client=Client()     
    print(client.dashboard_link)
    plt.close("all")
    ctnames = [ "Ci", "As", "Ac", "St", "Sc", "Cu", "Ns", "Dc"]
    axlabel_dict={"clear": "Clear sky fraction","Ci":"Cirrus/Cirrostratus fraction",
                  "As":"Altostratus fraction", "Ac":"Altocumulus fraction",
                  "St":"Stratus fraction", "Sc": "Stratocumulus fraction",
                  "Cu": "Cumulus fraction", "Ns": "Nimbostratus fraction",
                  "Dc": "Deep convection fraction","clear_p": "Predicted clear sky fraction",
                  "Ci_p":"Predicted Cirrus/Cirrostratus fraction",
                  "As_p":"Predicted Altostratus fraction", 
                  "Ac_p":"Predicted Altocumulus fraction",
                  "St_p":"Predicted Stratus fraction", "Sc_p": "Predicted Stratocumulus fraction",
                  "Cu_p": "Predicted Cumulus fraction", "Ns_p": "Predicted Nimbostratus fraction",
                  "Dc_p": "Predicted Deep convection fraction",
                  "cwp":"total water path", "twp":"total water path",
                  "lwp": "liquid water path", "iwp":"ice water path",
                  "cod": "cloud optical depth", "tsurf": "surface temperature",
                  "stemp_cloudy": "surface temperature", "cee": "emissivity",
                  "ptop": "cloud top pressure", "htop": "cloud top height",
                  "ttop": "cloud top temperature", "cerl": "liquid droplet radius",
                  "ceri": "ice particle radius","ctp":"cloud top pressure"}
    
    #client.wait_for_workers(1)
    try:
        sptemp = ["time","lat", "lon"]
        #clear sky fraction
        clear=dd.read_parquet( os.path.join(path, 
                                            sys.argv[1]),
                              columns =sptemp+["clear"]   )
        #cloud fractions
        df=dd.read_parquet( os.path.join(path, sys.argv[1]),
                           columns =sptemp+ctnames)
            
        
        
        times=0
    except Exception:
        traceback.print_exc()
        sptemp = [ "lat", "lon"] 
        clear=dd.read_parquet( os.path.join(path, 
                                            sys.argv[1]), 
                              columns =sptemp+["clear"]   )
        df=dd.read_parquet( os.path.join(path, 
                                         sys.argv[1]), columns =sptemp+ctnames)
    

    name=sys.argv[1]

    if False:
        print("MAKING REGIONAL SUBSET")    
        latmin=-90
        latmax =90
        lonmin = -180
        lonmax =180
        latfrac = (np.abs(latmax-latmin)/180)
        lonfrac= np.abs(lonmax-lonmin)/360
        df = df[df.lat>latmin]
        df = df[df.lat<latmax]
        df = df[df.lon>lonmin]
        df = df[df.lon<lonmax]    
        true = df.map_partitions(lambda x: globe.is_ocean(x.lat,x.lon))
        df = df.loc[true,:]
        name=name.replace(".parquet","_{}.parquet".format("Ocean"))
    else:
        latmin=-90
        latmax=90
        lonmin=-180
        lonmax=180
        lonfrac=1
        latfrac=1
    print("df loaded", df.npartitions)
    #samplefrac=1
    if "ICON" not in name and df.npartitions>1:
        df=df.sample(frac=0.001)
        print("sampled")
    rounddict={key:{"lat": 0, "lon": 0,"time":0}[key] for key in sptemp}
    
    #possibly also plot the clear sky fraction
    #clear = clear.sample(frac=samplefrac, replace=False, random_state=22345)
    clear = clear.loc[:,["lat","lon","clear"]]
    clear = clear.groupby(["lat","lon"]).mean().compute()
   
    cf=clear.reset_index(level=["lat","lon"])
    
    fig=plt.figure(figsize=(12,7))
    ax=fig.add_subplot(1,1,1,projection=ccrs.PlateCarree())
    
    #im goint to need this to manually create a grid
    un_lat = np.unique(cf.lat)
    un_lon = np.unique(cf.lon)
    
    colors=cf.loc[:,"clear"].values
    #gg=np.empty((len(step_lat),len(step_lon)))*np.nan
    gg=np.ones((len(un_lat),len(un_lon)))*-1
    #i have the locations of the cells and theri values in order, 
    #but now i populate a grid with that
    for x,y,z in zip(cf.lat,cf.lon,colors):
        i = np.argwhere(un_lat==x)
        j= np.argwhere(un_lon==y)
        gg[i,j] = z
    #i dont care bout points wehre there is no cloud
    gg=np.where(gg<0,np.nan,gg)
    lonlon,latlat = np.meshgrid(un_lon,un_lat)
    cloud_fraction = ax.pcolormesh(lonlon,latlat,gg,cmap="gist_stern",shading="auto",
                                transform=ccrs.PlateCarree(),vmin=0,vmax=1)
    cbar=fig.colorbar(cloud_fraction, orientation="horizontal",fraction=0.12,pad=0.12,
                            shrink=0.8)
    cbar.ax.tick_params(labelsize=20)
    ax.coastlines()
    ax.set_xticks([-160,-120,-80,-40,0,40,80,120,160])
    ax.set_yticks([-80,-40,0,40,80])
    ax.set_xticklabels([-160,-120,-80,-40,0,40,80,120,160], fontsize=16)
    ax.set_yticklabels([-80,-40,0,40,80], fontsize=16)
    ax.set_title("Clear sky fraction",fontsize=15)
    fig.tight_layout()
    fig.savefig(os.path.join(work,"stats",name.replace(".parquet","_csf.pdf")))
    

    del clear
    
    # bin on a two degree grid instead of one, making comparison with cloudsat better
    twodegreegrid=1
    if not twodegreegrid:
        df=df.round(rounddict)
    else:
        df=(df/2).round(rounddict)*2
        name="2"+name
    
    #compute the sum to normalize
    s=df.loc[:,ctnames].sum(axis="columns")
    df[[x+"n" for x in ctnames]] = df.loc[:,ctnames].truediv(s,axis="index")
    df=df.drop(ctnames,axis=1)
    df.columns=sptemp+ctnames
    
    if "2010" in name:
        #exclude july 2010 because that has wrong data
        start = datetime.fromisoformat("1970-01-01")
        start_july = datetime.fromisoformat("2010-07-01")
        end_july = datetime.fromisoformat("2010-07-31")
        ex_july = (df.time>(end_july-start).days)|( df.time<(start_july-start).days)
        df=df[ex_july]
    elif "1994" in name:
        #exclude last quarter of 1994 because weird/missing data
        start = datetime.fromisoformat("1970-01-01")
        start_january = datetime.fromisoformat("1994-01-01")
        ex_260 = df.time- (start_january-start).days < 260
        df=df[ex_260]
    elif "1986" in name:
        #exclude first january
        start = datetime.fromisoformat("1970-01-01")
        start_january = datetime.fromisoformat("1986-01-01")
        ex_11 = df.time!=(start_january-start).days
        df=df[ex_11]
    
    if len(sys.argv)>2:
        #analyse seasonal subset of the data
        seas = sys.argv[2]
        if seas in ["DJF","SON","JJA","MAM"]:
            name=name.replace(".parquet","_{}.parquet".format(seas))
            
            df.time=dd.to_datetime(df.time,unit="D")
            months={"DJF":(12,1,2),"MAM":(3,4,5),"JJA":(6,7,8),"SON":(9,10,11)}
            one,two,thr = months[seas]
            inseason = ((df.time.dt.month==one)|
                    (df.time.dt.month==two)|
                    (df.time.dt.month==thr))
            df=df.loc[inseason]
            df.time=df.time.astype(int)/1e9/3600/24
            #print(df.time.mean().compute(),df.time.max().compute())
        elif seas=="All":
            pass
        else:
            raise ValueError("Season must be a season code or All")
        
    df_label=client.persist(df)
    progress(df_label)
    del df # I dont think that does anything
    
    allfig,allaxes=plt.subplots(4,2,figsize=(9*lonfrac,8),dpi=300,
                                subplot_kw=dict(projection=ccrs.PlateCarree())   ,
                                )    
    allaxes=allaxes.flatten()               
    cloudmax=0
    pmeshdict={}
    
    if "2ESA" in name:
        #comparison matrices
        compare=np.zeros((2,5,8))#((CCClim,ISCCP),(mean,90thmean, diffCS,90thdiff,corr),(cloudtypes))
        # if we are on two degree grid, load CloudSat data and compare
        for pos,cname in enumerate(ctnames):
            CS=np.load(os.path.join(work,"frames","CS_{}.npy".format(cname)))
            if pos==0:
                CS_norm=np.zeros(CS[2].shape)
            CS_norm+=CS[2]
        #also load ISCCP data and compare to that
        for pos,cname in enumerate(ctnames):
            
            _H=np.load(os.path.join(work,"frames","ISCCP-H_{}.npy".format(cname)))[:,1:,:]
            assert np.all(~np.isnan(_H)),cname
            _H = np.roll(_H,axis=2,shift=179)
            _H[1] = np.where(_H[1]>180,_H[1]-360,_H[1])
            H = (_H[:,:-1:2,::2]/2) + (_H[:,1::2,1::2]/2)
            H=H[...,:-1]#removes a trailing column of 0°, which i already have
            
            if pos==0:
                H_norm=np.zeros(H[2].shape)
            H_norm+=H[2]
            
    corrmean=0
    ESAnorm=df_label.loc[:,["lat","lon"]+ctnames]
    print((ESAnorm.lat.nunique()*ESAnorm.lon.nunique()).compute())
    ESAnorm=ESAnorm.groupby(["lat","lon"]).count()#.sum()# # fraction of each class per cell
    
    ESAnorm=ESAnorm.compute()#.sum(1).compute() # total cloud amount per cell
    #ESAnorm = ESAnorm.to_frame("cloud_amount")
    #assert np.all(ESAnorm.values[:,0,np.newaxis] == ESAnorm.values)
    print(ESAnorm.head())
    print(len(ESAnorm))
    for pos,cname in enumerate(ctnames):
        #compute maximum cloud fraction for color bar scale
        cloud = df_label.loc[:,["lat","lon",cname]]
        cloud = cloud.groupby(["lat","lon"]).sum()
        cloud=cloud.compute()
        
        cloud.loc[:,cname]=cloud.loc[:,cname]/ESAnorm.iloc[:,0]
        cloudmax=max(cloud.loc[:,cname].max(),cloudmax)
    print("cloudmax",cloudmax)
    #individual cloud densities        
    for pos,cname in enumerate(ctnames):
        cloud = df_label.loc[:,["lat","lon",cname]]
        cloud = cloud.groupby(["lat","lon"]).sum()
        #going to plot the total amount of each cloud type per cell
        #relative to the total amount of clout in each cell
        cloud=cloud.compute()
        
        cloud.loc[:,cname]=cloud.loc[:,cname]/ESAnorm.iloc[:,0]#relative
        cloud.reset_index(level=["lat","lon"],inplace=True)
        #im goint to need this to manually create a grid
        un_lat = np.unique(cloud.lat)
        un_lon = np.unique(cloud.lon)
        colors=cloud.loc[:,cname].values
        #gg=np.empty((len(step_lat),len(step_lon)))*np.nan
        gg=np.ones((len(un_lat),len(un_lon)))*-1
        #i have the locations of the cells and theri values in order, 
        #but now i populate a grid with that
        for x,y,z in zip(cloud.lat,cloud.lon,colors):
            i = np.argwhere(un_lat==x)
            j= np.argwhere(un_lon==y)
            gg[i,j] = z
        #i dont care bout points wehre there is no cloud
        gg=np.where(gg<0,np.nan,gg)
        lonlon,latlat = np.meshgrid(un_lon,un_lat)
        pmeshdict[cname]=np.stack((gg,lonlon,latlat))
        fig=plt.figure()
        ax=fig.add_subplot(1,1,1,projection=ccrs.PlateCarree())
        if "2ESA" in name:
            #so that seems to be formatted like a multiindex df, not a grid
            CS=np.load(os.path.join(work,"frames","CS_{}.npy".format(cname)))
            CS_lon = CS[0]
            CS_lat = CS[1]
            _H=np.load(os.path.join(work,"frames","ISCCP-H_{}.npy".format(cname)))[:,1:,:]
            #this nonsense puts if from -180 to 180 and on the 2°Grid
            _H = np.roll(_H,axis=2,shift=179)
            _H[1] = np.where(_H[1]>180,_H[1]-360,_H[1])
            H = (_H[:,:-1:2,::2]/2) + (_H[:,1::2,1::2]/2)
            H=H[:,:,:-1]
            H_lon = H[1]
            H_lat = H[0]
            CS = CS[2]
            H=H[2]
            
            
            figdiff=plt.figure(figsize=(12,7))
            axdiff=figdiff.add_subplot(1,1,1,projection=ccrs.PlateCarree())
            
            figHdiff=plt.figure(figsize=(12,7))
            axHdiff=figHdiff.add_subplot(1,1,1,projection=ccrs.PlateCarree())
            
            
            #for colorbar
            minlat=max(np.min(CS_lat),np.min(latlat),np.min(H_lat))
            maxlat=min(np.max(CS_lat),np.max(latlat),np.max(H_lat))
            minlon=max(np.min(CS_lon),np.min(lonlon),np.min(H_lon))
            maxlon=min(np.max(CS_lon),np.max(lonlon),np.max(H_lon))
            
            difflon,difflat=np.meshgrid(np.arange(minlon,maxlon,2),np.arange(minlat,maxlat,2))
            
            #first we find the are of the date that is within the grid cell
            diffCS = CS[FI(CS_lat[:,0],minlat):FI(CS_lat[:,0],maxlat),
                        FI(CS_lon[0],minlon):FI(CS_lon[0],maxlon)]
            diffH = H[FI(H_lat[:,0],minlat):FI(H_lat[:,0],maxlat),
                        FI(H_lon[0],minlon):FI(H_lon[0],maxlon)]
            #and the the corresponding cloud amount
            assert np.all(H_norm.shape==H.shape),(H_norm.shape, H.shape)
            CS_norm_temp = CS_norm[FI(CS_lat[:,0],minlat):FI(CS_lat[:,0],maxlat),
                        FI(CS_lon[0],minlon):FI(CS_lon[0],maxlon)]
            H_norm_temp = H_norm[FI(H_lat[:,0],minlat):FI(H_lat[:,0],maxlat),
                        FI(H_lon[0],minlon):FI(H_lon[0],maxlon)]
            #and we use that to normalize
            diffCS/=CS_norm_temp
            diffCS=np.where(np.isnan(diffCS),0,diffCS)
            assert difflon.shape==diffCS.shape, (difflon.shape, diffCS.shape)
            diffH/=H_norm_temp
            diffH=np.where(np.isnan(diffH),0,diffH)
            assert difflon.shape==diffH.shape, (difflon.shape, diffH.shape)
            assert difflat.shape==diffH.shape, (difflat.shape, diffH.shape)
            diffESA=gg[FI(latlat[:,0],minlat):FI(latlat[:,0],maxlat),
                        FI(lonlon[0],minlon):FI(lonlon[0],maxlon)]
            

            print(np.sum(diffCS==0),diffCS.shape)
            print(np.sum(diffESA==0),diffESA.shape)
            print(np.sum(diffH==0),diffH.shape)
            diff = diffESA-diffCS
            #find how large the deviation is at the important points
            p90th = np.percentile(diffESA[~np.isnan(diffESA)],90)
            qtdiff=np.where(diffESA>p90th,diff,np.nan)
            assert np.any(~np.isnan(qtdiff)),p90th
            np.save("diffESA"+cname,diffESA)
            ESAtemp = diffESA.flatten()
            CStemp = diffCS.flatten()
            notnan = ~(np.isnan(ESAtemp)+np.isnan(CStemp))
            corr = pearsonr(ESAtemp[notnan],CStemp[notnan])[0]
            corrmean+=corr
            print("{}, ESA: {:.3}, CS: {:.3}, 90Abs: {:.3}".format(cname,np.nanmean(diffESA),
                                                          np.nanmean(diffCS),
                                                          np.nanmean(diffESA[diffESA>p90th])))
            print("{}: Corr: {:.3}, diff {:.3}, 90thdiff: {:.3}".format(cname,corr,
                                                                        np.nanmean(diff),
                                                                        np.nanmean(qtdiff)))
            compare [0,:,pos]=[np.mean(diffESA),np.mean(diffESA[diffESA>p90th]), np.mean(diff),
                                np.nanmean(qtdiff), corr]
            diffplot=axdiff.pcolormesh(difflon, difflat,diff,cmap="gist_stern",shading="auto",
                                transform=ccrs.PlateCarree(),
                                vmin=-max(np.max(diffCS),np.max(diffESA)),
                                vmax = max(np.max(diffCS),np.max(diffESA)))
            
            cbar=figdiff.colorbar(diffplot, orientation="vertical",fraction=0.06,pad=0.02,
                                shrink=0.75)
            cbar.ax.tick_params(labelsize=16)
            customize(axdiff)
            axdiff.set_title("Mean {} difference".format(cname),fontsize=21)
            figdiff.tight_layout()
            #figdiff.savefig(os.path.join(work,"stats",name.replace(".parquet","_diff{}.eps".format(cname))), bbox_inches="tight")
            figdiff.savefig(os.path.join(work,"stats",name.replace(".parquet","_diff{}.pdf".format(cname))), bbox_inches="tight")
            
            
            diff = diffH-diffCS
            p90th = np.percentile(diffH,90)
            qtdiff=np.where(diffH>p90th,diff,np.nan)
            np.save("diffH"+cname,diffH)
            corr = pearsonr(diffH.flatten(),diffCS.flatten())[0]
            corrmean+=corr
            print("{}, ISCCP: {:.3}, CS: {:.3}, 90Abs: {:.3}".format(cname,np.nanmean(diffH),
                                                          np.nanmean(diffCS),
                                                          np.nanmean(diffH[diffH>p90th])))
            print("{}: Corr: {:.3}, diff {:.3}, 90thdiff: {:.3}".format(cname,corr,
                                                                        np.nanmean(diff),
                                                                        np.nanmean(qtdiff)))
            compare [1,:,pos]=[np.mean(diffH),np.mean(diffH[diffH>p90th]), np.mean(diff),
                                np.nanmean(qtdiff), corr]
            diffHplot=axHdiff.pcolormesh(difflon, difflat,diff,cmap="gist_stern",shading="auto",
                                transform=ccrs.PlateCarree(),
                                vmin=-max(np.max(diffCS),np.max(diffH)),
                                vmax = max(np.max(diffCS),np.max(diffH)))
            
            cbar=figHdiff.colorbar(diffHplot, orientation="vertical",fraction=0.06,pad=0.02,
                                shrink=0.75)
            cbar.ax.tick_params(labelsize=16)
            customize(axHdiff)
            axHdiff.set_title("Mean {} difference".format(cname),fontsize=21)
            figHdiff.tight_layout()
            #figdiff.savefig(os.path.join(work,"stats",name.replace(".parquet","_diff{}.eps".format(cname))), bbox_inches="tight")
            figHdiff.savefig(os.path.join(work,"stats",name.replace(".parquet","_diffH{}.pdf".format(cname))), bbox_inches="tight")
        if cname =="Dc" and twodegreegrid:
            np.save(os.path.join(work,"frames","compare_matrix.npy"),compare)
            raise KeyboardInterrupt("stop here")
        meshplot=ax.pcolormesh(lonlon, latlat,gg,cmap="gist_stern",shading="auto",
                            transform=ccrs.PlateCarree(),vmin=0,vmax=cloudmax)
        
        cbar=fig.colorbar(meshplot, orientation="vertical",fraction=0.08,pad=0.02,
                            )
        cbar.ax.tick_params(labelsize=12)
        customize(ax)
        ax.set_title("Mean {} fraction".format(cname),fontsize=21)
        xmin, xmax = ax.get_xbound()
        ymin, ymax = ax.get_ybound()
         
        
        y2x_ratio = (ymax-ymin)/(xmax-xmin)
        fig.set_figheight(fig.get_figwidth()* y2x_ratio)
        #fig.savefig(os.path.join(work,"stats",name.replace(".parquet","_{}.svg".format(cname))))
        fig.savefig(os.path.join(work,"stats",name.replace(".parquet","_{}.pdf".format(cname))))
        
        del cloud
    allone = sum([x for x,y,z in pmeshdict.values()])
    print("should sum to 1",allone)
    #assert np.allclose(allone[~np.isnan(allone)],1),np.max(np.abs(allone[~np.isnan(allone)]-1))
    for pos,cname in enumerate(ctnames):    
        gg,lonlon,latlat = pmeshdict[cname]
        aax=allaxes[pos]
        meshplot=aax.pcolormesh(lonlon,latlat,gg,cmap="gist_stern",shading="auto",
                                transform=ccrs.PlateCarree(),vmin=0,vmax=cloudmax)
        
        if pos>5:
            mid = np.round((df_label.lon.max()+df_label.lon.min())/2,0)
            end = float(df_label.lon.max().compute())
            tick1 = np.round(((end-mid)*7/18),0)
            tick2 = np.round((end-mid)*14/18,0)
            if np.abs(mid-end)>40:
                aax.set_xticks([mid-tick2,mid-tick1,mid,mid+tick1,mid+tick2])
                aax.set_xticklabels([mid-tick2,mid-tick1,mid,tick1+mid,tick2+mid], fontsize=13)
            else:
                aax.set_xticks([mid-tick2,mid,mid+tick2])
                aax.set_xticklabels([mid-tick2,mid,tick2+mid], fontsize=13)
        if pos%2==0:
            mid = np.round((df_label.lat.max()+df_label.lat.min())/2,0)
            end = df_label.lat.max()
            tick1 = np.round((end-mid)*4/9,0)
            tick2 = np.round((end-mid)*8/9,0)
            if np.abs(end-mid)>40:
                aax.set_yticks([mid-tick2,mid-tick1,mid,tick1+mid,tick2+mid])
                aax.set_yticklabels([mid-tick2,mid-tick1,mid,tick1+mid,tick2+mid], fontsize=13)
            else:
                aax.set_yticks([mid-tick2,mid,tick2+mid])
                aax.set_yticklabels([mid-tick2,mid,tick2+mid], fontsize=13)
        aax.set_title("{}".format(cname),fontsize=15)
        aax.coastlines()
    allfig.subplots_adjust(bottom=0.06, top=0.92, left=0.03, right=0.95,
                    wspace=0.02, hspace=0.18)
    allcb_ax = allfig.add_axes([0.92, 0.1, 0.03, .78])
    allcbar = allfig.colorbar(meshplot, cax=allcb_ax)
    #allcbar =allfig.colorbar(meshplot, ax=allaxes.ravel().tolist(),
    #                         orientation="vertical",fraction=0.08,pad=0.2,
    #                        shrink=0.8)
    allcbar.ax.tick_params(labelsize=12)
    allfig.suptitle("Mean cloud-type fractions",fontsize=20)
    #allfig.tight_layout()
    #allfig.savefig(os.path.join(work,"stats",name.replace(".parquet","_allctypes.svg")))
    allfig.savefig(os.path.join(work,"stats",name.replace(".parquet","_allctypes.pdf")))

    np.savez_compressed(name.replace(".parquet",".npz"),**pmeshdict)

    ctype_sum = df_label.sum(axis=0).compute()
    print(ctype_sum.head(),ctype_sum.shape)
    import seaborn as sns
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    fig,ax = plt.subplots(1,1,figsize=(8,8),gridspec_kw ={"wspace":0,"hspace":0})
    #ax=ax.flatten()
    patches,*_=ax.pie(ctype_sum[ctnames],autopct="%.2f",labels=None,colors=colors[1:9],
                        textprops={"fontsize":20})
    cbax = fig.add_axes([0.2,0.05,0.6,0.05])
    sns.heatmap(np.arange(8).reshape(1,-1),ax=cbax,annot=[["Ci","As","Ac","St","Sc","Cu","Ns","Dc"]],fmt="",
                    cmap=ListedColormap(colors[1:9]),cbar=False)
    cbax.tick_params(left=False,labelleft=False,bottom=False,labelbottom=False, right=False, labelright=False)
    fig.subplots_adjust(bottom=0.06, top=0.99, left=0.03, right=0.97,
                    wspace=0.02, hspace=0.0)
    ax.set_title("Full ICON",pad=-10)
    fig.savefig(os.path.join(work,"stats",name.replace(".parquet","_Pie.pdf")))

    fig, ax = plt.subplots(3,figsize=(8,8))
    bindict ={"lat":np.linspace(-90,90,900),"lon":np.linspace(-180,180,1800),"time":10}
    
    
    #very convoluted way to count how much data I have at each spatiotemporal cell
    futures=[]
    for i in sptemp:
        futures.append(client.submit(map_remote,df_label,i,bindict))
    client.wait_for_workers(2)
    for i,fut in enumerate(as_completed(futures)):
        height ,gp,bindict= fut.result()
        bins=bindict[gp]
        height.compute_chunk_sizes()
        try:
            shape = len(bins)-1
        except TypeError:
            shape = bins-1
        height=height.reshape(-1,shape).sum(axis=0)
        height=height.compute()
        ax[i].bar(bins[:-1], height,label=gp,width=bins[1]-bins[0])
        ax[i].set_xlabel(gp,fontsize=17)   
        ylim_bot = max(0,np.min(height)*0.9)
        ylim_top = np.max(height)*1.02     
        ax[i].set_ylim(ylim_bot,ylim_top)
    
    fig.tight_layout()
    #fig.savefig(os.path.join(work, "stats", name.replace(".parquet","_")+"counts.eps"))
    fig.savefig(os.path.join(work, "stats", name.replace(".parquet","_")+"counts.pdf"))

    #most dominant cloud type
    most_common=client.submit(get_max_fraction_simple,df_label).result()
    print(most_common.columns)
   
    
    #each pixel is assigned the cloud type which is largest than the mean occurence of the cloud type
    means = df_label.loc[:,ctnames].mean()
    gpby = df_label.groupby(["lat", "lon"])
    most_increased_all = gpby.mean()#split_out=int(df_label.npartitions/10))
    most_increased = most_increased_all.loc[:,ctnames]/means
    most_increased.columns=np.arange(len(ctnames))
    most_increased = most_increased.idxmax(axis=1).to_frame()
    most_common.columns=["mc ctype"]
    most_increased.columns=["mi ctype"]
    
    #most_increased = client.persist(most_increased)
    most_common = client.persist(most_common)
    
    if "seas" not in globals():
        gpby_lat=df_label.groupby("lat")
        #gpby_lon= df_label.groupby("lon")
        #meridional = gpby_lon.agg(["mean", "std"],split_out=int(df_label.npartitions/10)).iloc[:,-16:]
        zonal=gpby_lat.agg(["mean", "std"],split_out=int(df_label.npartitions/10)).iloc[:,-16:]
        del gpby, gpby_lat
    else:
        gpby_lat_north = df_label[df_label.lat>=0].groupby("lat")
        gpby_lat_south = df_label[df_label.lat<0].groupby("lat")
        zonal_north=gpby_lat_north.agg(["mean", "std"],split_out=int(df_label.npartitions/10)).iloc[:,-16:]
        zonal_south=gpby_lat_south.agg(["mean", "std"],split_out=int(df_label.npartitions/10)).iloc[:,-16:]        
        del gpby_lat_north,gpby_lat_south
    
    results=[]
    for i in [most_common,most_increased]:
        progress(i)
        if "mc" in list(i.columns)[0]:
            plotname="most_common"
        elif "mi" in list(i.columns)[0]:
            plotname="most_increased"    
            
        else:
            print("dunno")
            continue
        i.columns=["ctype"]
        print("beforeindex") 
        index=client.compute(i.index,)
        print("afterindex")
        #client.wait_for_workers(12)
        index=index.result()
        progress(index)
        
        coords=np.stack(list(index)).astype("float16")
        del index
        print(coords.shape)
        colors = client.compute(i.ctype.to_dask_array(),)
        print(type(colors))
        colors=colors.result()
        print(type(colors))
        progress(colors)
        fig=plt.figure(figsize=(50,5))
        ax=fig.add_subplot(1,1,1,projection=ccrs.PlateCarree())
        customize(ax)
        lon = coords[:,1]
        lat = coords[:,0]
        un_lat = np.unique(lat)
        un_lon = np.unique(lon)
        
        print(coords.shape, np.max(lat), np.max(lon), np.min(lat), np.min(lon))
        gg=np.empty((len(un_lat),len(un_lon)))
        for x,y,z in zip(lat,lon,colors):
            i = np.argwhere(un_lat==x)
            j= np.argwhere(un_lon==y)
            gg[i,j] = z
            
        lonlon,latlat = np.meshgrid(np.unique(lon),np.unique(lat))
        
        print("colors:",np.unique(colors))
        print(latlat.shape,lonlon.shape,gg.shape, np.max(latlat), np.max(lonlon), 
              np.sum(np.isnan(gg)))
        meshplot=ax.pcolormesh(lonlon, latlat,gg,cmap=cMap,shading="auto",
                           norm=Normalize(0,8), transform=ccrs.PlateCarree())
        plt.draw() 
        # Get proper ratio here
        xmin, xmax = ax.get_xbound()
        ymin, ymax = ax.get_ybound()
         
        
        y2x_ratio = (ymax-ymin)/(xmax-xmin)
        fig.set_figheight(50 * y2x_ratio)
        
        
        #fig.set_figwidth(50 / y2x_ratio)
        fig.tight_layout()
        cbar = fig.colorbar(meshplot, orientation ="horizontal", fraction=0.12,pad=0.02,
                            )
        cbar.ax.get_xaxis().set_ticks(np.arange(8)*1.0+0.5)
        cbar.ax.get_xaxis().set_ticklabels( [ 'Ci', 'As', 'Ac', 'St', 'Sc', 'Cu', 'Ns', 'Dc'],
                                           fontsize=12)
        
        fig.tight_layout()
        #fig.savefig(os.path.join(work,"stats", name.replace(".parquet","_")+plotname+".eps"))
        fig.savefig(os.path.join(work,"stats", name.replace(".parquet","_")+plotname+".pdf"))
        print(plotname, "done")
    """
    #make statistics for cloud types
    fig,ax = plt.subplots(6,3)
    for i, a in enumerate(ax):
        if i!=0:
            qu=[0.1,0.3,0.6,0.9,0.99][i-1]
            qu = df_label.Ci.quantile(qu)
            df_now = df_label[df_label.Ci>=qu].compute()
            print(df_now.head())
            df_now.iwp.hist(ax=a[0])
            df_now.ceri.hist(ax=a[1])
            df_now.ptop.hist(ax=a[2])
        else:
            
            df_now = most_increased_all[most_increased.mi_ctype==0].compute()
            print(df_now.head())
            df_now.iwp.hist(ax=a[0])
            df_now.ceri.hist(ax=a[1])
            df_now.ptop.hist(ax=a[2])
    fig.savefig(os.path.join(work,"stats",name.replace(".parquet","_cidevel.pdf")))
    del colors,coords,most_common,most_increased, i 
    """
    
    

    if "seas" not in globals():
        zonal=zonal.compute()
        zonal.sort_index(inplace=True)
        fig, ax =plt.subplots(4,2, sharex=True, figsize=(10,10))
        ax=ax.flatten()
        for i in range(len(ax)):
            if i*2<zonal.shape[1]:
                zonal.iloc[:,i*2].plot(ax=ax[i], label="mean")
                ax[i].set_title(zonal.columns[i*2][0])
                bottom = zonal.iloc[:,i*2]-zonal.iloc[:,i*2+1]
                bottom = np.where(bottom<0, 0,bottom)
                top = zonal.iloc[:,i*2]+zonal.iloc[:,i*2+1]
                ax[i].fill_between(zonal.index.values, (bottom), (top), color='b', alpha=.1, label=u"$\sigma$")
                ax[i].grid()
        ax[i].legend(fontsize=15)
                
        fig.tight_layout()
        #fig.savefig(os.path.join(work,"stats", name.replace(".parquet","_")+"zonal.svg"))
        fig.savefig(os.path.join(work,"stats", name.replace(".parquet","_")+"zonal.pdf"))
        del zonal
    else:
        for hemi in ["north","south"]:
            if hemi=="north":
                zonal_temp = zonal_north.compute()
            elif hemi=="south":
                zonal_temp=zonal_south.compute()
            zonal_temp.sort_index(inplace=True)
            fig, ax =plt.subplots(4,2, sharex=True, figsize=(10,10))
            ax=ax.flatten()
            for i in range(len(ax)):
                if i*2<zonal_temp.shape[1]:
                    zonal_temp.iloc[:,i*2].plot(ax=ax[i], label="mean")
                    ax[i].set_title(zonal_temp.columns[i*2][0])
                    bottom = zonal_temp.iloc[:,i*2]-zonal_temp.iloc[:,i*2+1]
                    bottom = np.where(bottom<0, 0,bottom)
                    top = zonal_temp.iloc[:,i*2]+zonal_temp.iloc[:,i*2+1]
                    ax[i].fill_between(zonal_temp.index.values, (bottom), (top), color='b', alpha=.1, label=u"$\sigma$")
                    ax[i].grid()
            ax[i].legend(fontsize=15)
            fig.tight_layout()
            #fig.savefig(os.path.join(work,"stats", name.replace(".parquet","_{}")+"zonal.svg"))
            fig.savefig(os.path.join(work,"stats", name.replace(".parquet","_{}".format(hemi)+"zonal.pdf")))
        del zonal_temp
    
    #plot timeseries for each cloud type
    #only takes into account cells in the southern hemisphere where at least
    #once the fraction of the cloud type is rather large
    if "times" in globals():
        df_now = df_label[df_label.lat<0]
        gpby = df_now.loc[:, ["lat","lon"]+ctnames].groupby(["lat","lon"]).max()
        condition = gpby.quantile(0.8)
        goodlocs = gpby[gpby>condition].compute() 
        fig, ax =plt.subplots(4,2, sharex=True, figsize=(10,10))
        ax=ax.flatten()
        for i,cname in enumerate(ctnames):
            select = goodlocs.loc[:,cname]
            sometimes_large = select[select>0]
            sometimes_large = sometimes_large.reset_index()
            large_lat = sometimes_large.lat
            large_lon = sometimes_large.lon
            
            sub = df_now.loc[:,["lat","lon","time",cname]]
       
            is_lat = sub.lat.isin(list(large_lat.values))
            is_lon = sub.lon.isin(list(large_lon.values))
            relevant = sub[is_lat&is_lon]
            relevant = relevant.persist()
            progress(relevant)
        
            gpby= relevant.groupby("time")
            temporal=gpby.agg(["mean", "std"]).iloc[:,4:]
            temporal=temporal.compute()
            temporal.sort_index(inplace=True)
            
            temporal.iloc[:,0].plot(ax=ax[i], label="mean")
            ax[i].set_title(temporal.columns[0][0])
            bottom = temporal.iloc[:,0]-temporal.iloc[:,1]
            bottom = np.where(bottom<0, 0,bottom)
            top = temporal.iloc[:,0]+temporal.iloc[:,1]
            days = temporal.index.values
            start=datetime.fromisoformat("1970-01-01")
            
            dates = [start+timedelta(days=float(x)) for x in days]
            ax[i].fill_between(dates, (bottom), (top), color='b', alpha=.1, label=u"$\sigma$")
            ax[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax[i].grid()

            
            del select,sometimes_large, large_lat,large_lon,sub,is_lat,is_lon,relevant,gpby
        ax[i].legend(fontsize=15)
        fig.autofmt_xdate()
        fig.tight_layout()
        #fig.savefig(os.path.join(work,"stats", name.replace(".parquet","_")+"temporal.svg"))
        fig.savefig(os.path.join(work,"stats", name.replace(".parquet","_")+"temporal.pdf"))    
        del temporal
        
    """
    meridional=meridional.compute()
    meridional.sort_index(inplace=True)
    fig, ax =plt.subplots(4,2, sharex=True, figsize=(10,10))
    ax=ax.flatten()
    for i in range(len(ax)):
        if i*2<meridional.shape[1]:
            meridional.iloc[:,i*2].to_frame().plot(ax=ax[i])
            ax[i].set_title(meridional.columns[i*2][0])
            bottom = meridional.iloc[:,i*2]-meridional.iloc[:,i*2+1]
            bottom = np.where(bottom<0, 0,bottom)
            top = meridional.iloc[:,i*2]+meridional.iloc[:,i*2+1]
            ax[i].fill_between(meridional.index.values, (bottom), (top),
                               color='b', alpha=.1)
            ax[i].legend(fontsize=17)
            ax[i].grid()
    fig.tight_layout()
    fig.savefig(os.path.join(work,"stats", name.replace(".parquet","_")+"meridional.eps"))
    del meridional
    
    """
