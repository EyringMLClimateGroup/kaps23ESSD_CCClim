"""
does most of the analysis of CCClim and the ICON-A cloud-type predictions
requires a running dask scheduler, with a lot of RAM if applied to complete CCClim (like several TB)
"""        
import time
from distributed import Client, progress
import pandas as    pd
pd.options.display.width = 0
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
from matplotlib.colors import ListedColormap, Normalize
import matplotlib.dates as mdates
import cartopy.crs as ccrs
import sys
from datetime import datetime,timedelta
import os
import traceback
import glob
import dask.dataframe as dd
import global_land_mask as globe
from scipy.optimize import curve_fit
import seaborn as sns
from matplotlib.patches import Rectangle
import pyarrow 

def mk_increased(dataframe,source,extent,ctype_bounds=None):
    """Plots a global map of the most increased cloud type per grid cell
        this really only does the plotting, the data should be comput   ed with dask already
    Args:
        dataframe (pd.DataFrame): readymade most increased frame
        source (string): input data description
        extent (int or float?): modifies the figure size depending on the relative lat/lon range of the data
    """    

    latlat,lonlon = np.meshgrid(dataframe.index.values,
                                np.array([x[1] for x in dataframe.columns]))
    cMap= ListedColormap([(.9,.6,0),(.35,.7,.9),(0,.6,.5),(.95,.9,.25),(0,.45,.7),
                (.8,.4,0),(.8,.6,.7),(0,0,0)])
    increased_fig  =plt.figure(  figsize=(10,6*extent))
    increased_ax=increased_fig.add_subplot(1,1,1,projection=ccrs.PlateCarree())
                        
    miplot = increased_ax.pcolormesh(lonlon, latlat, dataframe.values.transpose(),cmap=cMap,
                                    norm=Normalize(0,7), transform=ccrs.PlateCarree())
    if ctype_bounds is not None:
        for i,((latmin,latmax),(lonmin,lonmax)) in enumerate(ctype_bounds):
            increased_ax.add_patch(Rectangle((lonmin, latmin), lonmax-lonmin, latmax-latmin,
                                   lw=3,fill=False,edgecolor="black"))
            increased_ax.text(lonmin+1,latmin+1,[ 'Ci', 'As', 'Ac', 'St', 'Sc', 'Cu', 'Ns', 'Dc'][i],color="white")

    cbar = increased_fig.colorbar(miplot, orientation ="horizontal", 
                                    fraction=0.12,pad=0.02, )
    cbar.ax.get_xaxis().set_ticks(np.arange(8)*0.88+0.45)
    cbar.ax.get_xaxis().set_ticklabels( [ 'Ci', 'As', 'Ac', 'St', 'Sc', 'Cu', 'Ns', 'Dc'],
                                           fontsize=12)        
    increased_ax.coastlines()   
    increased_ax.set_yticks([-80,-40,0,40,80]    )
    increased_ax.set_yticklabels(["80°S", "40°S", "0°N", "40°N", "80°N"])
    increased_ax.set_xticks([-150,-100,-50,0,50,100,150])
    increased_ax.set_xticklabels(["150°W",  "100°W", "50°W", "0°E", "50°E", "100°E",  "150°E"])
    increased_ax.tick_params(bottom=False, labelbottom=False, top=True, labeltop=True,)
    increased_fig.savefig(os.path.join(work,"stats","{}most_increased.pdf".format(source+qualifier)))

def mk_corr(dataframe,source):
    """makes a correllation heatmap of the data"""
    corrmat = dataframe.corr().compute()
    hmfig,hmax = plt.subplots(figsize=(12,12))
    sns.heatmap(corrmat.round(3),annot=True, ax=hmax)
    hmfig.savefig(os.path.join(work,"stats","{}_{}corr.pdf".format(source,qualifier)))            


def mk_cloudsums(dataframe,source):
    """makes stackplots of the total distribution of the types in the Data,
            """
    if "clr" in qualifier:
        cloudsum = dataframe.loc[:,clear+ctnames].sum(0)
        cloudsum=cloudsum.compute().to_frame("Dataset Distribution")
        progress(cloudsum)
        cloudsum.index=["undetermined"]+ctnames
        cMap= ListedColormap([(1,1,1),(.9,.6,0),(.35,.7,.9),(0,.6,.5),(.95,.9,.25),(0,.45,.7),
                (.8,.4,0),(.8,.6,.7),(0,0,0)])
    else:
        cloudsum = dataframe.loc[:,ctnames].sum(0)
        cloudsum=cloudsum.compute().to_frame("%")
        progress(cloudsum)
        cMap = ListedColormap([(.9,.6,0),(.35,.7,.9),(0,.6,.5),(.95,.9,.25),(0,.45,.7),
                (.8,.4,0),(.8,.6,.7),(0,0,0)])
        
    try:
        cloudsums = pd.read_pickle(os.path.join(work, "stats/CLMCLS_sums.pkl"))
        if qualifier not in cloudsums.index:
            to_append=cloudsum.transpose()
            to_append.index=[qualifier]
            cloudsums = pd.concat([cloudsums,to_append])
            cloudsums.to_pickle(os.path.join(work, "stats/CLMCLS_sums.pkl"))
    except FileNotFoundError:
        cloudsums = cloudsum.transpose()
        cloudsums.index=[qualifier]
        cloudsums.to_pickle(os.path.join(work, "stats/CLMCLS_sums.pkl"))

    stckfig, stckax =plt.subplots(  figsize=(8,8))
    cloudsum = (cloudsum/cloudsum.sum()*100).transpose()
    print(cloudsum.head())
    cloudsum.plot.bar(ax=stckax,subplots=False,legend=None,cmap=cMap,
                      #textprops = {"fontsize":"25","color":(0.1,.1,.1)},
                      stacked=True, width = 0.1)

    for j,c in enumerate(stckax.containers):
        stckax.bar_label(c, fmt="{}: %.2f".format(ctnames[j]), label_type='center',fontsize=20,color=(1,1,1) if j==7 else (.1,.1,.1))

    plt.axis("off")
    #stckfig.legend(False)
    stckfig.tight_layout()
    stckfig.savefig(os.path.join(work,"stats","{}stack.pdf".format(source+qualifier)))
    del cloudsum,stckax,stckfig


def develplot(dataframe,source,fig,ax,all_subaxs):
    """Does involved sampling of the data to plot distributions of physical variables in 
        the characteristic cells and the distribution of co-occurring cloud types

    Args:
        dataframe (dd.DataFrame): dask frame containing pretty much everything
        source (string): datasoource
        fig (_type_): pyplot args
        ax (_type_): pyplot args
        all_subaxs (_type_): pyplot args
    Raises:
        ValueError: If the construction of the bins goes wrong. It won't

    Returns:
        _type_:the input figure
    """    
    plt.style.use('seaborn-white')
    kwargs = dict(histtype='stepfilled', alpha=0.3, density=True, ec="k")
    #locatins where to put text
    locy = [ .8,.8,.8,.8]
    if source=="CCClim":
        tcwp="twp"
        locx = .5
        color=colors[0]
    elif source=="ICON":
        tcwp="cwp"
        locx = 0.8
        color=colors[1]
    interesting_features = {"Ci":["iwp","ceri","cod","ptop"],
                        "As":[tcwp,"cerl","ceri","ptop"],
                        "Ac": [tcwp,"cerl","ceri","ptop"],
                        "St": ["lwp","cerl","cod","ptop"],
                        "Sc": ["lwp","cerl","cod","ptop"],
                        "Cu": ["lwp","cerl","cod","ptop"],
                        "Ns": ["iwp","ceri","cod","ptop"],
                        "Dc": ["lwp","iwp","cod","ptop"]}
    for i, a in enumerate(ax):
        cname = ctnames[i]
        prop_choices = interesting_features[cname]
        #makes sure the cells contains essentially only the cloud type
        # of interest and clear sky
        b00l = (dataframe[cname]+dataframe["clear"])>0.85
        df_now = dataframe[b00l].compute()
        #makes sure the cells contain a significant amount of the cloud type of interest
        b00l = df_now[cname]>=df_now[cname].median()
        df_now = df_now[b00l]
        #makes sure the cells are not weird because predictions were weird
        df_now = df_now[df_now.iwp+df_now.lwp>0.1]
        #selection of the 4 most co-occurring types
        means = df_now[ctnames].mean().sort_values(ascending=False)
        means = means.drop(cname)
        subaxs = all_subaxs[i]
        for ht,hightype in enumerate(means.index[:4]):
            df_now[hightype].hist(ax=subaxs[ht],grid=False,color=color,
                                      bins=np.linspace(0,.3,15), **kwargs)
            if ht<2:
                subaxs[ht].tick_params(labelleft=False, left=False, 
                                   bottom=False, labelbottom=False)
            else:
                subaxs[ht].tick_params(labelleft=False, left=False, 
                                   bottom=False, labelbottom=True)
                subaxs[ht].set_xticks([0,0.2])
                subaxs[ht].set_xticklabels([0,0.2])
                subaxs[ht].set_xlabel("RFO")
            subaxs[ht].text(locx,locy[ht],hightype,horizontalalignment='center', verticalalignment='center',
                     transform=subaxs[ht].transAxes,fontsize=12,rotation="horizontal",
                     color = color)
        #main ctype histogram
        df_now[cname].hist(ax=a[0],grid=False,bins=np.linspace(0,1,50),label=source,
                           **kwargs)
        a[0].set_title(cname,fontsize=20)
        a[0].ticklabel_format(axis="y", style="sci",scilimits=[-1,1])
        a[0].set_xlabel("RFO")
        a[0].set_ylabel("Prob. Density")
        if source=="ICON" and cname=="Ci":
            a[0].legend(loc = "upper left")
        #interesting features histograms
        for jj,choice in enumerate(prop_choices):
            j=jj+1
            if len(df_now)==0:
                print("no samples")
                continue
            if choice=="ptop":
                bins=np.linspace(50,1100,40)
            else:
                max_val = df_now[choice].max()
                if cname=="Ci":
                    if choice=="cod":
                        max_val = 40
                    elif choice=="iwp":
                        max_val=100
                    elif choice=="ceri":
                        max_val = 50
                elif cname == "As":
                    if choice==tcwp:
                        max_val =100
                    elif choice == "cerl":
                        max_val =10
                    elif choice=="ceri":
                        max_val =50
                elif cname == "Ac":
                    if choice == tcwp:
                        max_val=800
                    elif choice == "cerl":
                        max_val =18
                    elif choice=="ceri":
                        max_val=10
                elif cname == "Sc":
                    if choice=="lwp":
                        max_val=400
                    elif choice=="cod":
                        max_val = 100
                elif cname=="Ns":
                    if choice=="iwp":
                        max_val=800
                    elif choice=="ceri":
                        max_val = 70
                    elif choice=="cod":
                        max_val=170
                    
                #constructs elaborate bins that are wide at 0, thin close to 0 and then become wider
                #this makes the height of the bars all roughly the same adn therefore easier to gauge whats going on
                max_log = np.log(max_val+1)
                numbins = 25#int(25.-((max_val-df_now[choice].min())/100))
                assert numbins >=5,numbins
                bins = (np.e**np.linspace(0,max_log,numbins)-0.9)
                diffs = bins[1:]-bins[:-1]
                while True:
                    maxmul=10
                    if maxmul<1:
                        break
                    try:
                        newbins = bins
                        newbins[1:]= bins[1:]+diffs*np.linspace(0,maxmul,len(diffs))[::-1]
                        if np.all(newbins[1:]-newbins[:-1]>0):
                            break
                        else:
                            raise ValueError("need monotonic increase")
                    except ValueError as e:
                        print(e)
                        maxmul*=0.9
            
            df_now[choice].hist(ax=a[j],bins=bins,grid=False,**kwargs)
            a[j].set_title("  "+longprops[choice],fontsize=12)
            a[j].set_xlabel(units[choice],fontsize=12)
            a[j].ticklabel_format(axis="y", style="sci",scilimits=[-1,1])
        
        [ a[x].tick_params(labelleft=False) for x in range(1,5) ]
    return fig            
    

def main(only_use=None):
        
    print(sys.argv, flush=True)
    global work, scratch,ctnames, propnames, qualifier,longprops,units
    work = os.environ["WORK"]
    scratch = os.environ["SCR"]
    tried=0
    while True:
        #launch a dask scheduler and save its configuraton in this file before starting this script
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
    
    print(client.dashboard_link,flush=True)
    plt.close("all")
    ctnames = [ "Ci", "As", "Ac", "St", "Sc", "Cu", "Ns", "Dc"]
    
    ctype_bounds = [((-10,10),(40,180)),((-90,-30),(-180,180)),
                    ((0,20),(-5,150)),((-60,50),(-150,10)),
                    ((-60,50),(-150,10)),((-40,30),(-180,-80)),
                    ((40,90),(-180,180)),((-10,10),(40,180))]
    propnames = ['twp', 'lwp', 'iwp', 'cerl', 'ceri', 'cod', 'ptop', 'tsurf']
    clear = ["clear"]
    longprops={"clear": "Clear sky fraction","Ci":"Cirrus/Cirrostratus fraction",
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
                  "cwp":"Cld. Water P.", "twp":"Cld. Water P.",
                  "lwp": "Liquid Water P.", "iwp":"Ice Water P.",
                  "cod": "Cld. Opt. Depth", "tsurf": "surface temperature",
                  "tsurf": "surface temperature", "cee": "emissivity",
                  "ptop": "Cld. Top Press.", "htop": "cloud top height",
                  "ttop": "cloud top temperature", "cerl": "Eff. Droplet Radius",
                  "ceri": "Eff. Ice Part. Rad.","ptop":"Cld. Top Press."}
    
    units ={'twp':"g/m²", 'cwp':"g/m²", 'lwp':"g/m²", 'iwp':"g/m²", 'cerl':"µm", 'ceri':"µm", 'cod':"-", 
                    'ptop':"hPa", 'tsurf':"K"}
    
    propfig, propax =plt.subplots(4,2, sharex=True, figsize=(25,10))
    propax=propax.flatten()
    ctfig, ctax =plt.subplots(4,2, sharex=True, figsize=(25,12))
    ctax=ctax.flatten()
    compfig, compax =plt.subplots(4,2, sharex=True, figsize=(25,10))
    compax=compax.flatten()
    MMfig, MMax =plt.subplots(4,2, sharex=True, figsize=(25,10))
    MMax=MMax.flatten()
    Mweekfig, Mweekax =plt.subplots(4,2, sharex=True, figsize=(25,14))
    Mweekax=Mweekax.flatten()
    
    ranges = [(0,800),(0,750),(0,500),(0,14),(0,28),(0,50),(0,950),(175,350)]
    ranges_dict = {x:y for x,y in zip(propnames,ranges)}
    flist = glob.glob(os.path.join(scratch, "ESACCI/parquets/w_anc/ESACCI_de????f*.parquet"))
    ICONfile = os.path.join(work, "frames/parquets/ICONframe_threshcodp2100_10000_r360x180_0123459.parquet")
    
    if only_use is not None:
        only_use = [x for x in only_use.split(",")]
        flist = [x for x in flist if np.any([y in x for y in only_use ])]
    assert len(flist)>0
    flist.sort()
    flist=flist[:]
    #the iconfile has to be last in the list for reasons
    flist.append(ICONfile)
    df_full =[]
    qualifier = sys.argv[1] 
    for fnum, fname in enumerate(flist):
        print(fname,flush=True)
        if fnum==len(flist)-1:
            #switches to ICON naming when that file is reached
            propnames[0]="cwp"
        #removes files that have errored before
        if os.path.exists(fname.replace(".parquet","error")):
            [os.remove(x) for x in glob.glob(fname+"/*")]
            os.rmdir(fname)
            continue

        sptemp = ["time","lat", "lon"] 
        df=dd.read_parquet( fname,columns=sptemp+clear+ctnames+propnames)

        dtypes = {x:"float16" for x in sptemp}
        for cname in ctnames :
            dtypes[cname]="float32"
            

        try:
            os.remove(fname.replace(".parquet","error"))
        except Exception as err:
            print("no previous errors: ",err)

        
        if "clr" in qualifier:
            #do not normalize independent of cloud amount
            pass
        else:
            #comlicated becuase values in dask can not be overwritten
            s=df.loc[:,ctnames].sum(axis="columns")
            df[[x+"n" for x in ctnames]] = df.loc[:,ctnames].truediv(s,axis="index")
            df=df.drop(ctnames,axis=1)
            df.columns=sptemp+clear+propnames+ctnames

        if "2010" in fname:
            #exclude july 2010 because that has wrong data
            start = datetime.fromisoformat("1970-01-01")
            start_july = datetime.fromisoformat("2010-07-01")
            end_july = datetime.fromisoformat("2010-07-31")
            ex_july = (df.time>(end_july-start).days)|( df.time<(start_july-start).days)
            df=df[ex_july]
        elif "1994" in fname:
            #exclude last quarter of 1994 because weird/missing data
            start = datetime.fromisoformat("1970-01-01")
            start_january = datetime.fromisoformat("1994-01-01")
            ex_260 = df.time- (start_january-start).days < 260
            df=df[ex_260]
        elif "1986" in fname or "1985" in fname:
            #exclude first january
            start = datetime.fromisoformat("1970-01-01")
            start_january = datetime.fromisoformat("1986-01-01")
            ex_11 = (df.time>(start_january-start).days+1) | (df.time<(start_january-start).days-1)
            df=df[ex_11]
        

        
        if fnum==len(flist)-1 and "ICON" in fname:
            df_full_ICON=df.copy().persist()
            
        else:
            #this is the unfiltered data used for normalization
            df_full.append(df.persist())

        #log at only some latitude bands
        if not ("All" in qualifier):
            if "special" in qualifier:
                df = df[df.lat>30]
                df = df[df.lat<60] 
                df = df[df.lon<0]
                df = df[df.lon>-60]
            elif "Trop" in qualifier:
                df = df[df.lat>-15]
                df = df[df.lat<15] 
            elif "Mid" in qualifier:
                df= df[np.abs(df.lat) < 60]
                df = df[np.abs(df.lat) > 30]
            elif "Sub" in qualifier:
                df= df[np.abs(df.lat) < 30]
                df = df[np.abs(df.lat) > 15]
            elif "nopoles" in qualifier:
                df=df[np.abs(df.lat)<75]
            if "North" in qualifier:
                df = df[df.lat>0]
            elif "South" in qualifier:
                df = df[df.lat<0]
            

            if "Ocean" in qualifier:
                true = df.map_partitions(lambda x: globe.is_ocean(x.lat,x.lon))
                df = df.loc[true,:]
            elif "Land" in qualifier:
                true = df.map_partitions(lambda x: globe.is_land(x.lat,x.lon))
                df = df.loc[true,:]
        #ICON has some absurd cod values    
        df.cod=df.cod.where(df.cod<250,other=250)
        if fnum==0:
            df_all=[df]
        elif fnum==len(flist)-1 and "ICON" in fname:
            df_all_ICON = df
        else:
            df_all.append(df.persist())

    #put the filtered data into distributed memory
    df_all=dd.concat(df_all,axis=0,ignore_index=True)
    df_full=dd.concat(df_full,axis=0,ignore_index=True).persist()
    df_all=df_all.persist()
    progress(df_all)
    """
    #create the multiplot setup for the develplots
    dfig,dax = plt.subplots(8,6,figsize=(14,18))
    
    subaxs=[]
    for a in dax:
        subsubaxs=[]
        subsubaxs.append(a[-1].inset_axes([0.0, 0.5, 0.45, 0.45]))
        subsubaxs.append(a[-1].inset_axes([0.5, 0.5, 0.45, 0.45]))
        subsubaxs.append(a[-1].inset_axes([0.0, 0.0, 0.45, 0.45]))
        subsubaxs.append(a[-1].inset_axes([0.5, 0.0, 0.45, 0.45]))
        a[-1].tick_params(labelleft=False, left=False, 
                                   bottom=False, labelbottom=False)
        subaxs.append(subsubaxs)
        a[-1].axis("off")
    dfig=develplot(df_all,"CCClim",dfig,dax,subaxs)
    dfig=develplot(df_all_ICON,"ICON",dfig,dax,subaxs)
    dfig.tight_layout() 
    dfig.subplots_adjust(#bottom=0.06, 
                        #top=.97, 
                        #left=0.05, 
                        #right=.98,
                    wspace=0.04,
                     # hspace=0.1
                     )
    dfig.savefig(os.path.join(work,"stats","both_alldevel_{}.pdf".format(qualifier)))
    
    #mk_cloudsums(df_all,"CCClim")
    #mk_cloudsums(df_all_ICON,"ICON")
    """
    #most increased plot
    progress(df_full)
    extent = np.abs(df_all.lat.max() - df_all.lat.min())/180.
    df_globe = df_all.groupby(["lat","lon"]).mean()
    cloudmean = df_full.loc[:,ctnames].mean()
    most_increased = (df_globe.loc[:,ctnames]/cloudmean).idxmax(axis=1).to_frame("most_increased")
    
    del  cloudmean, df_globe
    most_increased=most_increased.compute()
    #creates hierachichal indices so each cloud type can be accessed as a 2D field
    most_increased = most_increased.unstack()
    most_increased = most_increased.applymap(lambda x: ctnames.index(x) if isinstance(x,str) else np.nan)
    #most increased plot ICON
    progress(df_full_ICON)
    
    extent_ICON = np.abs(df_all_ICON.lat.max() - df_all_ICON.lat.min())/180.
    df_globe = df_all_ICON.groupby(["lat","lon"]).mean()
    cloudmean = df_full_ICON.loc[:,ctnames].mean()
    most_increased_ICON = (df_globe.loc[:,ctnames]/cloudmean).idxmax(axis=1).to_frame("most_increased")
    del  cloudmean, df_globe
    most_increased_ICON=most_increased_ICON.compute()
    most_increased_ICON = most_increased_ICON.unstack()
    most_increased_ICON = most_increased_ICON.applymap(lambda x: ctnames.index(x) if isinstance(x,str) else np.nan)
    
        
    #mk_corr(df_all,"CCClim")
    #mk_corr(df_all_ICON,"ICON")

    mk_increased(most_increased,"CCClim",extent,ctype_bounds=None)#ctype_bounds)
    del most_increased
    mk_increased(most_increased_ICON,"ICON",extent_ICON,ctype_bounds=None)#ctype_bounds)
    del most_increased_ICON
    
    #timeseries stuff    
    start=datetime.fromisoformat("1970-01-01")
    #gets the days since 1970 into the DF
    df_time = df_all.time.apply( lambda x: start+timedelta(days=x),
                                    meta=('time', 'datetime64[ns]')).compute()
    cloud = df_all.loc[:,ctnames].compute()
    #df_monthly = pd.concat([df_time.dt.month, df_time.dt.year, cloud],axis=1)
    #used to compute a typical seasonal cycle
    df_typical = pd.concat([ df_time.dt.month, df_time.dt.isocalendar().week, cloud],axis=1)
    #df_monthly.columns=["month","year"] +ctnames
    df_typical.columns=["month","week"] + ctnames
    #AllMM = df_monthly.groupby(["year","month"]).mean()
    AllMweek = df_typical.groupby(["week"]).mean()
    del df_typical
    df_weekly =pd.concat([  df_time.dt.isocalendar().week,
                            df_time.dt.year, cloud],axis=1)
    df_weekly.columns=["week","year"]+ctnames
    df_weekly = df_weekly.groupby(["week","year"]).mean()
    anomaly = df_weekly.reset_index()
    dw_temp = AllMweek.drop(labels="month",axis=1)
    anomaly = anomaly.join(dw_temp,on = "week",rsuffix="mean")
    del dw_temp
    for cname in ctnames:
        anomaly[cname]=anomaly.loc[:,cname]-anomaly.loc[:,cname+"mean"]
        anomaly.drop(labels=cname+"mean",axis=1,inplace=True)
    anomaly.set_index(["year","week"],inplace=True)
    print(anomaly.head())
    
    #df_yearly=df_yearly.reset_index().set_index(["month","week"])
    only_time = df_all.time.compute()#required for axis limits
    gpby = df_all.loc[:, ["lat","lon"]+ctnames].groupby(["lat","lon"]).max().compute()
    #filter out completely uninteresting samples
    condition = gpby.quantile(0.01)
    goodlocs = gpby[gpby>condition]
    del df_time
    #same sht for ICON
    gpby = df_all_ICON.loc[:, ["lat","lon"]+ctnames].groupby(["lat","lon"]).max().compute()
    df_time_ICON = df_all_ICON.time.apply( lambda x: start+timedelta(days=x),
                                    meta=('time', 'datetime64[ns]')).compute()
    cloud_ICON = df_all_ICON.loc[:,ctnames].compute()
    df_typical_ICON = pd.concat([ df_time_ICON.dt.month, df_time_ICON.dt.isocalendar().week, cloud_ICON],axis=1)
    df_typical_ICON.columns=["month","week"] + ctnames
    AllMweek_ICON = df_typical_ICON.groupby(["week"]).mean()
    df_weekly_ICON =pd.concat([  df_time_ICON.dt.isocalendar().week,
                            df_time_ICON.dt.year, cloud_ICON],axis=1)
    df_weekly_ICON.columns=["week","year"]+ctnames
    df_weekly_ICON = df_weekly_ICON.groupby(["week","year"]).mean()
    
    condition = gpby.quantile(0.01)
    goodlocs_ICON = gpby[gpby>condition]
    del gpby,df_time_ICON

    
    #multiplot of single classes
    for i,cname in enumerate(ctnames):
        print(cname,flush=True)
        select = goodlocs.loc[:,cname]
        select_ICON = goodlocs_ICON.loc[:,cname]
        sometimes_large = select[select>1e-4]
        sometimes_large_ICON = select_ICON[select_ICON>1e-4]
        sometimes_large = sometimes_large.reset_index()
        sometimes_large_ICON = sometimes_large_ICON.reset_index()
        #rectangles, in which a specific cloud type is interesting, as inferred visually from most_increased
        #(latmin, latmax),(lonmin,lonmax)=ctype_bounds[i]
        (latmin,latmax),(lonmin,lonmax) = (-90,90),(-180,180)
        sometimes_large = sometimes_large[(sometimes_large.lat<latmax)&(sometimes_large.lat>latmin)]
        sometimes_large = sometimes_large[(sometimes_large.lon<lonmax)&(sometimes_large.lon>lonmin)]
        sometimes_large_ICON = sometimes_large_ICON[(sometimes_large_ICON.lat<latmax)&(sometimes_large_ICON.lat>latmin)]
        sometimes_large_ICON = sometimes_large_ICON[(sometimes_large_ICON.lon<lonmax)&(sometimes_large_ICON.lon>lonmin)]
        large_lat = sometimes_large.lat
        large_lon = sometimes_large.lon
        large_lat_ICON = sometimes_large_ICON.lat
        large_lon_ICON = sometimes_large_ICON.lon
        sub = df_all.loc[:,["lat","lon","time",cname]]
        sub_ICON = df_all_ICON.loc[:,["lat","lon","time",cname]]
    
        is_lat = sub.lat.isin(list(large_lat.values))
        is_lon = sub.lon.isin(list(large_lon.values))
        relevant = sub[is_lat&is_lon]
        is_lat_ICON = sub_ICON.lat.isin(list(large_lat_ICON.values))
        is_lon_ICON = sub_ICON.lon.isin(list(large_lon_ICON.values))
        relevant_ICON = sub_ICON[is_lat_ICON&is_lon_ICON]

        gpby= relevant.groupby("time")
        temporal=gpby.agg(["mean", "std"]).iloc[:,4:].compute()
        temporal.sort_index(inplace=True)
        gpby_ICON= relevant_ICON.groupby("time")
        temporal_ICON=gpby_ICON.agg(["mean", "std"]).iloc[:,4:].compute()
        temporal_ICON.sort_index(inplace=True)
        
        tp =temporal.iloc[:,0].plot(ax=ctax[i],label="mean")
        if fnum==0:
            ctax[i].set_title(temporal.columns[0][0])
        bottom = temporal.iloc[:,0]-temporal.iloc[:,1]
        bottom = np.where(bottom<0, 0,bottom)
        top = temporal.iloc[:,0]+temporal.iloc[:,1]
        days = temporal.index.values
        
        dates = [start+timedelta(days=float(x)) for x in days]
        bp =ctax[i].fill_between(dates, (bottom), (top), color='b', alpha=.1, 
                                    label=u"$\sigma$" if i==0 else None )
        ctax[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ctax[i].tick_params(labelsize=19)
        ctax[i].grid()
        ctax[i].set_title(cname,fontsize=18)
        ctax[i].set_xlabel("")
        fit,_ = curve_fit(lambda x,m,c: m*x+c,xdata = only_time,ydata=cloud[cname])
        m,c = fit
        fp, =ctax[i].plot([only_time.min(),only_time.max()],
                    [only_time.min()*m+c,only_time.max()*m+c],"--r",label="slope {:.2e}/year".format(m*365))
        if not i%2:
            ctax[i].set_ylabel("RFO", fontsize=20)

        

        Mweek = AllMweek.loc[:,cname].to_frame(cname)
        
        Mweek_ICON = AllMweek_ICON.loc[:,cname].to_frame(cname)
        
        comp =anomaly.loc[:,cname].to_frame(cname)
        
        
        cloud_weekly = df_weekly.reset_index().set_index("week")
        cloud_weekly_ICON = df_weekly_ICON.reset_index().set_index("week")
        
        comp=comp.sort_index()
        print(comp.head(), comp.shape)
        comp.loc[:,cname].plot(ax=compax[i],linestyle="-", marker=".")
        #Mweek_std = df_monthly.groupby(["month"]).std()
        #AllMM[cname].plot(ax=MMax[i],label="monthly_mean")
        Mweek.index = np.arange(len(Mweek))
        
        Mweek[cname].plot(use_index=True,ax=Mweekax[i],label="CCClim: Avg RFO")
        Mweek_ICON[cname].plot(use_index=True,ax=Mweekax[i],label="ICON: Avg RFO")
        if not i%2:
            Mweekax[i].set_ylabel("RFO", fontsize=20)
        #mbottom = Mweek.loc[:,cname]-Mweek_std.loc[:,cname]
        #mbottom = np.where(mbottom<0, 0,mbottom)
        #mtop = Mweek.loc[:,cname]+Mweek_std.loc[:,cname]
        
        #mbp =Mweekax[i].fill_between(Mweek.index.values, (mbottom), (mtop), color='b', 
        #                    alpha=.1, label=u"$\sigma$" if i==0 else None )
        #MMax[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        MMax[i].grid()
        MMax[i].set_title(cname)
        #compax[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%W'))#
        compax[i].grid()
        compax[i].set_title(cname)
        #Mweekax[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        Mweekax[i].grid()
        Mweekax[i].tick_params(labelsize=24)
        Mweekax[i].set_title(cname,fontsize=26)
        Mweekax[i].set_xlabel("Calendar week",fontsize=28)
        if i==0:
            ctax[i].legend( fontsize=15)
            Mweekax[i].legend( fontsize=24)
        else:
            ctax[i].legend(handles=[fp], fontsize=15)

        del select,sometimes_large, large_lat,large_lon,sub,is_lat,is_lon,relevant,gpby



    
    ctfig.autofmt_xdate()
    ctfig.tight_layout()
    ctfig.savefig(os.path.join(work,"stats", "{}ESAtemporal.pdf".format(qualifier)))    
    propfig.autofmt_xdate()
    propfig.tight_layout()
    propfig.savefig(os.path.join(work,"stats","{}ESAtempprops.pdf".format(qualifier)))
    MMfig.autofmt_xdate()
    MMfig.tight_layout()
    MMfig.savefig(os.path.join(work,"stats","{}ESAMM.pdf".format(qualifier)))
    #Mweekfig.autofmt_xdate()
    Mweekfig.tight_layout()
    Mweekfig.savefig(os.path.join(work,"stats","{}ESAMweek.pdf".format(qualifier)))
    compfig.autofmt_xdate()
    compfig.tight_layout()
    compfig.savefig(os.path.join(work,"stats","{}comp.pdf".format(qualifier)))

if __name__=="__main__":
    if len(sys.argv)>2:
        main(only_use=sys.argv[2])
    else:
        main()
