"""defines a bunch of functions that make plots usable for process-based analysis of the ICON data"""
import numpy as np
import pandas as pd
import os
import glob
from matplotlib import pyplot as plt
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
from matplotlib.colors import ListedColormap as Cmap
import seaborn as sns
from tqdm import tqdm
import multiprocessing as mlp
from datetime import datetime, timedelta
from src.utils import timer_func, grid_amount,allinone_kde


@timer_func
def allinone_mesh(rea,limit=1e5,d=""):
    """plots the amounts of cloud type per CRE into a grid,
        both all types into one grid and a plot per ctype

    Args:
        rea (_type_): _description_
        limit (_type_, optional): _description_. Defaults to 1e5.
        d (str, optional): _description_. Defaults to "".
    """    
    print("allinone_mesh",flush=True)
    num_points = 200
    palette=sns.color_palette("colorblind")[1:]
    white = np.array([1,1,1.])
    colors2 = [np.array(rgb) for rgb in palette]
    gradients = [[x*y+white*(1-y) for y in np.linspace(0,1,num_points)**2] for x in colors2]
    qval = 1-limit/len(rea)
    fig, ax = plt.subplots(1,1,figsize=(12,12))
    fig2, ax2 = plt.subplots(2,4,figsize=(12,12),sharex=True, sharey=True)
    ax2=ax2.flatten()
    legend_elems=[]
    lw = rea.lw_cre.values
    sw = rea.sw_cre.values
    zz_mix = []
    print("qval",qval)
    for j,cname in tqdm(enumerate(ctnames)):
        sample = np.argwhere((rea[cname]>rea[cname].quantile(qval)).values)
        assert len(sample)<limit+10 and len(sample)>limit-10,(qval, len(sample))
        sample=sample.squeeze()
        temp = rea.iloc[sample]
        xx,yy,zz = grid_amount(lw,sw,rea[cname].values)
        zz_mix.append(zz)
        assert not np.any(np.isnan(xx))
        assert not np.any(np.isnan(yy))
        mehs = ax2[j].pcolormesh(xx, yy, zz, cmap="Greys",rasterized=False)#, vmin=0,vmax=1)
        ax2[j].set_title(cname,fontsize=20)
        ax2[j].tick_params(labelsize=16)
        textx=temp.lw_cre.median()
        texty=temp.sw_cre.median()
        ax.text(textx,texty,cname,fontsize=18,color=palette[(j % len(palette))])
        legend_elems+=[plt.Line2D([0], [0], color=palette[j % len(palette)], linewidth=4, label=cname)]
        
    fig2.tight_layout()
    fig2.subplots_adjust(bottom=0.06, top=.97, left=0.05, right=.98,
                    wspace=0.02, hspace=0.1)
    cbax = fig2.add_axes([0.1, 0.025, .8, .009])
    fig2.colorbar(mehs,cax=cbax,orientation="horizontal")
    cbax.tick_params(labelsize=16)
    zz_mix = np.stack(zz_mix)
    zz_mix = np.where(zz_mix == np.max(zz_mix,0),zz_mix,np.nan)
    for j,cname in enumerate(ctnames):
        mesh = ax.pcolormesh(xx,yy,zz_mix[j],cmap=Cmap(gradients[j]))
    #ax.plot([200,350],[0,0],"--w")
    ax.set_ylim(np.min(yy), np.max(yy))
    ax.set_xlim(np.min(xx), np.max(xx))
    ax.legend(handles=legend_elems, fontsize=18, ncol=4, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(work,"stats/ICONCREallinone_mesh{}.pdf".format(d)))
    fig2.savefig(os.path.join(work,"stats/ICONCREallseperate_mesh{}.pdf".format(d)))




def npz_extractor(path):
    """gets the relevant properties and puts it in a usable timescale"""
    ds=np.load(path)
    locs = ds["locations"]
    props = ds["properties"]
    props = props[-7:]
    lat,lon,Time = locs
    start = datetime.fromisoformat("1970-01-01")
    Time=Time.flatten().astype(str)
    Time = [datetime.fromisoformat("{}-{}-{}".format(x[:4],x[4:6],x[6:8])) for x in Time]
    Time = [(x - start).days for x in Time]
    wap = props[-1]
    SW_CRE = props[1]-props[3]
    LW_CRE = props[0]-props[4]
    #print([x.shape for x in [lat.flatten(),lon.flatten(),Time.flatten(), wap.flatten(),SW_CRE.flatten(), LW_CRE.flatten()]],flush=True)
    stack = np.stack([lat.flatten(),lon.flatten(),Time, wap.flatten(),SW_CRE.flatten(), LW_CRE.flatten()],axis=-1)
    return pd.DataFrame(stack,columns=["lat","lon","time","wap","sw_cre","lw_cre"])

def CREwap_join_maker(thresh):
    """depending on the cloud top determination loads different datasets and makes
    the joint dataframes"""
    if "e7" in thresh:
        folder = "full"
    elif "cod" in thresh and "p2" not in thresh:
        folder = "threshcod"
    elif "p2" in thresh:
        folder = "threshcodp2"
    else:
        raise NotImplementedError("whaddaya want")

    ls = glob.glob(os.path.join(scratch,"ICON_output/{}/numpy/*npz".format(folder)))
    assert len(ls)>0,os.listdir(os.path.join(scratch,"ICON_output/{}/numpy/".format(folder)))
    pool=mlp.Pool(40)
    stacks = pool.map(npz_extractor,ls)
    df = pd.concat(stacks)
    df = df.groupby(["lat","lon","time"]).mean()
    pred_df = pd.read_parquet(os.path.join(work,"frames/parquets/ICONframe{}100_10000_r360x180_0123459.parquet".format(thresh)))
    pred_df = pred_df.set_index(["lat","lon","time"])
    print(pred_df.head())
    print(df.head())
    df = df.join(pred_df, how="inner")
    assert len(df>0)
    return df


@timer_func
def waptsurf_mesh(rea,limit=1e5,d=""):
    """gets relative amount of cloud type in a
    sea-surface temp/ vertical velocity grid

    Args:
        rea (pandas.DataFrame): df containing the  cloud type fractions and properties by lat/lon/day
        limit (float,optional): maximum number of samples to consider per ctype. Defaults to 1e5.
        d (str, optional): qualifier for how to save. Defaults to "".
    """    
    print("waptsurf_mesh",flush=True)
    #number of colors in color gradient
    num_points = 200
    palette=sns.color_palette("colorblind")[1:]
    white = np.array([1,1,1.])
    colors2 = [np.array(rgb) for rgb in palette]
    gradients = [[x*y+white*(1-y) for y in np.linspace(0,1,num_points)] for x in colors2]
    #quantile that corresponds to the given limit
    qval = 1-limit/len(rea)
    fig, ax = plt.subplots(1,1,figsize=(12,12))
    fig2, ax2 = plt.subplots(2,4,figsize=(12,12),sharex=True,sharey=True)
    ax2=ax2.flatten()
    legend_elems=[]
    tsurf = rea.tsurf.values
    wap = rea.wap.values
    zz_mix = []

    for j,cname in tqdm(enumerate(ctnames)):        

        sample = np.argwhere((rea[cname]>rea[cname].quantile(qval)).values)
        #assert len(sample)<limit+10 and len(sample)>limit-10,(qval, len(sample),limit)
        sample=sample.squeeze()
        temp = rea.iloc[sample]
        xx,yy,zz = grid_amount(tsurf,wap,rea[cname].values)
        
        zz_mix.append(zz)
        assert not np.any(np.isnan(xx))
        assert not np.any(np.isnan(yy))
        mehs = ax2[j].pcolormesh(xx, yy, zz, cmap="Greys",vmin=0,vmax=1)
        ax2[j].set_title(cname,fontsize=15)
        ax2[j].tick_params(labelsize=15)
        textx=temp.tsurf.median()
        texty=temp.wap.median()
        ax.text(textx,texty,cname,fontsize=18,color=palette[(j % len(palette))])
        legend_elems+=[plt.Line2D([0], [0], color=palette[j % len(palette)], linewidth=3, label=cname)]
    [ax2[j].set_xlabel(u"$tsurf$  [K]", fontsize=16) for j in [4,5,6,7]]
    [ax2[j].set_ylabel(u"$\\omega_{500}$ [$\\frac{\\rm{Pa}}{\\rm{s}}$]", fontsize=16) for j in [0,4]]
    fig2.tight_layout()
    fig2.subplots_adjust(bottom=0.1, top=.97, left=0.1, right=.98,
                    wspace=0.02, hspace=0.1)
    cbax = fig2.add_axes([0.1, 0.03, .8, .008])
    fig2.colorbar(mehs,cax=cbax,orientation="horizontal")
    cbax.tick_params(labelsize=15)
    zz_mix = np.stack(zz_mix)
    zz_mix = np.where(zz_mix == np.max(zz_mix,0),zz_mix,np.nan)
    for j,cname in enumerate(ctnames):
        mesh = ax.pcolormesh(xx,yy,zz_mix[j],cmap=Cmap(gradients[j]))
    ax.plot([200,350],[0,0],"--w")
    ax.set_ylim(np.min(yy), np.max(yy))
    ax.set_xlim(np.min(xx), np.max(xx))
    #ax.set_xlim(xmin, xmax)
    #ax.set_ylim(ymin, ymax)
    fig.legend(handles=legend_elems, fontsize=18, ncol=4, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(work,"stats/ICONwtsallinone_mesh{}.pdf".format(d)))
    fig2.savefig(os.path.join(work,"stats/ICONwtsallseperate_mesh{}.pdf".format(d)))

def agg_calendar(df,agg="day"):
    df=df.reset_index()
    df["month"] = df.time.map(lambda x: (datetime.fromisoformat("1970-01-01")+timedelta(days=x)
                                            ).month)
    df["day"] = df.time.map(lambda x: (datetime.fromisoformat("1970-01-01")+timedelta(days=x)
                                        ).timetuple().tm_yday)
    df=df.reset_index().groupby(["lat","lon",agg]).agg("mean")
    df=df.drop(["index","time"],1)
    print(len(df))
    return df

if __name__=="__main__":
    work = os.environ["WORK"]
    scratch = os.environ["SCR"]
    ctnames =["Ci","As","Ac","St","Sc","Cu","Ns","Dc"]
    rng = np.random.default_rng(32)
    thresh = "_threshcodp2"
    if os.path.exists(os.path.join(scratch,"ICON_output/CREwapframe{}.parquet".format(thresh))):
        df = pd.read_parquet(os.path.join(scratch,"ICON_output/CREwapframe{}.parquet".format(thresh)))
    else:
        df = CREwap_join_maker(thresh)
        df.to_parquet(os.path.join(scratch,"ICON_output/CREwapframe{}.parquet".format(thresh)))

    _l=len(df)
    if False:
        df=agg_calendar(df)
        thresh+="_day"
    elif True:
        df=agg_calendar(df,"month")
        thresh+="_month"
    print(df.head())
    df.sw_cre*=-1
    df.lw_cre*=-1
    #rea = rea[rea.sw<0]
    print("filtered out",(1-len(df)/_l))
    df["clear"]=1-df[ctnames].sum(1)
    df = df[df.clear<0.4]
    print(df.columns)
    df = df[df.tsurf>275]
    count = df.groupby("lat").count()
    fig,ax = plt.subplots()
    ax.bar(x=count.index.values,height=count.clear.values)
    print(count.head())
    fig.tight_layout()
    fig.savefig(os.path.join(work,"stats/lathist.pdf"))
    allinone_kde(df,"lw_cre","sw_cre","ICONCRE",limit=min(len(df),8e4),bands =False,d=thresh)
    allinone_kde(df,"lw_cre","sw_cre","ICONCRE",limit=min(len(df),8e4),bands =True,d=thresh)
    waptsurf_mesh(df,limit=min(len(df),5e4))
     
