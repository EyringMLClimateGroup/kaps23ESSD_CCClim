"""for process based analysis of the CCClim data with
SST and WAP from ERA5
"""
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
from matplotlib.colors import ListedColormap as Cmap
import seaborn as sns
from tqdm import tqdm
from src.utils import timer_func, grid_amount,allinone_kde

def tpl_argmax(arr):
    """the tupel can be used for useful indexing, because the argmax result is raveled. 
    Then use that as indices to 2D field"""
    assert len(arr.shape)==2,"must be 2D"
    arr2 = np.where(np.isnan(arr),-1e3,arr)
    return (np.argmax(arr2)//arr2.shape[0],np.argmax(arr2)%arr2.shape[1])

@timer_func
def allinone_mesh(rea,limit=1e5,d=""):
    """plots the amounts of cloud type per SST/wap into a grid,
        both all types into one grid and a plot per ctype

    Args:
        rea (_type_): data
        limit (_type_, optional): number datapoints to compute percentile. Defaults to 1e5.
        d (str, optional): filename extension. Defaults to "".
    """    
    print("allinone_mesh",flush=True)
    num_points = 200
    white = np.array([1,1,1.])
    colors = [(.9,.6,0),(.35,.7,.9),(0,.6,.5),(.95,.9,.25),(0,.45,.7),
                (.8,.4,0),(.8,.6,.7),(0,0,0)]
    colors2 = [np.array(rgb) for rgb in colors]
    gradients = [[x*y+white*(1-y) for y in np.linspace(0,1,num_points)] for x in colors2]
    qval = 1-limit/len(rea)
    fig, ax = plt.subplots(1,1,figsize=(12,12))
    fig2, ax2 = plt.subplots(2,4,figsize=(12,12),sharex=True,sharey=True)
    ax2=ax2.flatten()
    legend_elems=[]
    sst = rea.sst.values
    wap = rea.wap.values
    zz_mix = []

    for j,cname in tqdm(enumerate(ctnames)):
        sample = np.argwhere((rea[cname]>rea[cname].quantile(qval)).values)
        sample=sample.squeeze()
        temp = rea.iloc[sample]
        xx,yy,zz = grid_amount(sst,wap,rea[cname].values)
        print(cname,"\n sstmax ", xx[tpl_argmax(zz)])
        print(cname,"wapmax ", yy[tpl_argmax(zz)],flush=True)
        
        zz_mix.append(zz)
        assert not np.any(np.isnan(xx))
        assert not np.any(np.isnan(yy))
        mehs = ax2[j].pcolormesh(xx, yy, zz, cmap="Greys",vmin=0,vmax=1)
        ax2[j].set_title(cname,fontsize=15)
        ax2[j].tick_params(labelsize=15)
        legend_elems+=[plt.Line2D([0], [0], color=colors[j % len(colors)], linewidth=3, label=cname)]
    [ax2[j].set_xlabel(u"$SST$  [K]", fontsize=16) for j in [4,5,6,7]]
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
    ax.set_xlabel(u"$SST$  [K]", fontsize=18)
    ax.set_ylabel(u"$\\omega_{500}$ [$\\frac{\\rm{Pa}}{\\rm{s}}$]", fontsize=18)
    ax.tick_params(labelsize=18)
    ax.legend(handles=legend_elems, fontsize=18, ncol=4, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(work,"stats/allinone_mesh{}.pdf".format(d)))
    fig2.savefig(os.path.join(work,"stats/allseperate_mesh{}.pdf".format(d)))

@timer_func
def allinone_mesh_byband(rea,d=""):
    """the same  as allinone_mesh but separated by latitude"""
    print("allinone_mesh_byband",flush=True)
    num_points = 200
    belts = [(0,15),(15,30),(30,60),(60,90)]
    colors = [(.9,.6,0),(.35,.7,.9),(0,.6,.5),(.95,.9,.25),(0,.45,.7),
                (.8,.4,0),(.8,.6,.7),(0,0,0)]
    white = np.array([1,1,1.])
    colors2 = [np.array(rgb) for rgb in colors]
    gradients = [[x*y+white*(1-y) for y in np.linspace(0,1,num_points)] for x in colors2]
   
    for b,(beltmin,beltmax) in enumerate(belts):
        rea2 = (rea[np.abs(rea.index.get_level_values("lat")) >= beltmin]).copy()
        rea2 = rea2[np.abs(rea2.index.get_level_values("lat")) <= beltmax]
        fig, ax = plt.subplots(1,1,figsize=(12,12))
        fig2, ax2 = plt.subplots(2,4,figsize=(12,12))
        ax2=ax2.flatten()
        qval = 0.8#1-limit/len(rea2)
        zz_mix = []
        legend_elems=[]
        for j,cname in enumerate(ctnames):
            sample = np.argwhere((rea2[cname]>rea2[cname].quantile(qval)).values)
            #assert len(sample)<limit+10 and len(sample)>limit-10,(qval, len(sample),limit)
            sample=sample.squeeze()
            assert len(sample)>0
            assert len(sample)<len(rea2)
            temp = rea2.iloc[sample].reset_index()
            xx,yy,zz = grid_amount(temp.sst.values,temp.wap.values,temp[cname].values,
                                   maxx=rea2.sst.max(),minx=rea2.sst.min(),
                                   maxy=rea2.wap.max(), miny = rea2.wap.min())
            assert rea2.sst.min()>=275,rea2.sst.min()
            assert np.min(xx)>=275, np.min(xx)
            print("\n",np.min(np.abs(temp.lat)),np.max(np.abs(temp.lat)),cname," sstmax ", xx[tpl_argmax(zz)].round(1))
            print(np.min(np.abs(temp.lat)),np.max(np.abs(temp.lat)),cname,"wapmax ", yy[tpl_argmax(zz)].round(3),flush=True)
            
            
            zz_mix.append(zz)
            assert not np.any(np.isnan(xx))
            assert not np.any(np.isnan(yy))
            mehs = ax2[j].pcolormesh(xx, yy, zz, cmap=Cmap(gradients[j]))
            ax2[j].set_title(cname)
            fig2.colorbar(mehs,ax=ax2[j],orientation="horizontal")
            textx=temp.sst.median()
            texty=temp.wap.median()
            ax.text(textx,texty,cname,fontsize=18,color=colors[(j % len(colors))])
            legend_elems+=[plt.Line2D([0], [0], color=colors[j % len(colors)], linewidth=3, label=cname)]
            
        zz_mix = np.stack(zz_mix)
        zz_mix = np.where(zz_mix == np.max(zz_mix,0),zz_mix,np.nan)
        for j,cname in enumerate(ctnames):
            mesh = ax.pcolormesh(xx,yy,zz_mix[j],cmap=Cmap(gradients[j]))
        ax.plot([200,350],[0,0],"--w")
        ax.set_ylim(np.min(yy), np.max(yy))
        ax.set_xlim(np.min(xx), np.max(xx))
        plt.figlegend(handles=legend_elems, fontsize=18, ncol=4, loc="upper left")
        fig2.tight_layout()
        fig2.savefig(os.path.join(work,"stats/allsep_mesh_belt{}{}.pdf".format(b,d)))
        fig.tight_layout()
        fig.savefig(os.path.join(work,"stats/allinone_mesh_belt{}{}.pdf".format(b,d)))

if __name__=="__main__":
    work = os.environ["WORK"]
    scratch = os.environ["SCR"]
    ctnames =["Ci","As","Ac","St","Sc","Cu","Ns","Dc"]
    rng = np.random.default_rng(32)
    day = "_day"
    
    rea= pd.read_parquet(os.path.join(scratch,"reanalysis_ocean{}.parquet".format(day)))
    rea["clear"]=1-rea[ctnames].sum(1)
    rea = rea[rea.clear<0.4]
    rea = rea[rea.sst>275]
    print("beforewapfilter", len(rea))
    rea = rea[np.abs(rea.wap)<0.2]#filters 2%, just for range of axis
    print("afterwapfilter", len(rea))
    
    print("beforewapfilter", len(rea))
    rea = rea[np.abs(rea.sst)<305]#filters <1%, just for range of axis
    print("afterwapfilter", len(rea))
    #allinone_mesh(rea,limit=min(len(rea),5e7),d=day)
    #allinone_kde(rea,"sst","wap","",limit=5e3,bands=True,d=day)
    
    allinone_mesh_byband(rea,d=day)