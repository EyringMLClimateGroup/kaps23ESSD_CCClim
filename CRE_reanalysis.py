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
from src.utils import grid_amount, timer_func,allinone_kde
 


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
    lw = rea.lw.values
    sw = rea.sw.values
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
        textx=temp.lw.median()
        texty=temp.sw.median()
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
    fig.savefig(os.path.join(work,"stats/CREallinone_mesh{}.pdf".format(d)))
    fig2.savefig(os.path.join(work,"stats/CREallseperate_mesh{}.pdf".format(d)))


@timer_func
def specific_kde(rea,cname,p1,p2,limit=1e5,d=""):
    """gets KDE plot of cloud type *cname* in the space of the properties *p1* and *p2*

    Args:
        rea (_type_): dataframe with all the properties and cloud types
        cname (string): column in rea
        p1 (string): column in rea
        p2 (string): column in rea
        limit (float, optional): max num samples. Defaults to 1e5.
        d (str, optional): suffix. Defaults to "".
    """    
    print("specific_kde",flush=True)
    cnames = ["Ci","As","Ac","St","Sc","Cu","Ns","Dc"]
    j=cnames.index(cname)
    palette=sns.color_palette()[1:]
    qval = 1-limit/len(rea)
    fig, ax = plt.subplots(1,1,figsize=(12,12))
    legend_elems=[]
    sample = np.argwhere((rea[cname] < rea[cname].quantile(qval)).values)
    sample=sample.squeeze()
    temp = rea.iloc[sample]
    kde = sns.kdeplot(data = temp,x=p1,y=p2,ax=ax,common_norm=True,color =palette[(j % len(palette))],
                        label=cname ,levels=np.linspace(0.01,1,6),fill=True, alpha=0.7)
    ax.set_title("{} vs {} for high {} content".format(p1,p2,cname))
    legend_elems+=[plt.Line2D([0], [0], color=palette[j % len(palette)], linewidth=3, label=cname)]
    #ax.plot([200,350],[0,0],"--k")
    ax.set_ylim(temp[p2].min(),temp[p2].max())
    ax.set_xlim(temp[p1].min(),temp[p1].max())
    plt.figlegend(handles=legend_elems, fontsize=18, ncol=4, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(work,"stats/CREspec{}_{}_{}{}.pdf".format(p1,p2,cname,d)))


if __name__=="__main__":
    work = os.environ["WORK"]
    scratch = os.environ["SCR"]
    ctnames =["Ci","As","Ac","St","Sc","Cu","Ns","Dc"]
    rng = np.random.default_rng(32)
    new_or_day = "_month"
    rea= pd.read_parquet(os.path.join(scratch,
                                      "full_CRE_join{}.parquet".format(new_or_day)))
    _l=len(rea)
    #rea = rea[rea.sw<0]
    rea["clear"]=1-rea[ctnames].sum(1)
    #these filters might not be those used for the published plots
    rea = rea[rea.clear<0.4]
    rea = rea[rea.sst>275]
    print("filtered out",(1-len(rea)/_l),_l,len(rea))
    
    count = rea.groupby("lat").count()
    fig,ax = plt.subplots()
    ax.bar(x=count.index.values,height=count.clear.values)
    print(count.head())
    fig.tight_layout()
    #just to check where the samples are from
    fig.savefig(os.path.join(work,"stats/lathist.pdf"))
    
    allinone_kde(rea,"lw","sw","CRE",limit=min(len(rea),5e4),bands =False,d=new_or_day)
    allinone_kde(rea,"lw","sw","CRE",limit=min(len(rea),5e4),bands =True,d=new_or_day)
    
    allinone_mesh(rea,limit=min(len(rea),5e4),d=new_or_day)
    
