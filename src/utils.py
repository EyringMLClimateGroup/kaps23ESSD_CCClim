import numpy as np
from numba import jit
import time
import netCDF4 as nc4
from tqdm import tqdm
from datetime import datetime, timedelta 
import os
import dask.dataframe as dd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

work = os.environ["WORK"]
scratch = os.environ["SCR"]
ctnames =["Ci","As","Ac","St","Sc","Cu","Ns","Dc"]

def timer_func(func):
    """ This function shows the execution time of 
     the function object passed"""
    def wrap_func(*args, **kwargs):
        t1 = time.time()
        result = func(*args, **kwargs)
        t2 = time.time()
        print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s')
        return result
    return wrap_func

@jit(nopython=True)
def signif(x, p=1):
    """rounds to first significant digit (if p=1)"""
    x = np.asarray(x)
    x_positive = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10**(p-1))
    mags = 10 ** (p - 1 - np.floor(np.log10(x_positive)))
    return np.round(x * mags) / mags

def grid_amount(x,y,z):
    """gets the amount of z found at a location on a grid spanned by x,y
        output shape is dynamically determined from significant digits
        but should be kinda close to 100x100 i think

    Args:
        x (np.ndarray): x-coord
        y (np.ndarray): y-coord
        z (np.ndarray): values, same shape as x and y

    Returns:
        xx,yy np.ndarray: meshgrid coordinates
        zz, np.ndarray
    """    
    maxx=np.max(x)
    minx=np.min(x)
    maxy=np.max(y)
    miny=np.min(y)
    #adaptive scale by finding significant digits
    xscale = signif((maxx-minx)/100)
    yscale = signif((maxy-miny)/100)
    assert xscale>1e-8,"xscale"
    assert yscale>1e-8,"yscale"
    x_gr = np.arange(minx,maxx+xscale,xscale)
    y_gr = np.arange(miny,maxy+yscale,yscale)
    xx,yy = np.meshgrid(x_gr,y_gr)
    zz=sumloop(x,y,z,x_gr,y_gr)
    zz=np.where(zz<=1e-4, np.nan, zz)/np.nanmax(zz)
    return xx,yy,zz    


@jit(nopython=True)
def sumloop(x,y,z,x_gr,y_gr):
    """add the values to the grid-point closest to the actual location"""
    zz=np.zeros((len(y_gr),len(x_gr)))
    for k,(i,j) in enumerate(zip(x,y)):
        c_x = np.argmin(np.abs(x_gr-i))
        c_y = np.argmin(np.abs(y_gr-j))
        zz[c_y,c_x]+=z[k]
    return zz

    


def extract(file):
    """makes a CCClim-similar df from corresponding sst and wap files"""
    ds = nc4.Dataset(file)
    ds2 = nc4.Dataset(os.path.join(file.replace("skin_temperature","vertical_velocity").replace(
                                    "ts","wap")))
    sst = ds.variables["skt"][...]
    wap = ds2.variables["w"][...]
    
    sst = sst.reshape(-1)
    wap = wap.reshape(-1)
    lat = ds.variables["latitude"][...]
    lon = ds.variables["longitude"][...]
    lon = np.where(lon>180,lon-360,lon)
    lonlon, latlat = np.meshgrid(lon,lat)
    latlat = latlat.flatten()
    lonlon = lonlon.flatten()
    time_in_hours = ds.variables["time"][...]
    assert ~np.any(time_in_hours.mask), "masked"
    start = datetime.fromisoformat("1900-01-01")
    _fulldate = [start + timedelta(hours=int(x)) for x in time_in_hours.data]
    
     
    latlat = np.hstack([ latlat for _ in range(len(time_in_hours))])
    lonlon = np.hstack([ lonlon for _ in range(len(time_in_hours))])
    
    day = np.hstack([[x.timetuple().tm_yday  for _ in range(len(lat)*len(lon))] for x in _fulldate])
    year = np.hstack([[x.year for _ in range(len(lat)*len(lon))] for x in _fulldate])
    print("processed", flush=True)
    df = dd.from_array(np.stack([latlat, lonlon, day,year, sst, wap], axis=1).astype("float32"),
                        columns=["lat","lon","day","year","sst","wap"],chunksize=100_000)
    print("df",df.shape, flush=True)
    
    df = df.round({"lat":0,"lon":0})
    print("round", flush=True)
    df =df.groupby(["lat","lon","day"]).mean().reset_index()
    print("returning {}".format(file), flush=True)
    return df


@timer_func
def allinone_kde(rea, x_string, y_string, wherefrom, limit=1e5, bands =False,d=""):
    """plots the kernel density estimates of all ctypes into one plot
    if belts is true istead makes the plot for 4 ranges of latitude

    Args:
        rea (pandas.DataFrame): the data
        x_string, y_string : which columns to plot
        wherefrom: just a string to distinguish what is being plotted
        limit (float, optional): number of samples per ctype. Defaults to 1e5.
        bands (bool, optional): if should be broken up by latitude. Defaults to False.
        d (str, optional): saving extension. Defaults to "".
    """    
    print("allinone_kde",flush=True)
    if bands:
        belts = [(0,15),(15,30),(30,60),(60,90)]
    else:
        belts = [(0,90)]
    alph ={"Ci": 1, "As": .6, "Ac": .6, 
           "St":.6 , "Sc":.7 , "Cu":.6, 
           "Ns": 0.8, "Dc": 1.}
    colors = [(.9,.6,0),(.35,.7,.9),(0,.6,.5),(.95,.9,.25),(0,.45,.7),
                (.8,.4,0),(.8,.6,.7),(0,0,0)]
    qval = 1-limit/len(rea)
    fig, ax = plt.subplots(1,len(belts),figsize=(11+len(belts),12), sharey=True)
    if not isinstance(ax,np.ndarray):
        ax=np.array([ax])
        assert len(ax)==1,len(ax)
    print("qval ",qval ,"of total length",len(rea))
    bar = tqdm(total = len(belts)*len(ctnames),leave=True)
    if os.path.exists(os.path.join(scratch,"CRE_reanalysis.csv")):
        os.remove(os.path.join(scratch,"CRE_reanalysis.csv"))
    for b,(beltmin,beltmax) in enumerate(belts):
        rea2 = rea[np.abs(rea.index.get_level_values("lat")) >= beltmin].copy()
        rea2 = rea2[np.abs(rea2.index.get_level_values("lat")) <= beltmax]
        legend_elems=[]
        xmin = rea2.loc[:,x_string].quantile(0.01)
        xmax = rea2.loc[:,x_string].quantile(0.98)
        ymin = rea2.loc[:,y_string].quantile(0.005)
        ymax = rea2.loc[:,y_string].quantile(0.995)
        for j,cname in enumerate(ctnames):
            global_thresh = rea[cname].quantile(qval)
            if not np.any((rea2[cname]>global_thresh)):
                continue
            sample = np.argwhere((rea2[cname]>global_thresh).values)
            
            sample=sample.squeeze()
            temp = rea2.iloc[sample]
            kde = sns.kdeplot(data = temp,x=x_string,y=y_string,ax=ax[b],common_norm=False,color =colors[(j % len(colors))],
                                label=cname ,levels=np.linspace(0.7,1,6),fill=True, alpha=alph[cname])
            
            #i guess it would be nicer if the context was around the loop but hey
            with open(os.path.join(scratch,"CRE_reanalysis.csv"),"a+") as csv:
                if j==0:
                    print(",", x_string+","+y_string,file=csv)
                print("Char. {},{},{}".format(cname,   temp.loc[:,x_string].mean(),
                        temp.loc[:,y_string].mean()), file=csv )
                print("Weighted {},{},{}".format(cname,   (rea2.loc[:,x_string]*rea2.loc[:,cname]).mean(), 
                     (rea2.loc[:,y_string]*rea2.loc[:,cname]).mean()),file = csv)
            textx=temp.loc[:,x_string].median()
            texty=temp.loc[:,y_string].median()
            ax[b].text(textx,texty,cname,fontsize=18,color=colors[(j % len(colors))])
            if j==0:
                ax[b].plot([-10,1000],[10,-1000], "--k",)
                legend_elems += [plt.Line2D([0],[0],color = "k", linewidth=2, label ="SW+LW=0",linestyle="--")]
            legend_elems+=[plt.Line2D([0], [0], color=colors[j % len(colors)], linewidth=4, label=cname)]
            ax[b].set_xlim(xmin, xmax)
                
            ax[b].set_ylim(ymin,ymax)
            ax[b].set_xlabel(u"Longwave CRE $\left[\\frac{W}{m^2}\\right]$",fontsize=20)
            ax[b].tick_params(labelsize=18)
            if not (beltmin ==0 and beltmax==90):
                ax[b].set_title("Latitude Range: {}°N/S-{}°N/S".format(beltmin, beltmax))
            bar.update(1)
    ax[0].set_ylabel("Shortwave CRE $\left[\\frac{W}{m^2}\\right]$",fontsize=20)
    
    if len(belts)==1:
        ax[b].legend(handles=legend_elems, fontsize=18, ncol=3, loc="lower left")
        fig.tight_layout()
        fig.savefig(os.path.join(work,"stats/{}allinone_kde{}.pdf".format(d,wherefrom)))
    else:
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.15, top=.96, left=0.1, right=.98,
                        wspace=0.02, hspace=0.1)
        fig.legend(bbox_to_anchor=[0.1, 0.025, .8, .009],handles=legend_elems,fontsize=16,ncol=8)
        fig.savefig(os.path.join(work,"stats/{}allinone_kde_belts{}.pdf".format(d,wherefrom)))


class compare_era(object):
    """used to give the call function access to the total df in subprocesses"""
    def __init__(self,era_df,timescale):
        self.era_df = era_df
        self.timescale = timescale

    def __call__(self,year):
        """makes a joint dataframe for CCClim and ERA5 data for a single year"""
        ccclim_y = pd.read_parquet(os.path.join(scratch,
            "ESACCI/parquets/nooverlap/ESACCI_de{}frame100_10000_10_0123459.parquet".format(year)),
            columns=["lat","lon","time"]+ctnames)
        ccclim_y[self.timescale] = ccclim_y.time.map(lambda x: (datetime.fromisoformat("1970-01-01")+
                                                    timedelta(days = x)).month)
        self.era_df = self.era_df.reset_index().set_index(["lat","lon",self.timescale])
        era_y = self.era_df[self.era_df.year == year].dropna().drop("year",axis=1)
        ccclim_y = ccclim_y.groupby(["lat","lon",self.timescale]).mean().drop("time",axis=1)
        joint = ccclim_y.join(era_y)

        return joint
