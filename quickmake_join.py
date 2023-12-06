"""
makes joint dataframes of CCClim and ERA5
"""
import numpy as np
import os
import glob
import seaborn as sns
import pandas as pd
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
import multiprocessing as mlp
import dask.dataframe as dd




def get_npz(file):
    """loads a file and turns it into a  dataframe"""
    a=np.load(file)
    l=a["locs"]
    p=a["props"]
    
    cre = np.stack([p[-2]-p[-6], p[-3]-p[-5]]).reshape(2,-1)
    isnight = p[-4].reshape(1,-1)<0.1#check if sw-downflux is basically vanishing
    #i checked and the filtering definitely works. filters out occasions where sw cre is basically 0
    cre[1][isnight.squeeze()]=np.nan
    choices = np.array([0,1,2,3,4,5,6,10])
    p = p[choices].reshape(8,-1)
    
    stack = pd.DataFrame(np.vstack([l.reshape(3,-1),p,cre]).transpose(),
                            columns=["lat","lon","time","cwp","lwp","iwp",
                                                                    "cerl","ceri","cod", "ptop","tsurf",
                                                                    "lw","sw"],dtype="float64")
    stack = stack.round({"lat":0,"lon":0})
    stack = stack.groupby(["lat","lon","time"]).mean().reset_index().astype("float16")
    return stack

if __name__=="__main__":
    ctnames = ["Ci","As","Ac","St","Sc","Cu","Ns","Dc"]
    scratch = os.environ["SCR"]
    pool=mlp.Pool(70)
    base_outname = "CRE"
    #this can really take forever for full CCClim so its done in stages
    if not os.path.exists("{}.parquet".format(base_outname)):
        files = glob.glob(os.path.join(scratch,"ESACCI/npz_daily/*npz"))
        choices = None#np.random.choice(np.arange(len(files),dtype=int),size=100)
        if choices is not None:
            dflist = pool.map(get_npz,[files[i] for i in choices])
        else:
            dflist = pool.map(get_npz,files)
        df = pd.concat(dflist)
        
        del dflist
        df=df.groupby(["lat","lon","time"]).mean().reset_index().astype("float32") 
        df.to_parquet(os.path.join(scratch, "{}.parquet".format(base_outname)))

    if not os.path.exists(os.path.join(scratch,"{}_join.parquet".format(base_outname))):
        df=pd.read_parquet(os.path.join(scratch,"{}.parquet".format(base_outname)))
        #parquet file from ERA_CCClim_day.py
        rea = pd.read_parquet(os.path.join(scratch, "reanalysis_ocean_day.parquet"))
        df["month"] = df.time.map(lambda x: (datetime.fromisoformat("1970-01-01")+timedelta(days=x)).month)
        df["day"] = df.time.map(lambda x: (datetime.fromisoformat("1970-01-01")+timedelta(days=x)).timetuple().tm_yday)

        
        print(rea.head())
        df = dd.from_pandas(df, chunksize=1_000_000)
        rea = dd.from_pandas(rea.reset_index(), chunksize=1_000_000)
        
        join = rea.join(df,how="inner").dropna("all")

        print(df.head())
        print(rea.head())
        print(join.head())
        print(len(join))
        assert len(join)<=min(len(rea),len(df)),(len(df),len(rea))
        join.to_parquet(os.path.join(scratch, "{}_join.parquet".format(base_outname)))
    else:
        join = pd.read_parquet(os.path.join(scratch, "{}_join.parquet".format(base_outname)))

    join["clear"]=1-join[ctnames].sum(1)
    cond =  join[ctnames]>0.1#+join["clear"].values.reshape(-1,1))>0.4 
    join[ctnames] = join[ctnames].where(cond, other=np.nan)
    join = join[join.clear<0.4]
    cond = (join.sst.values>275).reshape(-1,1)
    join= join.iloc[cond,:]
   
    print(join.head())
    rng=np.random.default_rng()
    ids = rng.choice(np.arange(len(join)),size=min(int(5e5),len(join)),replace=False)
    pp = sns.pairplot(data=join.iloc[ids],kind="kde",x_vars=ctnames,
            y_vars=["sst","wap","lw","sw"], plot_kws={"levels": np.linspace(0.001,1,10),"fill":True})
    locs = join.reset_index()[["lat","lon","month"]]
    
    fig=plt.gcf()
    fig.savefig(os.path.join(scratch, "short_CRE.png"))
    lc=locs.hist()
    fig=plt.gcf()
    fig.savefig("lochist_dn_half.png")
