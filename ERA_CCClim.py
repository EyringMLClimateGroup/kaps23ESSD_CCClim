"""same as ERA_CCClim_daily.py but makes monthly means
why i copied this into a second script...?
i must have had my reasons"""
import numpy as np
import os
import glob
import pandas as pd
from multiprocessing import Pool
import global_land_mask as globe
from src.utils import extract,compare_era
import dask.dataframe as dd

if __name__=="__main__":
    work = os.environ["WORK"]
    scratch = os.environ["SCR"]
    wap_files = glob.glob(os.path.join(scratch,"ERA5/wap/*"))
    ts_files = glob.glob(os.path.join(scratch,"ERA5/ts/*"))
    ctnames = ["Ci","As","Ac","St","Sc","Cu","Ns","Dc"]
    spec = "cropped" 
    
    if not os.path.exists(os.path.join(scratch,"ERA5.parquet")):
        print("making new parquet file")
        pool=Pool(30)
        dfs = pool.map(extract, ts_files[:]    )    
        df = dd.concat(dfs).compute()
        df = df.groupby(["lat","lon","year","month"])
        df = df.mean()
        df.to_parquet(os.path.join(scratch,"ERA5.parquet"))
    else:
        print("comparing ERA and CCClim")
        era_df = pd.read_parquet(os.path.join(scratch,"ERA5.parquet"))
        
        compare = compare_era(  era_df, "month",spec)
        pool=Pool(10)

        joints = pool.map(compare,np.arange(1983,2016))
        joints =pd.concat(joints).reset_index()
        joints_land = joints.loc[globe.is_land(joints.lat,joints.lon),:]
        joints_land = joints_land.set_index(["lat","lon","month"])
        joints_land.to_parquet(os.path.join(scratch,"reanalysis_land{}.parquet".format(spec)))
        print(len(joints), len(joints_land))
        del joints_land
        joints_ocean = joints.loc[globe.is_ocean(joints.lat,joints.lon),:]
        joints_ocean = joints_ocean.set_index(["lat","lon","month"])
        print(len(joints), len(joints_ocean))
        joints_ocean.to_parquet(os.path.join(scratch,"reanalysis_ocean{}.parquet".format(spec)))
            
