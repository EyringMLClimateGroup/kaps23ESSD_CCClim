"""makes a joint dataframe of cloud types from CCClim ant sea-surface temp and vertical velocity from ERA5
it seems to me now that this is super inefficient, not to inlcude the whole CCClim data here already
but it works so..."""
import numpy as np
import os
import glob
import pandas as pd
from multiprocessing import Pool
import global_land_mask as globe
import dask.dataframe as dd
from tqdm import tqdm
from src.utils import extract,compare_era



if __name__=="__main__":
    work = os.environ["WORK"]
    scratch = os.environ["SCR"]
    #wap_files = glob.glob(os.path.join(scratch,"ERA5/wap/*"))
    ts_files = glob.glob(os.path.join(scratch,"ERA5/daily/ts/*"))
    ts_files.sort()
    ctnames = ["Ci","As","Ac","St","Sc","Cu","Ns","Dc"]
    
    if not os.path.exists(os.path.join(scratch,"ERA5_day.parquet")):
        print("making new parquet file")
        incr=5
        pool=Pool(incr)
        
        for sub_ids in tqdm(range(0,len(ts_files),incr)):
            dfs = pool.map(extract, ts_files[sub_ids:sub_ids+incr]    )    
        
            df = dd.concat(dfs)
            del dfs
            df = df.groupby(["lat","lon","year","day"]).mean()
            df.to_parquet(os.path.join(scratch, "ERA5_day.parquet"),append=True)
            del df
    else:
        print("putting together ERA and CCClim")
        era_df = pd.read_parquet(os.path.join(scratch,"ERA5_day.parquet"))
  
        compare = compare_era(  era_df,"day")
        pool=Pool(10)

        joints = pool.map(compare, np.arange(2000,2002))
        joints = [x for x in joints if x is not None]
        joints = [x for x in joints if len(x)>0]
        joints = pd.concat(joints).reset_index()
        joints_land = joints.loc[globe.is_land(joints.lat,joints.lon),:]
        joints_land = joints_land.set_index(["lat","lon","day"])
        joints_land.to_parquet(os.path.join(scratch,"reanalysis_land_day.parquet"))
        print(len(joints), len(joints_land))
        del joints_land
        joints_ocean = joints.loc[globe.is_ocean(joints.lat,joints.lon),:]
        joints_ocean = joints_ocean.set_index(["lat","lon","day"])
        print(len(joints), len(joints_ocean))
        joints_ocean.to_parquet(os.path.join(scratch,"reanalysis_ocean_day.parquet"))
            
