"""
creates the actual ESACCI dataset from the parquet files produces by the ML pipeline
"""
import pandas as pd
import netCDF4 as nc4
import os
import glob
import time
import warnings
import traceback
import numpy as np
from tqdm import tqdm
import multiprocessing as mlp
from datetime import datetime
import sys

ctnames = [ "Ci", "As", "Ac", "St", "Sc", "Cu", "Ns", "Dc"]
propnames = ['twp', 'lwp', 'iwp', 'cerl', 'ceri', 'cod', 'ptop', 'tsurf','clt']
            #"rlut", "rsut", "rsutcs", "rlutcs"]
units = ["kg/m2","kg/m2","kg/m2","m","m","1","Pa","K","1"]#,"1","1","1","1"]
clear = ["clear"]
s_names = ["undetermined_cloud_type_or_clear_sky",
           "cirrus_or_cirrostratus", "altostratus", "altocumulus",
            "stratocumulus", "stratus", "cumulus",
            "nimbostratus", "deep_convective",
            "atmosphere_mass_content_of_cloud_condensed_water", "atmosphere_mass_content_of_cloud_liquid_water", "atmosphere_mass_content_of_cloud_ice",
            "effective_radius_of_cloud_liquid_water_particles",
            "effective_radius_of_cloud_ice_particles",
            "atmosphere_optical_thickness_due_to_cloud",
            "air_pressure_at_cloud_top",
            "surface_temperature", 
            "cloud_area_fraction",
            "toa_upwelling_longwave_flux",
            "toa_upwelling_shortwave_flux", 
            "toa_upwelling_shortwave_flux_assuming_clear_sky",
            "toa_upwelling_longwave_flux_assuming_clear_sky"]

l_names = ["predicted relative frequency of occurence of the 'undetermined' type",
           "predicted relative frequency of occurence of the 'Ci' type", 
           "predicted relative frequency of occurence of the 'As' type",
             "predicted relative frequency of occurence of the 'Ac' type",
            "predicted relative frequency of occurence of the 'St' type",
              "predicted relative frequency of occurence of the 'Sc' type", 
              "predicted relative frequency of occurence of the 'Cu' type",
            "predicted relative frequency of occurence of the 'Ns' type",
             "predicted relative frequency of occurence of the 'Dc' type",
            "vertically integrated cloud water", 
            "vertically integrated liquid cloud water",
              "vertically integrated cloud ice",
            "effective radius of cloud liquid droplets",
            "effective radius of cloud ice particles",
            "cloud optical thickness",
            "cloud top pressure",
            "surface temperature", "2D cloud cover",
            "all-sky top of atmosphere longwave upwelling solar radiation",
            "all-sky top of atmosphere shortwave upwelling solar radiation",
            "top of atmosphere shortwave upwelling solar radiation assuming clear sky",
            "top of atmosphere longwave upwelling solar radiation assuming clear sky"]

def dayproc(data,d,day):
    """computes the daily 1°x1° averages and assigns them to the correct boxes in the output

    Args:
        data (pd.Dataframe): CCClim dataframe with up to 2 daily values
        d (int): the dth nonezero entry
        day (daiy): calendar day

    Returns:
        tuple: (index, properly managed daily averages)
    """    
    data = data.groupby(["lat","lon","time"]).mean().reset_index()
    temp = data.set_index("time")
    temp = temp.loc[day]
    
    temp = temp.set_index(["lat","lon"]).unstack()
    out = []
    for c,col in enumerate(clear+ctnames+propnames):
        vals = temp.loc[:,col].values
        #this handles if there are missing grid boxes
        if not np.all(vals.shape==(180,360)):
            nanframe=pd.DataFrame(index=np.arange(-90,90,1.),columns=np.arange(-180,180,1.))
            nanframe.update(temp.loc[:,col])
            vals = nanframe.values
        out.append(vals)
    out = np.stack(out,axis=-1)
    return d,out

def assign(ds,vals,shift,j,prop):
    """puts the data in the nc file with correct name and units

    Args:
        ds (nc4.Dataset): ouput dataset
        vals (np.ndarray): output data
        shift (int): units and names and stuff dont all have the same length
        j (int): index of property
        prop (string): name of property
    """    
    shifted_j = j+shift
    arr = vals[...,shifted_j]
    temp = ds.createVariable(prop,"f4",("time","lat","lon",))
    temp.standard_name = s_names[shifted_j]
    temp.long_name = l_names[shifted_j]
    
    if prop in propnames:
        temp.units = units[j]
        if units[j]=="kg/m2":
            arr/=1000
        elif units[j]=="Pa":
            arr*=100
        elif units[j]=="m":
            arr/=1e6
        
    else:
        temp.units= "1"
    temp[:] = arr

def modname(inname):
    """modifies the detailed input data name to the effective output data"""
    return inname.replace("parquet", "nc").replace(
                            "ESACCI_de","CCClim_").replace("frame100_10000_10_0123459","")

class Dayproc(object):
    """ensures that I can use dayproc with variable ancillary inputs in subprocesses"""
    def __init__(self, df):
        self.df = df
    def __call__(self, x,y):
        return dayproc( self.df,x,y)

def exists(path,outfolder):
    """shorthand to check wether the CCClim file for the day already exists"""
    bname = os.path.basename(path)
    outpath = os.path.join(outfolder,modname(bname))
    return os.path.exists(outpath)
    
def main(year=None):
    """easy way to call the pipeline for each individual year

    Args:
        year (int, optional): If not specified does everything, which is possible I guess but takes ages . Defaults to None.
    """    
    scratch = os.environ["SCR"]
    infolder = os.path.join(scratch, "ESACCI/parquets/w_anc")
    outfolder = os.path.join(scratch,"ESACCI","CCClim")
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
        
    files = glob.glob(os.path.join(infolder,"ESACCI_de????f*.parquet"))[:]
    files = [x for x in files if not exists(x,outfolder)]
    if year is not None:
        files = [x for x in files if year in x]
    assert len(files)>0
    for file in tqdm(files,total=len(files)):
        name = os.path.basename(file)
        
        outds = nc4.Dataset(os.path.join(outfolder,modname(name)),
                            "w",format = "NETCDF4")
        time_out = outds.createDimension("time", None)
        lat_out = outds.createDimension("lat", 180)
        lon_out = outds.createDimension("lon", 360)

        
        outds.title = "CCClim" 
        outds.comment = "Machine learning powered Cloud Class Climatology based on ESA Cloud_cci" 
        outds.Conventions = "CF-1.10" 
        outds.institution = "Deutsches Zentrum für Luft und Raumfahrt - Institut für Physik der Atmosphäre"
        outds.source = "satellite observations"

        times_out = outds.createVariable("time","f4",("time",))

        lats_out = outds.createVariable("lat","f4",("lat",))
        lons_out = outds.createVariable("lon","f4",("lon",))
        times_out.units = "days since 1970-01-01"
        lats_out.units = "degrees_north"
        lons_out.units = "degrees_east"
        times_out.standard_name = "time"
        lons_out.standard_name = "longitude"
        lats_out.standard_name = "latitude"
        lats_out[:]=np.arange(-90,90,1.)
        lons_out[:] = np.arange(-180,180,1.)
        
        data = pd.read_parquet(file)
        if "2010" in file:
            #exclude july 2010 because that has wrong data
            start = datetime.fromisoformat("1970-01-01")
            start_july = datetime.fromisoformat("2010-07-01")
            end_july = datetime.fromisoformat("2010-07-31")
            ex_july = (data.time>(end_july-start).days)|( data.time<(start_july-start).days)
            data=data[ex_july]
        elif "1994" in file:
            #exclude last quarter of 1994 because weird/missing data
            start = datetime.fromisoformat("1970-01-01")
            start_january = datetime.fromisoformat("1994-01-01")
            ex_260 = data.time- (start_january-start).days < 260
            data=data[ex_260]
        elif "1986" in file:
            #exclude first january
            start = datetime.fromisoformat("1970-01-01")
            start_january = datetime.fromisoformat("1986-01-01")
            ex_11 = data.time!=(start_january-start).days
            data=data[ex_11]

        un_time = data.time.unique()
        un_time.sort()
        
        times_out[:] = un_time
        pool=mlp.Pool(10)
        # I dont know why i carry the enumerate index around here but there must be a reason
        outs = pool.starmap(Dayproc(data),list(enumerate(un_time)))
        
        while True:
            #you could also do this i tink: out= [x for x in out if x is not None]
            try:
                outs.remove(None)
            except Exception:
                break

        outs.sort()
        base=np.stack([x for _,x in outs])
        print(base.shape)
        for j, prop in enumerate(propnames):
            assign(outds,base,len(clear+ctnames),j,prop)
        for j,cname in enumerate(clear+ctnames):
            assign(outds,base,0,j,cname)



        outds.close()
if __name__=="__main__":
    try:
        year=sys.argv[1]
    except IndexError:
        year=None
    main(year = year)
