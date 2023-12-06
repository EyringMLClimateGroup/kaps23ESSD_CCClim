import glob
import numpy as np
import os
import time
import traceback
import netCDF4 as nc4
import torch
from torch.utils.data import Dataset, SubsetRandomSampler
coordinates = ['latitude', 'longitude']




class LoadingError(Exception):
    pass



def get_avg_all(dataset, allowed_idx=None, ext="npy", random=True):

    paths = np.arange(len(dataset.file_paths))

    if allowed_idx is not None:
        paths = [paths[i] for i in allowed_idx]
    
    if random:
        return SubsetRandomSampler(paths)
    else:
        return paths

def window_nd(a, window, steps = None, axis = None, gen_data = False):
        """
        Create a windowed view over `n`-dimensional input that uses an 
        `m`-dimensional window, with `m <= n`
        
        Parameters
        -------------
        a : Array-like
            The array to create the view on
            
        window : tuple or int
            If int, the size of the window in `axis`, or in all dimensions if 
            `axis == None`
            
            If tuple, the shape of the desired window.  `window.size` must be:
                equal to `len(axis)` if `axis != None`, else 
                equal to `len(a.shape)`, or 
                1
            
        steps : tuple, int or None
            The offset between consecutive windows in desired dimension
            If None, offset is one in all dimensions
            If int, the offset for all windows over `axis`
            If tuple, the steps along each `axis`.  
                `len(steps)` must me equal to `len(axis)`
    
        axis : tuple, int or None
            The axes over which to apply the window
            If None, apply over all dimensions
            if tuple or int, the dimensions over which to apply the window

        gen_data : boolean
            returns data needed for a generator
    
        Returns
        -------
        
        a_view : ndarray
            A windowed view on the input array `a`, or `a, wshp`, where `whsp` is the window shape needed for creating the generator
            
        """
        ashp = np.array(a.shape)
        
        if axis != None:
            axs = np.array(axis, ndmin = 1)
            assert np.all(np.in1d(axs, np.arange(ashp.size))), "Axes out of range"
        else:
            axs = np.arange(ashp.size)
            
        window = np.array(window, ndmin = 1)
        assert (window.size == axs.size) | (window.size == 1), "Window dims and axes don't match"
        wshp = ashp.copy()
        wshp[axs] = window
        assert np.all(wshp <= ashp), "Window is bigger than input array in axes"
        
        stp = np.ones_like(ashp)
        if steps:
            steps = np.array(steps, ndmin = 1)
            assert np.all(steps > 0), "Only positive steps allowed"
            assert (steps.size == axs.size) | (steps.size == 1), "Steps and axes don't match"
            stp[axs] = steps
    
        astr = np.array(a.strides)
        
        shape = tuple((ashp - wshp) // stp + 1) + tuple(wshp)
        strides = tuple(astr * stp) + tuple(astr)
        as_strided = np.lib.stride_tricks.as_strided
        a_view = np.squeeze(as_strided(a, 
                                     shape = shape, 
                                     strides = strides))
        if gen_data :
            return a_view, shape[:-wshp.size]
        else:
            return a_view





def get_coords(nc_file):
    """gets latitude and longitude in  that order"""
    file = nc4.Dataset(nc_file, "r", format="NETCDF4")
    coords = np.vstack([file.variables[name][:] for name in coordinates])
    return coords



def ICON_collate(arr):
    #stack both filenames and output
    arr=np.stack(arr)
    #ignore the filename move batch size back
    filename, arr, locations = arr.transpose()
    #now batch size is in front
    ##stack along some axis that is not the first
    
    arr = np.hstack(arr)
    arr = arr.reshape(arr.shape[0],-1).transpose()
    filename = np.hstack(filename)
    locations = np.hstack(locations)
    locations = locations.reshape(locations.shape[0],-1).transpose()
    #pixelwise and for every variable
    return filename,torch.tensor(arr), torch.tensor(locations)


class ICONDataset(Dataset):
    def __init__(self, root_dir,grid, normalizer=None, 
                  indices=None, variables = None,
                   output_tiles=True,tilesize=3,
                 transform=None):
        """
        

        Parameters
        ----------
        root_dir : str
            location of transformed ICON npy files
        grid : str
            grid the files are regridded to
        normalizer : function or method, optional
            function that is applied to the inputs The default is None.
        indices : np.ndarray (n,), optional
            loads n files indexed this way. The default is None.
        variables : np.ndarray (v,), optional
            indices of variables to be loaded. The default is None.
        output_tiles : Bool, optional
            if tiles of cell averages are to be returned The default is True.
        transform : str, optional
            something like "log". The default is None.
        Returns
        -------
        None.

        """
        self.grid=grid 
        self.transform=transform
        self.root_dir = root_dir
        self.output_tiles = output_tiles
        if self.output_tiles:
            self.tilesize=tilesize
        else:
            self.tilesize=None
        
        self.file_paths = glob.glob(os.path.join(root_dir, "{}*.npz".format(grid)))
        if indices is not None:
            try:
                self.file_paths = [self.file_paths[i] for i in indices]
            except IndexError as e:
                print(self.file_paths)
                raise e
            
        assert len(self.file_paths)>0      
        
        self.normalizer = normalizer
        # this allows the same variable indexing for ICON as for CUMULO without bothering with the source files
        variable_translator = {"cwp":0,"lwp":1, "iwp":2,"cerl":3, "ceri":4,
                                "ptop":5,"tsurf":6, "cod":7,"clt":14}
        self.variables = np.array([variable_translator[i] for i in variables])
        

        
        
    def __len__(self):

        return len(self.file_paths)

    def __getitem__(self, info):

        if isinstance(info, tuple):
            # load single tile
            idx, *tile_idx = info
        else:
            idx, tile_idx = info, None

        
        filename = self.file_paths[idx]
        
        
        failsafe =0
        while True:
            try:
                #this should be [5,96,192]
                #wehre 5=[ps, cllvi, clivi, topmax, ts]
                properties = np.load(filename)
                locations = properties["locations"] 
                properties = properties["properties"]
                
                break
            
            except Exception as err:
                traceback.print_exc()
                time.sleep(5)
                failsafe+=1
                if failsafe>120:
                    raise LoadingError("loading timed out")
                    
        properties = np.vstack((properties[np.newaxis,0]+properties[np.newaxis,1],
                                properties))
        properties[5] /= 100 #from Pa to hPa
        properties[:3] *= 1000 # form kg/m^2 to g/m^2
        properties[:3] *= properties[0]>1e-5
        properties[2] *= properties[2]>1e-5
        properties[1] *= properties[1]>1e-5
        assert np.mean(properties[[7,8]]<200),(np.mean(properties,(1,2)),filename)
        properties[[7,8]] = np.where(properties[[7,8]]>2000,2000,properties[[7,8]]) # clip stupidly large cod values
        assert np.all(properties[~np.isnan(properties)]<=20000), (np.where(properties>20000),filename)
        tiles_mean = properties[ self.variables]
        if self.normalizer is not None:
            tiles_mean = self.normalizer(tiles_mean)
        
                                

        del properties
        ignore = np.any(np.isnan(tiles_mean) | np.isinf(tiles_mean), 0)
        tiles_mean[:, ignore]=0
        
 
        if self.output_tiles:
            
            temp = window_nd(tiles_mean, self.tilesize,self.tilesize, axis=(1,2)).copy()

            tiles_mean= np.copy(temp)

            locations = np.copy(window_nd(locations, self.tilesize, self.tilesize, axis=(1,2)))

            del temp

            locations = locations.reshape(-1, locations.shape[2],locations.shape[3],

                                           locations.shape[4])

            tiles_mean = tiles_mean.reshape(-1, tiles_mean.shape[2],

                                              tiles_mean.shape[3],

                                              tiles_mean.shape[4])


            tiles_mean= np.copy(temp)
            del temp
        
        if self.transform == "log":
            tiles_mean=np.log(tiles_mean+1)
            

        self.locations = None
        
        tiles_mean = tiles_mean.astype(np.float32)
        
        if tile_idx is not None:
            x,y=tile_idx
            if self.output_tiles:
                o_s = self.tilesize//2
                return (filename, tiles_mean[x,y], locations[x,y])
            else:
                return(filename, tiles_mean[:,x,y], locations[:,x,y])
        else:
            if self.output_tiles:
                o_s = self.tilesize//2
                return filename, tiles_mean, locations
            return filename, tiles_mean, locations


    def __str__(self):
        return 'ICONSet'

    