This repository contains the code used to create the CCClim (Cloud Class Climatology) dataset and the plots used for the associated analysis submitted to Earth System Science Data. CCClim provides cloud class distributions from the ESA Cloud_cci AVHRR cloud dataset, predicted using a machine-learning framework developed in a previously published article.  
Author: Arndt Kaps (arndt.kaps@dlr.de)  


CCClim is publicly available and can be downloaded from [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8369202.svg)](https://zenodo.org/records/8369202). The accompanying paper "Characterizing clouds with the CCClim dataset, as machine learning cloud class climatology" is available as a preprint (https://doi.org/10.5194/essd-2023-424). This code is released as *otherzenodobadge*. The code used to obtain the trained Random Forest used here is published under [![DOI](https://zenodo.org/badge/530182569.svg)](https://zenodo.org/badge/latestdoi/530182569), with the corresponding proof-of-concept paper being published in Transactions in Geoscience and Remote Sensing (https://doi.org/10.1109/TGRS.2023.3237008).  

## The workflow is as follows:  

To be able to run this code, first install the required packages with:  
```
conda create -n CCClim python=3.11 pytorch pytorch-cuda=11 dask distributed matplotlib=3.8 seaborn=0.12 graphviz scikit-learn=1.3 cartopy pandas=2.1 numba h5py=3.10 netCDF4 -c pytorch -c nvidia
conda activate CCClim
pip install tqdm prefetch-generator global-land-mask
``` 
Have a trained RF, for example by running the repo mentioned above. Download the ESA-CCI AVHRR data from https://public.satproj.klima.dwd.de/data/ESA_Cloud_CCI/CLD_PRODUCTS/v3.0/L3U/. The cloud properties, cloud Mask, and radiative properties are required. Then run `extract_ESACCI.py`, which produces the `.npz` files.  

Now `predict_ESACCI.py` can use the RF to predict the cloud types. This should be done by year to end up with a `.parquet`file for each year.  

`parq_to_nc.py` then creates the `netCDF`files that constitute the CCClim release. All analysis is done using the `.parquet` files because that can be parallelized with dask.  

In the paper, we compared CCClim to cloud type distributions predicted using the same RF from ICON-A simulation output. `ICON_pipeline.py` creates the `.npz` files required for `sequential_ICON_predict.py` to create the corresponding cloud class predictions.  
Most of the analysis is done with `timewise_ESACCI.py` which requires a running dask scheduler. It plots the Piecharts (see below), the time series, the typical seasonal cycles, the areas of most increased cloud types and the characteristic cell analysis.  
<img src="https://github.com/EyringMLClimateGroup/kaps23ESSD_CCClim/blob/main/figures/fig2.png" />  

More physical analysis of the cloud-type distributions is done with `reanalysis_plot.py`, `CRE_reanalysis.py` and `CRE_ICON.py`. The first two require joint dataframes made with `ERA_CCClim(_day).py` and `quickmake_join.py`, the latter only needs the ICON data.  The cloud radiative effects for the cloud types in CCClim (Fig. 7 in the paper) are shown below.  
<img src="https://github.com/EyringMLClimateGroup/kaps23ESSD_CCClim/blob/main/figures/fig7.png" width="400" />  
Comparison between geographical distributions of the cloud types in CCClim and ICON are produced with `diffESA_ICON.py`, as shown below (Fig 3./ Fig. 9 in the paper)  
<img src="https://github.com/EyringMLClimateGroup/kaps23ESSD_CCClim/blob/main/figures/fig3.png" width="500" />  
<img src="https://github.com/EyringMLClimateGroup/kaps23ESSD_CCClim/blob/main/figures/fig9.png" width="500" />  
