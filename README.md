# libglm-da

8 Nov 2020

## Requirements
`xarray, numpy, dask, pyproj`, and [`glmtools`](https://github.com/deeplycloudy/glmtools).

## Scripts

Given 1 min GLM imagery files from glmtools, create 5 min aggregates from an hour of data:
```
python agg_glm.py OR_GLM-L2-GLMC-M3_G16_s202016120*.nc
```
It writes imagery files at a single time that are summed over the 5 min intervals. The time interval is configurable in the script.

The `glm_lon_lat_fed_obs.py` script navigates each GLM pixel centroid (in geostationary fixed grid coordiantes) to the lightning ellipsoid, and adds a new 2D variable cooresponding to each pixel in the GLM imagery to the GLM dataset. The latitudes and longitudes are therefore parallax corrected, and will be directly comparable to the flash, group, and event locations in the GLM L2 LCFA files.

As currently configured, the script perfoms all the calculations described above, but doesn't do anything further with those data. When run with, for example, an hour of data,
```
python glm_lon_lat_fed_obs.py HRRR_GLM-L2-GLMC-M3_G16_s202017*.nc
```
the script prepares two datasets, `glm`, which are 2D imagery, and `glm_px`, the same data but subset to individual flattened pixels. Either dataset can be written to a new NetCDF file with `glm.to_netcdf('filename.nc')`, or further processing can be performed by modifying this script. The script gives an example of obtaining a flattened list of flash extent density, longitude, and latitude arrays.
