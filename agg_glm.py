import sys
from datetime import datetime
from glmtools.io.imagery import open_glm_time_series, aggregate

def compress_all(glm_grids):
    for var in glm_grids:
        glm_grids[var].encoding["zlib"] = True
        glm_grids[var].encoding["complevel"] = 4
        glm_grids[var].encoding["contiguous"] = False
    return glm_grids

def simplify_one_time(thisglm):
    # Patch up at least some metadata

    start = thisglm.time_bins.item().left
    end = thisglm.time_bins.item().right
    now = datetime.now()

    dataset_name = "HRRR_GLM-L2-GLM{5}-{0}_{1}_s{2}_e{3}_c{4}.nc"

    outname = dataset_name.format(
        'M3', thisglm.platform_ID,
        start.strftime('%Y%j%H%M%S0'),
        end.strftime('%Y%j%H%M%S0'),
        now.strftime('%Y%j%H%M%S0'),
        'C')
    # print(outname)

    # Something in here isn't getting compressed properly - these shouldn't be 150 MB.

    thisglm.attrs['dataset_name']=outname
    thisglm.attrs['time_coverage_start'] = start.isoformat()
    thisglm.attrs['time_coverage_end'] = end.isoformat()
    thisglm.attrs['date_created'] = now.isoformat()

    thisglm = thisglm.drop_vars('time_bins')
    # print(thisglm)

    thisglm = compress_all(thisglm)

    return thisglm, outname

def write_ncs(glmagg):

    outnames = []
    for ti in range(glmagg.dims['time_bins']):
        thisglm = glmagg[{'time_bins':ti}]
        glmout, outname = simplify_one_time(thisglm)
        glmout.to_netcdf(outname)
        outnames.append(outname)
    return outnames

if __name__ == '__main__':

    # === configurable parameters ===
    # Aggregate the glm imagery into chunks of this many minutes
    agg_min = 5

    # truncate the dataset to this time range
    time_range = None
    # time_range = datetime(2020,6,9,3,0), datetime(2020,6,9,5,0)

    # === end config

    # Load and aggregate
    glm_files = sys.argv[1:]
    glm = open_glm_time_series(glm_files)
    glmagg = aggregate(glm, agg_min, start_end=time_range)
    outnames = write_ncs(glmagg)
    print(outnames)