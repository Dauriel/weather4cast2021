import sys
from pathlib import Path

import netCDF4 as nc4
from tqdm.contrib.concurrent import process_map
from os import remove

def copy_nc_file(src_file, dst_file):
    with nc4.Dataset(src_file) as srcid:
         with nc4.Dataset(dst_file, mode='w') as dstid:
              # Create the dimensions of the file
              for name, dim in srcid.dimensions.items():
                  dstid.createDimension(name, len(dim) if not dim.isunlimited() else None)

              # Copy the global attributes
              dstid.setncatts({a:srcid.getncattr(a) for a in srcid.ncattrs()})

              # Create the variables in the file
              for name, var in srcid.variables.items():
                  dstid.createVariable(name, var.dtype, var.dimensions, zlib=True, complevel=1)

                  # Copy the variable attributes
                  dstid.variables[name].setncatts({a:var.getncattr(a) for a in var.ncattrs()})

                  # Copy the variables values
                  dstid.variables[name][:] = srcid.variables[name][:]

def _process(path):
        src_path, dst_path = path
        copy_nc_file(src_path, dst_path)
        remove(src_path)

if __name__ == '__main__':
    src_paths= Path(sys.argv[1]).rglob('*.nc')
    dst_paths = Path(sys.argv[2]).rglob('*.nc')
    paths = list(zip(src_paths, dst_paths))
    process_map(_process, paths, max_workers=4)
