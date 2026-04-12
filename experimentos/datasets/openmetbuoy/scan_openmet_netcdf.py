#!/usr/bin/env python3
"""
Inspeccion minima de variables en un NetCDF de OpenMetBuoy.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

from netCDF4 import Dataset


RAW_IMU_PATTERNS = (
    r"(?:^|_)acceleration_(?:x|y|z)(?:$|_)",
    r"(?:^|_)rotation_rate_(?:x|y|z)(?:$|_)",
    r"(?:^|_)gyro_(?:x|y|z)(?:$|_)",
    r"(?:^|_)(?:ax|ay|az|gx|gy|gz)(?:$|_)",
)


def looks_like_raw_imu_var(var_name: str) -> bool:
    n = var_name.lower()
    return any(re.search(p, n) for p in RAW_IMU_PATTERNS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect variables in a NetCDF file")
    parser.add_argument("netcdf_file", type=Path, help="Path to .nc file")
    args = parser.parse_args()

    if not args.netcdf_file.exists():
        raise SystemExit(f"File not found: {args.netcdf_file}")

    with Dataset(args.netcdf_file) as ds:
        dims = list(ds.dimensions.keys())
        vars_ = list(ds.variables.keys())

    print(f"file: {args.netcdf_file}")
    print(f"num_dims: {len(dims)}")
    print(f"num_vars: {len(vars_)}")
    print("dims:", ", ".join(dims))
    print("vars:")
    for v in vars_:
        print(f"  - {v}")

    imu_raw = [v for v in vars_ if looks_like_raw_imu_var(v)]
    print("\nraw_imu_vars:")
    if imu_raw:
        for v in imu_raw:
            print(f"  - {v}")
    else:
        print("  (none)")


if __name__ == "__main__":
    main()
