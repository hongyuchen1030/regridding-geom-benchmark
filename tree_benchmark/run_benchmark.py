#!/usr/bin/env python3
# Quick benchmark for nearest-neighbor on UGRID face centers using UXarray + sklearn.
# - KDTree on (x, y, z) unit vectors (Euclidean)
# - BallTree on (lat, lon) in radians (haversine)
# - BallTree again on (x, y, z) (Euclidean)

import math, time, os, gc
import numpy as np
import pandas as pd
import psutil
import uxarray as ux
from sklearn.neighbors import KDTree as SKKDTree, BallTree as SKBallTree

def deg2rad(x):
    return x * math.pi / 180.0

def rss_mb():
    # rough memory usage in MB
    return psutil.Process(os.getpid()).memory_info().rss / (1024.0**2)

def time_call(fn, repeat=1):
    t0 = time.perf_counter()
    ret = None
    for _ in range(repeat):
        ret = fn()
    return ret, (time.perf_counter() - t0)

def xyz_to_latlon_deg(xyz):
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    lat = np.degrees(np.arcsin(np.clip(z, -1.0, 1.0)))
    lon = np.degrees(np.arctan2(y, x))
    return lat, lon

def read_faces_from_ugrid(nc_path):
    """Return (lat_deg, lon_deg, xyz_unit) for face centers."""
    grid = ux.open_grid(nc_path)

    x = np.asarray(grid.face_x.values, dtype=float)
    y = np.asarray(grid.face_y.values, dtype=float)
    z = np.asarray(grid.face_z.values, dtype=float)
    X_xyz = np.column_stack((x, y, z))

    # normalize to unit radius
    nrm = np.linalg.norm(X_xyz, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    X_xyz = X_xyz / nrm

    if getattr(grid, "face_lat", None) is not None and getattr(grid, "face_lon", None) is not None:
        lat_deg = np.asarray(grid.face_lat.values, dtype=float)
        lon_deg = np.asarray(grid.face_lon.values, dtype=float)
    else:
        lat_deg, lon_deg = xyz_to_latlon_deg(X_xyz)

    # wrap lon to [-180, 180)
    lon_deg = ((lon_deg + 180.0) % 360.0) - 180.0
    return lat_deg, lon_deg, X_xyz

# ---------- build-time bench ----------
def bench_build_one_grid(nc_path, leaf_sizes=(32,), repeats=3, label=""):
    lat_deg, lon_deg, X_xyz = read_faces_from_ugrid(nc_path)
    N = X_xyz.shape[0]
    rows = []

    for leaf in leaf_sizes:
        X_rad = np.column_stack((deg2rad(lat_deg), deg2rad(lon_deg)))

        _, t_kd_build = time_call(
            lambda: SKKDTree(X_xyz, metric="euclidean", leaf_size=leaf),
            repeat=repeats,
        )
        _, t_bh_build = time_call(
            lambda: SKBallTree(X_rad, metric="haversine", leaf_size=leaf),
            repeat=repeats,
        )
        _, t_be_build = time_call(
            lambda: SKBallTree(X_xyz, metric="euclidean", leaf_size=leaf),
            repeat=repeats,
        )
        del X_rad
        gc.collect()

        row = dict(
            grid=os.path.basename(nc_path),
            label=label,
            N=N,
            repeats=repeats,
            leaf_size=leaf,
            build_time_kdtree_s=t_kd_build,
            build_time_balltree_haversine_s=t_bh_build,
            build_time_balltree_euclidean_s=t_be_build,
            rss_mb_after_build=round(rss_mb(), 1),
        )
        rows.append(row)

        print(
            f"[BUILD {label}] {os.path.basename(nc_path)} N={N} leaf={leaf} "
            f"KD={t_kd_build:.3f}s  BH={t_bh_build:.3f}s  BE={t_be_build:.3f}s  "
            f"RSS≈{rss_mb():.0f}MB"
        )
    return rows

# ---------- query-time bench ----------
def bench_query_one_grid(nc_path, k=1, leaf_sizes=(32,), repeats=3, exclude_self=False, label=""):
    lat_deg, lon_deg, X_xyz = read_faces_from_ugrid(nc_path)
    N = X_xyz.shape[0]
    k_eff = k + 1 if exclude_self else k

    Q_xyz = X_xyz
    X_rad = np.column_stack((deg2rad(lat_deg), deg2rad(lon_deg)))
    Q_rad = X_rad

    rows = []
    for leaf in leaf_sizes:
        kd = SKKDTree(X_xyz, metric="euclidean", leaf_size=leaf)
        bt_h = SKBallTree(X_rad, metric="haversine", leaf_size=leaf)
        bt_e = SKBallTree(X_xyz, metric="euclidean", leaf_size=leaf)

        _, t_kd = time_call(
            lambda: kd.query(Q_xyz, k=k_eff, return_distance=False),
            repeat=repeats,
        )
        _, t_bh = time_call(
            lambda: bt_h.query(Q_rad, k=k_eff, return_distance=False),
            repeat=repeats,
        )
        _, t_be = time_call(
            lambda: bt_e.query(Q_xyz, k=k_eff, return_distance=False),
            repeat=repeats,
        )

        del kd, bt_h, bt_e
        gc.collect()

        qtotal = Q_xyz.shape[0] * repeats
        row = dict(
            grid=os.path.basename(nc_path),
            label=label,
            N=N,
            k=k,
            k_eff=k_eff,
            leaf_size=leaf,
            repeats=repeats,
            time_kdtree_s=t_kd,
            time_balltree_haversine_s=t_bh,
            time_balltree_euclidean_s=t_be,
            qps_kdtree=(qtotal / t_kd) if t_kd > 0 else np.nan,
            qps_balltree_haversine=(qtotal / t_bh) if t_bh > 0 else np.nan,
            qps_balltree_euclidean=(qtotal / t_be) if t_be > 0 else np.nan,
            rss_mb_after_queries=round(rss_mb(), 1),
        )
        rows.append(row)

        print(
            f"[QUERY {label}] {os.path.basename(nc_path)} N={N} leaf={leaf} "
            f"KD_QPS={row['qps_kdtree']:.1f}  "
            f"BH_QPS={row['qps_balltree_haversine']:.1f}  "
            f"BE_QPS={row['qps_balltree_euclidean']:.1f}  "
            f"RSS≈{rss_mb():.0f}MB"
        )
    return rows

def main():
    # change these when switching inputs/settings
    grids = [
        "/pscratch/sd/h/hyvchen/meshes/uniform_grid_res184.g",
        "/pscratch/sd/h/hyvchen/meshes/uniform_grid_res369.g",
        "/pscratch/sd/h/hyvchen/meshes/uniform_grid_res922.g",
        "/pscratch/sd/h/hyvchen/meshes/uniform_grid_res1844.g",
        # "/pscratch/sd/h/hyvchen/meshes/uniform_grid_res4610.g",
    ]
    label = "uniform_facecenters"
    k = 1
    leaf_sizes = (16, 32, 64)
    repeats_query = 3
    repeats_build = 3
    exclude_self = True
    out_csv_query = "./output/results_uniform_facecenters.csv"
    out_csv_build = "./output/results_uniform_facecenters_build.csv"

    # query-time
    query_rows_all = []
    for nc in grids:
        if not os.path.exists(nc):
            print(f"missing file {nc}, skip")
            continue
        query_rows_all.extend(
            bench_query_one_grid(
                nc_path=nc,
                k=k,
                leaf_sizes=leaf_sizes,
                repeats=repeats_query,
                exclude_self=exclude_self,
                label=label,
            )
        )
        gc.collect()

    if query_rows_all:
        dfq = pd.DataFrame(query_rows_all)
        os.makedirs(os.path.dirname(out_csv_query), exist_ok=True)
        dfq.to_csv(out_csv_query, index=False)
        print(f"Wrote {out_csv_query}")

    # build-time
    build_rows_all = []
    for nc in grids:
        if not os.path.exists(nc):
            print(f"missing file {nc}, skip")
            continue
        build_rows_all.extend(
            bench_build_one_grid(
                nc_path=nc,
                leaf_sizes=leaf_sizes,
                repeats=repeats_build,
                label=label,
            )
        )
        gc.collect()

    if build_rows_all:
        dfb = pd.DataFrame(build_rows_all)
        os.makedirs(os.path.dirname(out_csv_build), exist_ok=True)
        dfb.to_csv(out_csv_build, index=False)
        print(f"Wrote {out_csv_build}")

if __name__ == "__main__":
    main()
