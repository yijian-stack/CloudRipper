# -*- coding: utf-8 -*-
"""
Sparse-scene 3D instance segmentation for scene B:
- train on local crops (positive + hard negative + random negative)
- infer on full scene
- model only learns foreground segmentation
- instance output obtained by spatial clustering + physical-size prior filtering

Dependencies:
  numpy, pandas, torch, spconv, scipy
"""

import os
import sys
import glob
import math
import argparse
import logging
from datetime import datetime

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import spconv.pytorch as spconv
from scipy.spatial import cKDTree


def get_tqdm():
    try:
        from tqdm.auto import tqdm as _tqdm
        if callable(_tqdm):
            return _tqdm
    except Exception:
        pass
    try:
        import tqdm as _tqdm_mod
        if hasattr(_tqdm_mod, "tqdm") and callable(_tqdm_mod.tqdm):
            return _tqdm_mod.tqdm
        if callable(_tqdm_mod):
            return _tqdm_mod
    except Exception:
        pass
    raise ImportError("Cannot import callable tqdm().")


tqdm = get_tqdm()


def setup_logging():
    logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(logs_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    log_filename = f"fginst_sparsecrop_v2_{timestamp}.log"
    log_filepath = os.path.join(logs_dir, log_filename)

    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.FileHandler(log_filepath, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger(__name__)
    logger.info("=" * 90)
    logger.info("FG-FIRST INSTANCE SEGMENTATION FOR SPARSE SCENE (HYSTERESIS MERGE)")
    logger.info("=" * 90)
    logger.info(f"Log file: {log_filepath}")
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return logger, log_filepath


def set_seed(seed: int):
    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def torch_safe_nan_to_num(x: torch.Tensor, nan=0.0, posinf=0.0, neginf=0.0):
    if hasattr(torch, "nan_to_num"):
        return torch.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)
    x = torch.where(torch.isnan(x), torch.full_like(x, float(nan)), x)
    x = torch.where(x == float("inf"), torch.full_like(x, float(posinf)), x)
    x = torch.where(x == float("-inf"), torch.full_like(x, float(neginf)), x)
    return x


def read_csv_points(path: str, input_dim: int, require_label: bool = False):
    if input_dim < 3:
        raise ValueError(f"input_dim must be >= 3, got {input_dim}")

    try:
        arr = np.loadtxt(path, delimiter=",", dtype=np.float32)
    except Exception:
        df = pd.read_csv(path, header=None)
        arr = df.values.astype(np.float32, copy=False)

    if arr.ndim == 1:
        arr = arr[None, :]

    if arr.shape[1] < input_dim:
        raise ValueError(f"CSV columns ({arr.shape[1]}) < input_dim ({input_dim}): {path}")

    xyz = arr[:, :3].astype(np.float32, copy=False)
    feats = arr[:, :input_dim].astype(np.float32, copy=False)

    inst = None
    if arr.shape[1] >= input_dim + 1:
        inst = arr[:, -1].astype(np.int64, copy=False)

    if require_label and inst is None:
        raise ValueError(f"Training requires label as last column, but not found: {path}")

    return xyz, feats, inst


def resolve_coord_unit_scale(
    xyz: np.ndarray,
    coord_unit_scale: float = 0.0,
    auto_unit_normalize: bool = True,
    auto_mm_threshold: float = 200.0,
):
    if coord_unit_scale and coord_unit_scale > 0:
        return float(coord_unit_scale)
    if (not auto_unit_normalize) or xyz.shape[0] == 0:
        return 1.0

    span = xyz.max(axis=0) - xyz.min(axis=0)
    scene_span = float(np.max(span))
    return 0.001 if scene_span > float(auto_mm_threshold) else 1.0


def voxel_grid_downsample(
    xyz: np.ndarray,
    feats_raw: np.ndarray,
    inst: np.ndarray = None,
    voxel_size: float = 0.0,
):
    N = int(xyz.shape[0])
    if voxel_size <= 0 or N <= 1:
        inverse = np.arange(N, dtype=np.int64)
        return xyz.astype(np.float32), feats_raw.astype(np.float32), (inst.astype(np.int64) if inst is not None else None), inverse

    coords = np.floor(xyz / float(voxel_size)).astype(np.int64)
    _, inverse = np.unique(coords, axis=0, return_inverse=True)

    order = np.argsort(inverse, kind="mergesort")
    inv_s = inverse[order]
    starts = np.r_[0, np.flatnonzero(inv_s[1:] != inv_s[:-1]) + 1]
    counts = np.diff(np.r_[starts, inv_s.shape[0]]).astype(np.int64)

    xyz_s = xyz[order].astype(np.float32, copy=False)
    feats_s = feats_raw[order].astype(np.float32, copy=False)
    xyz_ds = (np.add.reduceat(xyz_s, starts, axis=0) / counts[:, None]).astype(np.float32)
    feats_ds = (np.add.reduceat(feats_s, starts, axis=0) / counts[:, None]).astype(np.float32)
    feats_ds[:, :3] = xyz_ds

    inst_ds = None
    if inst is not None:
        inst_ds = np.zeros((starts.shape[0],), dtype=np.int64)
        for i, s in enumerate(starts.tolist()):
            e = s + int(counts[i])
            labs = inst[order[s:e]]
            pos = labs[labs > 0]
            if pos.size > 0:
                vals, cnts = np.unique(pos, return_counts=True)
                inst_ds[i] = int(vals[np.argmax(cnts)])

    return xyz_ds, feats_ds, inst_ds, inverse.astype(np.int64)


def prepare_scene_points(
    xyz: np.ndarray,
    feats_raw: np.ndarray,
    inst: np.ndarray = None,
    coord_unit_scale: float = 0.0,
    auto_unit_normalize: bool = True,
    auto_mm_threshold: float = 200.0,
    pre_downsample_voxel: float = 0.0,
):
    scale = resolve_coord_unit_scale(
        xyz,
        coord_unit_scale=coord_unit_scale,
        auto_unit_normalize=auto_unit_normalize,
        auto_mm_threshold=auto_mm_threshold,
    )

    xyz_scaled = (xyz.astype(np.float32, copy=False) * float(scale)).astype(np.float32)
    feats_scaled = feats_raw.astype(np.float32, copy=True)
    feats_scaled[:, :3] = xyz_scaled

    xyz_ds, feats_ds, inst_ds, inverse_map = voxel_grid_downsample(
        xyz_scaled,
        feats_scaled,
        inst=inst,
        voxel_size=pre_downsample_voxel,
    )

    meta = {
        "coord_unit_scale": float(scale),
        "orig_points": int(xyz.shape[0]),
        "downsampled_points": int(xyz_ds.shape[0]),
        "downsample_ratio": float(xyz_ds.shape[0] / max(1, xyz.shape[0])),
        "inverse_map": inverse_map,
    }
    return xyz_ds, feats_ds, inst_ds, meta


def normalize_points(
    points_xyz: np.ndarray,
    mode: str = "center_scale",
    scale_quantile: float = 95.0,
    scale_clip_min: float = 0.0,
    scale_clip_max: float = 0.0,
):
    if points_xyz.shape[0] == 0:
        return points_xyz.astype(np.float32), {"centroid": np.zeros((3,), np.float32), "scale": 1.0}

    pts = points_xyz.astype(np.float32, copy=False)
    centroid = pts.mean(axis=0).astype(np.float32)
    centered = pts - centroid
    centered = np.nan_to_num(centered, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    if mode == "none":
        return pts.astype(np.float32), {"centroid": np.zeros((3,), np.float32), "scale": 1.0}

    if mode == "center":
        return centered, {"centroid": centroid, "scale": 1.0}

    q = float(np.clip(scale_quantile, 50.0, 99.9))
    dist = np.linalg.norm(centered, axis=1)
    scale = float(np.percentile(dist, q))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    if scale_clip_min > 0:
        scale = max(scale, float(scale_clip_min))
    if scale_clip_max > 0:
        scale = min(scale, float(scale_clip_max))

    scale = max(scale, 1e-6)
    pn = (centered / scale).astype(np.float32)
    pn = np.nan_to_num(pn, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return pn, {"centroid": centroid, "scale": scale}


def build_point_inputs(
    xyz: np.ndarray,
    feats_raw: np.ndarray,
    input_dim: int,
    normalize_mode: str,
    scale_quantile: float,
    scale_clip_min: float,
    scale_clip_max: float,
    abs_coord_scale: float,
    use_abs_coords: bool = False,
):
    xyz_scene, meta = normalize_points(
        xyz,
        mode=normalize_mode,
        scale_quantile=scale_quantile,
        scale_clip_min=scale_clip_min,
        scale_clip_max=scale_clip_max,
    )

    feat_list = [xyz_scene]
    if bool(use_abs_coords):
        abs_xyz = (xyz / float(abs_coord_scale)).astype(np.float32)
        feat_list.append(abs_xyz)

    extra_dim = max(0, int(input_dim) - 3)
    if input_dim > 3:
        extra = feats_raw[:, 3:input_dim].astype(np.float32, copy=False)
        extra = np.nan_to_num(extra, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        feat_list.append(extra)

    point_inputs = np.concatenate(feat_list, axis=1).astype(np.float32)
    point_inputs = np.nan_to_num(point_inputs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    meta = dict(meta)
    meta["use_abs_coords"] = int(bool(use_abs_coords))
    meta["extra_dim"] = int(extra_dim)
    return xyz_scene, point_inputs, meta


def select_voxel_point_features(
    xyz_scene: np.ndarray,
    point_inputs: np.ndarray,
    extra_dim: int,
    voxel_feature_mode: str,
):
    mode = str(voxel_feature_mode).lower()
    extra_dim = int(max(0, extra_dim))

    if mode == "extra_only" and extra_dim > 0 and point_inputs.shape[1] >= extra_dim:
        feats = point_inputs[:, -extra_dim:]
    elif mode == "scene_xyz_extra":
        if extra_dim > 0 and point_inputs.shape[1] >= extra_dim:
            feats = np.concatenate([xyz_scene, point_inputs[:, -extra_dim:]], axis=1).astype(np.float32)
        else:
            feats = xyz_scene.astype(np.float32)
    else:
        feats = point_inputs

    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return feats


def build_instance_aux_targets(
    xyz_scene: np.ndarray,
    inst: np.ndarray,
    min_points: int = 8,
    min_axis_scale: float = 0.12,
):
    xyz_scene = np.asarray(xyz_scene, dtype=np.float32)
    inst = np.asarray(inst, dtype=np.int64)
    n = int(xyz_scene.shape[0])

    center_target = np.zeros((n,), dtype=np.float32)
    offset_target = np.zeros((n, 3), dtype=np.float32)
    offset_weight = np.zeros((n,), dtype=np.float32)
    size_target = np.zeros((n, 3), dtype=np.float32)
    size_weight = np.zeros((n,), dtype=np.float32)

    if n == 0:
        return center_target, offset_target, offset_weight, size_target, size_weight

    obj_ids = np.unique(inst[inst > 0]).astype(np.int64)
    for uid in obj_ids.tolist():
        idx = np.where(inst == int(uid))[0]
        if idx.size == 0:
            continue

        pts = xyz_scene[idx].astype(np.float32, copy=False)
        center = pts.mean(axis=0).astype(np.float32)
        q05 = np.percentile(pts, 5, axis=0).astype(np.float32)
        q95 = np.percentile(pts, 95, axis=0).astype(np.float32)
        axis_scale = np.maximum(0.5 * (q95 - q05), float(min_axis_scale)).astype(np.float32)

        rel = (pts - center[None, :]) / axis_scale[None, :]
        dist = np.linalg.norm(rel, axis=1)
        centerness = np.exp(-0.5 * np.minimum(dist, 4.0) ** 2).astype(np.float32)
        if idx.size >= int(min_points):
            centerness = np.clip(centerness, 0.05, 1.0).astype(np.float32)
        else:
            centerness = np.full((idx.size,), 0.10, dtype=np.float32)

        center_target[idx] = np.maximum(center_target[idx], centerness)
        offset_target[idx] = (center[None, :] - pts).astype(np.float32)
        offset_weight[idx] = np.maximum(offset_weight[idx], 0.30 + 0.70 * centerness)
        bbox_size = np.maximum(q95 - q05, float(min_axis_scale) * 2.0).astype(np.float32)
        size_target[idx] = np.log1p(bbox_size[None, :]).astype(np.float32)
        size_weight[idx] = np.maximum(size_weight[idx], 0.25 + 0.75 * centerness)

    return center_target, offset_target, offset_weight, size_target, size_weight


def fuse_fg_center_prob_np(
    fg_prob: np.ndarray,
    center_prob: np.ndarray,
    center_mix: float = 0.15,
    center_power: float = 0.50,
):
    fg_prob = np.asarray(fg_prob, dtype=np.float32)
    center_prob = np.asarray(center_prob, dtype=np.float32)
    mix = float(np.clip(center_mix, 0.0, 1.0))
    power = float(max(center_power, 1e-6))
    center_term = np.clip(center_prob, 0.0, 1.0) ** power
    obj_prob = fg_prob * ((1.0 - mix) + mix * center_term)
    return np.clip(obj_prob, 0.0, 1.0).astype(np.float32)


def fuse_fg_center_prob_torch(
    fg_prob: torch.Tensor,
    center_prob: torch.Tensor,
    center_mix: float = 0.15,
    center_power: float = 0.50,
):
    mix = float(np.clip(center_mix, 0.0, 1.0))
    power = float(max(center_power, 1e-6))
    center_term = center_prob.clamp(0.0, 1.0).pow(power)
    return (fg_prob * ((1.0 - mix) + mix * center_term)).clamp(0.0, 1.0)


def clip_vector_norm_np(vec: np.ndarray, max_norm: float):
    vec = np.asarray(vec, dtype=np.float32)
    if max_norm <= 0 or vec.shape[0] == 0:
        return vec.astype(np.float32, copy=False)
    norm = np.linalg.norm(vec, axis=1, keepdims=True)
    scale = np.minimum(1.0, float(max_norm) / (norm + 1e-6))
    return (vec * scale).astype(np.float32)


def is_cuda_oom_error(exc: Exception):
    msg = str(exc).lower()
    return isinstance(exc, RuntimeError) and ("out of memory" in msg or "cudaerrormemoryallocation" in msg)


def coord_hash_4d_torch(coords: torch.Tensor):
    coords = coords.to(torch.int64)
    if coords.dim() != 2 or coords.shape[1] != 4:
        raise ValueError(f"coords must be (N,4), got {tuple(coords.shape)}")
    b = coords[:, 0]
    x = coords[:, 1]
    y = coords[:, 2]
    z = coords[:, 3]
    return (((b << 16) | x) << 16 | y) << 16 | z


def gather_sparse_to_points(
    sparse_tensor,
    point_base_coords: torch.Tensor,
    stride_div: int,
):
    if int(stride_div) <= 1:
        coarse_coords = point_base_coords.to(torch.int64)
    else:
        coarse_coords = point_base_coords.to(torch.int64).clone()
        coarse_coords[:, 1:] = torch.div(coarse_coords[:, 1:], int(stride_div), rounding_mode="floor")

    sparse_coords = sparse_tensor.indices.to(torch.int64)
    sparse_hash = coord_hash_4d_torch(sparse_coords)
    point_hash = coord_hash_4d_torch(coarse_coords)

    sparse_hash_sorted, order = torch.sort(sparse_hash)
    pos = torch.searchsorted(sparse_hash_sorted, point_hash)
    valid = pos < sparse_hash_sorted.shape[0]
    valid_idx = torch.where(valid)[0]
    if valid_idx.numel() > 0:
        valid_match = sparse_hash_sorted[pos[valid_idx]] == point_hash[valid_idx]
        valid = torch.zeros_like(valid, dtype=torch.bool)
        valid[valid_idx] = valid_match

    out = sparse_tensor.features.new_zeros((point_base_coords.shape[0], sparse_tensor.features.shape[1]))
    if bool(valid.any()):
        matched = order[pos[valid]]
        out[valid] = sparse_tensor.features[matched]
    return out


def partition_scene_indices_recursive(
    xyz_raw: np.ndarray,
    max_points: int = 260000,
    min_points: int = 32000,
    split_overlap: float = 2.0,
    max_span_xyz=(192.0, 192.0, 96.0),
    max_depth: int = 10,
):
    xyz_raw = np.asarray(xyz_raw, dtype=np.float32)
    n = int(xyz_raw.shape[0])
    if n == 0:
        return [np.zeros((0,), dtype=np.int64)]

    max_points = int(max(1, max_points))
    min_points = int(max(1, min_points))
    max_span_xyz = np.asarray(max_span_xyz, dtype=np.float32)
    out = []

    def rec(idx: np.ndarray, depth: int):
        idx = np.asarray(idx, dtype=np.int64)
        if idx.size == 0:
            return

        pts = xyz_raw[idx]
        span = (pts.max(axis=0) - pts.min(axis=0)).astype(np.float32)
        need_split = (idx.size > max_points) or bool(np.any(span > max_span_xyz))
        if (not need_split) or idx.size <= min_points or depth >= int(max_depth):
            out.append(np.unique(idx).astype(np.int64))
            return

        axis = int(np.argmax(span))
        coord = pts[:, axis]
        mid = float(np.median(coord))
        overlap = float(min(max(float(split_overlap), 0.0), max(float(span[axis]) * 0.45, 0.0)))

        left = idx[coord <= (mid + overlap)]
        right = idx[coord >= (mid - overlap)]

        too_small = int(max(256, min_points // 3))
        if (
            left.size == idx.size
            or right.size == idx.size
            or left.size < too_small
            or right.size < too_small
        ):
            order = np.argsort(coord, kind="mergesort")
            idx_sorted = idx[order]
            cut = idx_sorted.size // 2
            margin = int(max(32, round(idx_sorted.size * 0.08)))
            left = idx_sorted[: min(idx_sorted.size, cut + margin)]
            right = idx_sorted[max(0, cut - margin) :]

        left = np.unique(left).astype(np.int64)
        right = np.unique(right).astype(np.int64)
        if left.size == idx.size or right.size == idx.size:
            out.append(idx.astype(np.int64))
            return

        rec(left, depth + 1)
        rec(right, depth + 1)

    rec(np.arange(n, dtype=np.int64), 0)
    return out if len(out) > 0 else [np.arange(n, dtype=np.int64)]


@torch.no_grad()
def infer_scene_prob_from_arrays(
    model,
    device,
    pts_scene: np.ndarray,
    raw_xyz: np.ndarray,
    point_inputs: np.ndarray,
    extra_dim: int,
    voxel_size: float,
    max_points_per_voxel: int,
    max_voxels: int,
    feature_mode: str,
    voxel_feature_mode: str,
    center_prob_mix: float = 0.15,
    center_prob_power: float = 0.50,
    vote_max_offset_norm: float = 4.0,
    scene_scale: float = 1.0,
):
    pts_scene = np.asarray(pts_scene, dtype=np.float32)
    raw_xyz = np.asarray(raw_xyz, dtype=np.float32)
    point_inputs = np.asarray(point_inputs, dtype=np.float32)

    n = int(pts_scene.shape[0])
    if n == 0:
        z = np.zeros((0,), dtype=np.float32)
        return {
            "keep_idx": np.zeros((0,), dtype=np.int64),
            "fg_prob": z,
            "center_prob": z,
            "score_prob": z,
            "vote_xyz": np.zeros((0, 3), dtype=np.float32),
        }

    vin = select_voxel_point_features(
        pts_scene,
        point_inputs,
        extra_dim=int(extra_dim),
        voxel_feature_mode=voxel_feature_mode,
    )
    vcoords, vfeats, _, p2v, keep_mask = voxelize_with_mapping(
        pts_scene,
        voxel_size=voxel_size,
        point_features=vin,
        max_points_per_voxel=max_points_per_voxel,
        max_voxels=max_voxels,
        feature_mode=feature_mode,
        quantize="floor",
    )

    km = keep_mask.cpu().numpy().astype(bool)
    keep_idx = np.where(km)[0].astype(np.int64)
    if vcoords.shape[0] == 0 or keep_idx.size == 0:
        z = np.zeros((0,), dtype=np.float32)
        return {
            "keep_idx": np.zeros((0,), dtype=np.int64),
            "fg_prob": z,
            "center_prob": z,
            "score_prob": z,
            "vote_xyz": np.zeros((0, 3), dtype=np.float32),
        }

    coords_b = torch.cat([torch.zeros((vcoords.shape[0], 1), dtype=torch.int32), vcoords], dim=1).to(device)
    feats_b = torch_safe_nan_to_num(vfeats.to(device), nan=0.0, posinf=0.0, neginf=0.0)
    p2v_b = p2v[keep_mask].to(device)
    pin_b = torch_safe_nan_to_num(torch.from_numpy(point_inputs[km]).to(device=device, dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
    pb_b = torch.zeros((keep_idx.size,), dtype=torch.int64, device=device)

    use_amp = bool(device.type == "cuda")
    with torch.cuda.amp.autocast(enabled=use_amp):
        outputs = model(
            coords_b,
            feats_b,
            batch_size=1,
            spatial_shape=calculate_spatial_shape(coords_b),
            point2voxel=p2v_b,
            point_inputs=pin_b,
            point_batch_ids=pb_b,
        )
    outputs = unpack_model_outputs(outputs)
    fg_prob = torch.sigmoid(outputs["fg_logits"]).detach().cpu().numpy().astype(np.float32)
    center_prob = torch.sigmoid(outputs["center_logits"]).detach().cpu().numpy().astype(np.float32)
    score_prob = fuse_fg_center_prob_np(
        fg_prob,
        center_prob,
        center_mix=center_prob_mix,
        center_power=center_prob_power,
    )
    offset_scene = outputs["offset_pred"].detach().cpu().numpy().astype(np.float32)
    offset_scene = clip_vector_norm_np(offset_scene, max_norm=float(vote_max_offset_norm))
    vote_xyz = raw_xyz[km] + offset_scene * float(scene_scale)
    return {
        "keep_idx": keep_idx,
        "fg_prob": fg_prob,
        "center_prob": center_prob,
        "score_prob": score_prob,
        "vote_xyz": vote_xyz.astype(np.float32),
    }


@torch.no_grad()
def infer_scene_prob_chunked(
    model,
    device,
    scene: dict,
    voxel_size: float,
    max_points_per_voxel: int,
    max_voxels: int,
    feature_mode: str,
    voxel_feature_mode: str,
    center_prob_mix: float = 0.15,
    center_prob_power: float = 0.50,
    vote_max_offset_norm: float = 4.0,
    chunk_max_points: int = 260000,
    chunk_min_points: int = 32000,
    chunk_overlap: float = 2.0,
    chunk_max_span_x: float = 192.0,
    chunk_max_span_y: float = 192.0,
    chunk_max_span_z: float = 96.0,
    oom_retry_depth: int = 0,
):
    raw_xyz = np.asarray(scene["xyz"], dtype=np.float32)
    pts_scene = np.asarray(scene["xyz_scene"], dtype=np.float32)
    point_inputs = np.asarray(scene["point_inputs"], dtype=np.float32)
    target = (np.asarray(scene["inst"], dtype=np.int64) > 0).astype(np.float32) if scene.get("inst") is not None else np.zeros((raw_xyz.shape[0],), dtype=np.float32)
    inst = np.asarray(scene["inst"], dtype=np.int64) if scene.get("inst") is not None else np.zeros((raw_xyz.shape[0],), dtype=np.int64)
    extra_dim = int(scene.get("extra_dim", max(0, point_inputs.shape[1] - 3)))
    scene_scale = float(scene.get("scene_scale", 1.0))

    n = int(raw_xyz.shape[0])
    if n == 0:
        z = np.zeros((0,), dtype=np.float32)
        return z, z, z, np.zeros((0, 3), dtype=np.float32), raw_xyz, target, inst, [scene["name"]]

    chunks = partition_scene_indices_recursive(
        raw_xyz,
        max_points=int(chunk_max_points),
        min_points=int(chunk_min_points),
        split_overlap=float(chunk_overlap),
        max_span_xyz=(float(chunk_max_span_x), float(chunk_max_span_y), float(chunk_max_span_z)),
        max_depth=10,
    )

    if len(chunks) <= 1 and n <= int(chunk_max_points):
        try:
            out = infer_scene_prob_from_arrays(
                model,
                device,
                pts_scene,
                raw_xyz,
                point_inputs,
                extra_dim=extra_dim,
                voxel_size=voxel_size,
                max_points_per_voxel=max_points_per_voxel,
                max_voxels=max_voxels,
                feature_mode=feature_mode,
                voxel_feature_mode=voxel_feature_mode,
                center_prob_mix=center_prob_mix,
                center_prob_power=center_prob_power,
                vote_max_offset_norm=vote_max_offset_norm,
                scene_scale=scene_scale,
            )
        except Exception as exc:
            if (not is_cuda_oom_error(exc)) or int(oom_retry_depth) >= 3 or n <= int(max(1, chunk_min_points)):
                raise
            if device.type == "cuda":
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            fallback_max_points = int(max(chunk_min_points, min(max(chunk_min_points + 1, n - 1), max(80000, n // 2))))
            fallback_span_x = float(max(24.0, chunk_max_span_x * 0.6))
            fallback_span_y = float(max(24.0, chunk_max_span_y * 0.6))
            fallback_span_z = float(max(12.0, chunk_max_span_z * 0.6))
            return infer_scene_prob_chunked(
                model,
                device,
                scene,
                voxel_size=voxel_size,
                max_points_per_voxel=max_points_per_voxel,
                max_voxels=max_voxels,
                feature_mode=feature_mode,
                voxel_feature_mode=voxel_feature_mode,
                center_prob_mix=center_prob_mix,
                center_prob_power=center_prob_power,
                vote_max_offset_norm=vote_max_offset_norm,
                chunk_max_points=fallback_max_points,
                chunk_min_points=chunk_min_points,
                chunk_overlap=chunk_overlap,
                chunk_max_span_x=fallback_span_x,
                chunk_max_span_y=fallback_span_y,
                chunk_max_span_z=fallback_span_z,
                oom_retry_depth=int(oom_retry_depth) + 1,
            )
        fg_prob = np.zeros((n,), dtype=np.float32)
        center_prob = np.zeros((n,), dtype=np.float32)
        score_prob = np.zeros((n,), dtype=np.float32)
        vote_xyz = raw_xyz.astype(np.float32, copy=True)
        fg_prob[out["keep_idx"]] = out["fg_prob"]
        center_prob[out["keep_idx"]] = out["center_prob"]
        score_prob[out["keep_idx"]] = out["score_prob"]
        vote_xyz[out["keep_idx"]] = out["vote_xyz"]
        return fg_prob, center_prob, score_prob, vote_xyz, raw_xyz, target, inst, [scene["name"]]

    fg_sum = np.zeros((n,), dtype=np.float32)
    center_sum = np.zeros((n,), dtype=np.float32)
    vote_sum = np.zeros((n, 3), dtype=np.float32)
    hit = np.zeros((n,), dtype=np.float32)

    for idx in chunks:
        idx = np.asarray(idx, dtype=np.int64)
        if idx.size == 0:
            continue
        try:
            out = infer_scene_prob_from_arrays(
                model,
                device,
                pts_scene[idx],
                raw_xyz[idx],
                point_inputs[idx],
                extra_dim=extra_dim,
                voxel_size=voxel_size,
                max_points_per_voxel=max_points_per_voxel,
                max_voxels=max_voxels,
                feature_mode=feature_mode,
                voxel_feature_mode=voxel_feature_mode,
                center_prob_mix=center_prob_mix,
                center_prob_power=center_prob_power,
                vote_max_offset_norm=vote_max_offset_norm,
                scene_scale=scene_scale,
            )
        except Exception as exc:
            if (not is_cuda_oom_error(exc)) or int(oom_retry_depth) >= 3 or idx.size <= int(max(1, chunk_min_points)):
                raise
            if device.type == "cuda":
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            sub_scene = {
                "name": scene["name"],
                "xyz": raw_xyz[idx],
                "xyz_scene": pts_scene[idx],
                "point_inputs": point_inputs[idx],
                "inst": inst[idx] if scene.get("inst") is not None else None,
                "extra_dim": extra_dim,
                "scene_scale": scene_scale,
            }
            sub_fg, sub_center, sub_score, sub_vote, _, _, _, _ = infer_scene_prob_chunked(
                model,
                device,
                sub_scene,
                voxel_size=voxel_size,
                max_points_per_voxel=max_points_per_voxel,
                max_voxels=max_voxels,
                feature_mode=feature_mode,
                voxel_feature_mode=voxel_feature_mode,
                center_prob_mix=center_prob_mix,
                center_prob_power=center_prob_power,
                vote_max_offset_norm=vote_max_offset_norm,
                chunk_max_points=max(int(chunk_min_points), int(max(4096, idx.size // 2))),
                chunk_min_points=max(4096, int(chunk_min_points // 2)),
                chunk_overlap=chunk_overlap,
                chunk_max_span_x=max(12.0, float(chunk_max_span_x * 0.6)),
                chunk_max_span_y=max(12.0, float(chunk_max_span_y * 0.6)),
                chunk_max_span_z=max(8.0, float(chunk_max_span_z * 0.6)),
                oom_retry_depth=int(oom_retry_depth) + 1,
            )
            fg_sum[idx] += sub_fg
            center_sum[idx] += sub_center
            vote_sum[idx] += sub_vote
            hit[idx] += (sub_fg >= 0.0).astype(np.float32)
            continue
        if out["keep_idx"].size == 0:
            continue
        gidx = idx[out["keep_idx"]]
        fg_sum[gidx] += out["fg_prob"]
        center_sum[gidx] += out["center_prob"]
        vote_sum[gidx] += out["vote_xyz"]
        hit[gidx] += 1.0

    miss = hit <= 0
    if np.any(miss):
        out = infer_scene_prob_from_arrays(
            model,
            device,
            pts_scene[miss],
            raw_xyz[miss],
            point_inputs[miss],
            extra_dim=extra_dim,
            voxel_size=voxel_size,
            max_points_per_voxel=max_points_per_voxel,
            max_voxels=max_voxels,
            feature_mode=feature_mode,
            voxel_feature_mode=voxel_feature_mode,
            center_prob_mix=center_prob_mix,
            center_prob_power=center_prob_power,
            vote_max_offset_norm=vote_max_offset_norm,
            scene_scale=scene_scale,
        )
        if out["keep_idx"].size > 0:
            miss_idx = np.where(miss)[0][out["keep_idx"]]
            fg_sum[miss_idx] += out["fg_prob"]
            center_sum[miss_idx] += out["center_prob"]
            vote_sum[miss_idx] += out["vote_xyz"]
            hit[miss_idx] += 1.0

    hit_safe = np.maximum(hit, 1.0)
    fg_prob = (fg_sum / hit_safe).astype(np.float32)
    center_prob = (center_sum / hit_safe).astype(np.float32)
    score_prob = fuse_fg_center_prob_np(
        fg_prob,
        center_prob,
        center_mix=center_prob_mix,
        center_power=center_prob_power,
    )
    vote_xyz = raw_xyz.astype(np.float32, copy=True)
    ok = hit > 0
    vote_xyz[ok] = (vote_sum[ok] / hit_safe[ok, None]).astype(np.float32)
    return fg_prob, center_prob, score_prob, vote_xyz, raw_xyz, target, inst, [scene["name"]]


def voxelize_with_mapping(
    coords_norm_xyz: np.ndarray,
    voxel_size: float,
    point_features: np.ndarray = None,
    max_points_per_voxel: int = 50,
    max_voxels: int = 0,
    feature_mode: str = "safe",
    quantize: str = "floor",
):
    if point_features is None:
        point_features = coords_norm_xyz

    if coords_norm_xyz.shape[0] == 0:
        D = int(point_features.shape[1])
        return (
            torch.zeros((0, 3), dtype=torch.int32),
            torch.zeros((0, 3 * D + 2), dtype=torch.float32),
            np.zeros((0, 3), dtype=np.float32),
            torch.zeros((0,), dtype=torch.int64),
            torch.zeros((0,), dtype=torch.bool),
        )

    if coords_norm_xyz.shape[1] != 3:
        raise ValueError(f"coords_norm_xyz must be (N,3), got {coords_norm_xyz.shape}")

    if point_features.shape[0] != coords_norm_xyz.shape[0]:
        raise ValueError("point_features N must match coords_norm_xyz N")

    point_features = np.nan_to_num(point_features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    D = int(point_features.shape[1])

    if quantize == "round":
        coords = np.round(coords_norm_xyz / voxel_size).astype(np.int32)
    else:
        coords = np.floor(coords_norm_xyz / voxel_size).astype(np.int32)

    min_coords = coords.min(axis=0)
    coords_shift = coords - min_coords

    unique_coords, inverse = np.unique(coords_shift, axis=0, return_inverse=True)
    V = unique_coords.shape[0]

    order = np.argsort(inverse, kind="mergesort")
    inv_s = inverse[order]
    feats_s = point_features[order].astype(np.float32)

    starts = np.r_[0, np.flatnonzero(inv_s[1:] != inv_s[:-1]) + 1]
    gids = inv_s[starts]
    counts_full = np.diff(np.r_[starts, inv_s.shape[0]]).astype(np.int32)

    rep_starts = np.repeat(starts, counts_full)
    rank = np.arange(inv_s.shape[0]) - rep_starts

    K = int(max_points_per_voxel) if max_points_per_voxel and max_points_per_voxel > 0 else 10**9
    keep = rank < K
    keep_f = keep.astype(np.float32)[:, None]

    sums = np.add.reduceat(feats_s * keep_f, starts, axis=0)
    cnt = np.minimum(counts_full, K).astype(np.float32)
    cnt_safe = cnt + 1e-6
    mean_g = sums / cnt_safe[:, None]

    sums2 = np.add.reduceat((feats_s * feats_s) * keep_f, starts, axis=0)
    var_g = np.maximum(0.0, sums2 / cnt_safe[:, None] - mean_g * mean_g)
    std_g = np.sqrt(var_g).astype(np.float32)

    dev = np.abs(feats_s - mean_g[inv_s])
    neg_inf = -1e10
    dev = np.where(keep[:, None], dev, neg_inf)
    maxdev_g = np.maximum.reduceat(dev, starts, axis=0).astype(np.float32)

    voxel_volume = float(voxel_size**3)
    density_raw = (cnt / voxel_volume).astype(np.float32)

    cfeat = np.zeros((V,), dtype=np.float32)
    dfeat = np.zeros((V,), dtype=np.float32)

    if feature_mode == "legacy":
        cfeat[gids] = cnt
        dfeat[gids] = density_raw
    else:
        cfeat[gids] = cnt / float(max(K, 1))
        max_density = float(max(K, 1)) / voxel_volume
        dfeat[gids] = np.log1p(density_raw) / (np.log1p(max_density) + 1e-6)

    mean = np.zeros((V, D), dtype=np.float32)
    std = np.zeros((V, D), dtype=np.float32)
    maxdev = np.zeros((V, D), dtype=np.float32)

    mean[gids] = mean_g
    std[gids] = std_g
    maxdev[gids] = maxdev_g

    feats = np.concatenate([mean, std, maxdev, cfeat[:, None], dfeat[:, None]], axis=1).astype(np.float32)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    voxel_centers = ((unique_coords + min_coords) * voxel_size).astype(np.float32)
    point2voxel = inverse.astype(np.int64)

    if max_voxels and max_voxels > 0 and V > max_voxels:
        raw_counts = np.zeros((V,), dtype=np.int32)
        raw_counts[gids] = counts_full
        keep_vox = np.argsort(-raw_counts)[:max_voxels]
        keep_vox = np.sort(keep_vox)

        old2new = -np.ones((V,), dtype=np.int64)
        old2new[keep_vox] = np.arange(keep_vox.shape[0], dtype=np.int64)

        new_p2v = old2new[point2voxel]
        keep_points = new_p2v >= 0

        unique_coords = unique_coords[keep_vox]
        feats = feats[keep_vox]
        voxel_centers = voxel_centers[keep_vox]
        point2voxel = new_p2v

        return (
            torch.from_numpy(unique_coords).to(torch.int32),
            torch.from_numpy(feats).to(torch.float32),
            voxel_centers,
            torch.from_numpy(point2voxel).to(torch.int64),
            torch.from_numpy(keep_points).to(torch.bool),
        )

    return (
        torch.from_numpy(unique_coords).to(torch.int32),
        torch.from_numpy(feats).to(torch.float32),
        voxel_centers,
        torch.from_numpy(point2voxel).to(torch.int64),
        torch.ones((coords_norm_xyz.shape[0],), dtype=torch.bool),
    )


def deduplicate_files(files, logger=None):
    """Remove duplicate files by normalizing basenames (e.g. 'name.csv' == 'name - Cloud.csv').

    When multiple files share the same normalized name, keep the ' - Cloud.csv' version
    if available, otherwise keep the alphabetically first one.
    """
    import re
    groups = {}
    for fp in files:
        bn = os.path.basename(fp)
        norm = re.sub(r"\s*-\s*Cloud", "", bn, flags=re.IGNORECASE)
        # Normalize variant like '...10-24-28a.csv' → '...10-24-28.csv'
        norm = re.sub(r"(\d)a\.csv$", r"\1.csv", norm)
        norm = norm.lower().strip()
        groups.setdefault(norm, []).append(fp)

    result = []
    for norm_key, candidates in groups.items():
        if len(candidates) == 1:
            result.append(candidates[0])
        else:
            cloud_versions = [f for f in candidates if "- cloud" in os.path.basename(f).lower()]
            chosen = cloud_versions[0] if cloud_versions else sorted(candidates)[0]
            if logger is not None:
                dropped = [os.path.basename(f) for f in candidates if f != chosen]
                logger.info(f"Dedup: keeping '{os.path.basename(chosen)}', dropping {dropped}")
            result.append(chosen)
    return sorted(result)


def split_files(files, train_ratio=0.9, seed=42):
    files = sorted(files)
    rng = np.random.RandomState(seed)
    idx = np.arange(len(files))
    rng.shuffle(idx)
    n_train = int(len(files) * train_ratio)
    train_files = [files[i] for i in idx[:n_train]]
    val_files = [files[i] for i in idx[n_train:]]
    return train_files, val_files


def convex_hull_2d(points_xy: np.ndarray):
    pts = np.asarray(points_xy, dtype=np.float32)
    if pts.shape[0] <= 1:
        return pts.astype(np.float32)

    pts = np.unique(np.round(pts, decimals=6), axis=0).astype(np.float32)
    if pts.shape[0] <= 1:
        return pts.astype(np.float32)

    order = np.lexsort((pts[:, 1], pts[:, 0]))
    pts = pts[order]

    def cross(o, a, b):
        return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    hull = np.asarray(lower[:-1] + upper[:-1], dtype=np.float32)
    if hull.shape[0] == 0:
        hull = pts[:1].astype(np.float32)
    return hull.astype(np.float32)


def minimum_area_rect_xy(points_xy: np.ndarray):
    pts = np.asarray(points_xy, dtype=np.float32)
    if pts.shape[0] == 0:
        return {
            "center_xy": np.zeros((2,), dtype=np.float32),
            "dims_xy": np.zeros((2,), dtype=np.float32),
            "angle": 0.0,
        }
    if pts.shape[0] == 1:
        return {
            "center_xy": pts[0].astype(np.float32),
            "dims_xy": np.zeros((2,), dtype=np.float32),
            "angle": 0.0,
        }

    hull = convex_hull_2d(pts)
    if hull.shape[0] == 1:
        return {
            "center_xy": hull[0].astype(np.float32),
            "dims_xy": np.zeros((2,), dtype=np.float32),
            "angle": 0.0,
        }

    if hull.shape[0] == 2:
        edge = hull[1] - hull[0]
        angle = float(math.atan2(edge[1], edge[0]))
        ca = math.cos(-angle)
        sa = math.sin(-angle)
        rot = np.array([[ca, -sa], [sa, ca]], dtype=np.float32)
        hull_r = hull @ rot.T
        mn = hull_r.min(axis=0)
        mx = hull_r.max(axis=0)
        center_r = 0.5 * (mn + mx)
        center_xy = center_r @ rot
        dims_xy = (mx - mn).astype(np.float32)
        return {
            "center_xy": center_xy.astype(np.float32),
            "dims_xy": dims_xy.astype(np.float32),
            "angle": float(angle),
        }

    best = None
    for i in range(hull.shape[0]):
        j = (i + 1) % hull.shape[0]
        edge = hull[j] - hull[i]
        if float(np.linalg.norm(edge)) <= 1e-6:
            continue
        angle = float(math.atan2(edge[1], edge[0]))
        ca = math.cos(-angle)
        sa = math.sin(-angle)
        rot = np.array([[ca, -sa], [sa, ca]], dtype=np.float32)
        hull_r = hull @ rot.T
        mn = hull_r.min(axis=0)
        mx = hull_r.max(axis=0)
        dims_xy = (mx - mn).astype(np.float32)
        area = float(max(dims_xy[0], 1e-6) * max(dims_xy[1], 1e-6))
        center_r = 0.5 * (mn + mx)
        center_xy = center_r @ rot
        cand = {
            "center_xy": center_xy.astype(np.float32),
            "dims_xy": dims_xy.astype(np.float32),
            "angle": float(angle),
            "area": float(area),
        }
        if best is None or cand["area"] < best["area"]:
            best = cand

    if best is None:
        center_xy = pts.mean(axis=0).astype(np.float32)
        dims_xy = (pts.max(axis=0) - pts.min(axis=0)).astype(np.float32)
        best = {"center_xy": center_xy, "dims_xy": dims_xy, "angle": 0.0, "area": float(np.prod(dims_xy + 1e-6))}

    return {
        "center_xy": best["center_xy"].astype(np.float32),
        "dims_xy": best["dims_xy"].astype(np.float32),
        "angle": float(best["angle"]),
    }


def oriented_box_stats_xyz(points_xyz: np.ndarray):
    pts = np.asarray(points_xyz, dtype=np.float32)
    if pts.shape[0] == 0:
        return {
            "center": np.zeros((3,), dtype=np.float32),
            "dims": np.zeros((3,), dtype=np.float32),
            "angle": 0.0,
            "min_xyz": np.zeros((3,), dtype=np.float32),
            "max_xyz": np.zeros((3,), dtype=np.float32),
        }

    rect = minimum_area_rect_xy(pts[:, :2])
    dims_xy = np.sort(rect["dims_xy"].astype(np.float32))
    zmin = float(np.min(pts[:, 2]))
    zmax = float(np.max(pts[:, 2]))
    height = float(zmax - zmin)
    center = np.array([rect["center_xy"][0], rect["center_xy"][1], 0.5 * (zmin + zmax)], dtype=np.float32)
    dims = np.array([float(dims_xy[1]), float(dims_xy[0]), height], dtype=np.float32)
    return {
        "center": center,
        "dims": dims.astype(np.float32),
        "angle": float(rect["angle"]),
        "min_xyz": pts.min(axis=0).astype(np.float32),
        "max_xyz": pts.max(axis=0).astype(np.float32),
    }


def analyze_gt_instances(xyz: np.ndarray, inst: np.ndarray):
    inst_info = []
    if inst is None:
        return inst_info

    for uid in np.unique(inst[inst > 0]):
        idx = np.where(inst == uid)[0]
        if idx.size == 0:
            continue
        pts = xyz[idx]
        center = pts.mean(axis=0).astype(np.float32)
        min_xyz = pts.min(axis=0).astype(np.float32)
        max_xyz = pts.max(axis=0).astype(np.float32)
        size_xyz = (max_xyz - min_xyz).astype(np.float32)
        obox = oriented_box_stats_xyz(pts)
        dist = np.linalg.norm(pts - center[None, :], axis=1)
        rad90 = float(np.percentile(dist, 90))
        inst_info.append(
            {
                "id": int(uid),
                "center": center,
                "min_xyz": min_xyz,
                "max_xyz": max_xyz,
                "size_xyz": size_xyz,
                "obox_dims": obox["dims"].astype(np.float32),
                "obox_center": obox["center"].astype(np.float32),
                "num_points": int(idx.size),
                "rad90": rad90,
            }
        )
    return inst_info


def load_scenes(
    files,
    input_dim: int,
    normalize_mode: str,
    scale_quantile: float,
    scale_clip_min: float,
    scale_clip_max: float,
    abs_coord_scale: float,
    use_abs_coords: bool = False,
    coord_unit_scale: float = 0.0,
    auto_unit_normalize: bool = True,
    auto_mm_threshold: float = 200.0,
    pre_downsample_voxel: float = 0.0,
    logger=None,
):
    scenes = []
    downsample_ratios = []
    unit_scales = []
    for fp in files:
        try:
            xyz_raw, feats_raw, inst = read_csv_points(fp, input_dim=input_dim, require_label=(True if input_dim else False))
        except Exception as exc:
            if logger is not None:
                logger.warning(f"Skipping file (load error): {fp} | {exc}")
            continue
        xyz, feats_raw, inst, prep_meta = prepare_scene_points(
            xyz_raw,
            feats_raw,
            inst=inst,
            coord_unit_scale=coord_unit_scale,
            auto_unit_normalize=auto_unit_normalize,
            auto_mm_threshold=auto_mm_threshold,
            pre_downsample_voxel=pre_downsample_voxel,
        )
        xyz_scene, point_inputs, norm_meta = build_point_inputs(
            xyz,
            feats_raw,
            input_dim=input_dim,
            normalize_mode=normalize_mode,
            scale_quantile=scale_quantile,
            scale_clip_min=scale_clip_min,
            scale_clip_max=scale_clip_max,
            abs_coord_scale=abs_coord_scale,
            use_abs_coords=use_abs_coords,
        )
        inst_info = analyze_gt_instances(xyz, inst) if inst is not None else []
        scene = {
            "path": fp,
            "name": os.path.basename(fp),
            "xyz": xyz.astype(np.float32),
            "xyz_scene": xyz_scene.astype(np.float32),
            "point_inputs": point_inputs.astype(np.float32),
            "inst": inst.astype(np.int64) if inst is not None else None,
            "norm_meta": norm_meta,
            "inst_info": inst_info,
            "fg_ratio": float(np.mean(inst > 0)) if inst is not None else 0.0,
            "num_points": int(xyz.shape[0]),
            "orig_num_points": int(prep_meta["orig_points"]),
            "downsample_ratio": float(prep_meta["downsample_ratio"]),
            "coord_unit_scale": float(prep_meta["coord_unit_scale"]),
            "scene_scale": float(norm_meta.get("scale", 1.0)),
            "use_abs_coords": int(norm_meta.get("use_abs_coords", 0)),
            "extra_dim": int(norm_meta.get("extra_dim", max(0, input_dim - 3))),
        }
        scenes.append(scene)
        downsample_ratios.append(float(prep_meta["downsample_ratio"]))
        unit_scales.append(float(prep_meta["coord_unit_scale"]))

    if logger is not None and len(scenes) > 0:
        fg_ratio = np.mean([s["fg_ratio"] for s in scenes])
        logger.info(
            f"Loaded {len(scenes)} scenes | avg points/scene={np.mean([s['num_points'] for s in scenes]):.0f} | "
            f"avg fg ratio={fg_ratio:.4f} | avg keep ratio={np.mean(downsample_ratios):.4f} | "
            f"unit_scales={sorted(set([round(x, 6) for x in unit_scales]))}"
        )
    return scenes


def estimate_gt_priors(scenes):
    count_list = []
    rad_list = []
    sx, sy, sz = [], [], []
    centers = []
    insts_per_scene = []
    fg_points_per_scene = []
    fg_ratio_per_scene = []

    for s in scenes:
        insts_per_scene.append(len(s["inst_info"]))
        if s["inst"] is not None:
            fg_count = int((s["inst"] > 0).sum())
            fg_points_per_scene.append(float(fg_count))
            fg_ratio_per_scene.append(float(fg_count / max(1, s["inst"].shape[0])))
        for obj in s["inst_info"]:
            count_list.append(float(obj["num_points"]))
            rad_list.append(float(obj["rad90"]))
            sx.append(float(obj["size_xyz"][0]))
            sy.append(float(obj["size_xyz"][1]))
            sz.append(float(obj["size_xyz"][2]))
            centers.append(obj["center"].astype(np.float32))

    priors = {}
    if len(count_list) == 0:
        return priors

    priors["num_points_p10"] = float(np.percentile(count_list, 10))
    priors["num_points_p50"] = float(np.percentile(count_list, 50))
    priors["num_points_p90"] = float(np.percentile(count_list, 90))
    priors["rad90_p10"] = float(np.percentile(rad_list, 10))
    priors["rad90_p50"] = float(np.percentile(rad_list, 50))
    priors["rad90_p90"] = float(np.percentile(rad_list, 90))
    priors["size_x_p10"] = float(np.percentile(sx, 10))
    priors["size_x_p50"] = float(np.percentile(sx, 50))
    priors["size_x_p90"] = float(np.percentile(sx, 90))
    priors["size_y_p10"] = float(np.percentile(sy, 10))
    priors["size_y_p50"] = float(np.percentile(sy, 50))
    priors["size_y_p90"] = float(np.percentile(sy, 90))
    priors["size_z_p10"] = float(np.percentile(sz, 10))
    priors["size_z_p50"] = float(np.percentile(sz, 50))
    priors["size_z_p90"] = float(np.percentile(sz, 90))
    priors["max_instances_per_scene"] = int(np.max(insts_per_scene)) if len(insts_per_scene) > 0 else 1
    priors["inst_per_scene_p90"] = float(np.percentile(insts_per_scene, 90)) if len(insts_per_scene) > 0 else 1.0
    centers = np.stack(centers, axis=0).astype(np.float32)
    priors["center_mean_x"] = float(centers[:, 0].mean())
    priors["center_mean_y"] = float(centers[:, 1].mean())
    priors["center_mean_z"] = float(centers[:, 2].mean())
    priors["center_std_x"] = float(centers[:, 0].std() + 1e-6)
    priors["center_std_y"] = float(centers[:, 1].std() + 1e-6)
    priors["center_std_z"] = float(centers[:, 2].std() + 1e-6)
    if len(fg_points_per_scene) > 0:
        priors["fg_points_p10"] = float(np.percentile(fg_points_per_scene, 10))
        priors["fg_points_p50"] = float(np.percentile(fg_points_per_scene, 50))
        priors["fg_points_p90"] = float(np.percentile(fg_points_per_scene, 90))
    if len(fg_ratio_per_scene) > 0:
        priors["fg_ratio_p10"] = float(np.percentile(fg_ratio_per_scene, 10))
        priors["fg_ratio_p50"] = float(np.percentile(fg_ratio_per_scene, 50))
        priors["fg_ratio_p90"] = float(np.percentile(fg_ratio_per_scene, 90))
    return priors


def sample_keep_indices_with_fg_priority(inst: np.ndarray, max_points: int):
    N = inst.shape[0]
    if max_points <= 0 or N <= max_points:
        return np.arange(N, dtype=np.int64)

    fg_idx = np.where(inst > 0)[0]
    bg_idx = np.where(inst == 0)[0]

    if fg_idx.size >= max_points:
        keep = np.random.choice(fg_idx, max_points, replace=False)
    else:
        need_bg = max_points - fg_idx.size
        if bg_idx.size > need_bg:
            bg_keep = np.random.choice(bg_idx, need_bg, replace=False)
        else:
            bg_keep = bg_idx
        keep = np.concatenate([fg_idx, bg_keep], axis=0)

    np.random.shuffle(keep)
    return keep.astype(np.int64)


class CropTrainDataset(Dataset):
    def __init__(
        self,
        scenes,
        voxel_size=0.02,
        max_points_per_voxel=50,
        max_voxels=0,
        max_points=24000,
        feature_mode="safe",
        voxel_feature_mode="extra_only",
        samples_per_scene=12,
        crop_half_x=12.0,
        crop_half_y=12.0,
        crop_half_z=12.0,
        positive_fraction=0.45,
        hard_negative_fraction=0.35,
        hard_neg_shell_min=8.0,
        hard_neg_shell_max=22.0,
        hard_neg_max_fg_points=40,
        aug_jitter_xyz_std=0.02,
        aug_dropout=0.0,
        aug_rotate_z_deg=0.0,
        aug_flip_x_prob=0.0,
        aug_flip_y_prob=0.0,
        aug_bg_dropout_prob=0.0,
        aug_bg_dropout_min=0.0,
        aug_bg_dropout_max=0.0,
        aug_normal_noise_std=0.0,
        positive_context_scale_min=1.2,
        positive_context_scale_max=1.7,
        positive_safe_margin=0.25,
        positive_jitter_frac=0.50,
        positive_min_coverage=0.92,
        hard_neg_near_obj_prob=0.60,
        hard_neg_context_margin=2.50,
        hard_neg_exclude_margin=0.60,
    ):
        self.scenes = scenes
        self.voxel_size = float(voxel_size)
        self.max_points_per_voxel = int(max_points_per_voxel)
        self.max_voxels = int(max_voxels)
        self.max_points = int(max_points)
        self.feature_mode = str(feature_mode)
        self.voxel_feature_mode = str(voxel_feature_mode)

        self.samples_per_scene = int(samples_per_scene)
        self.crop_half_x = float(crop_half_x)
        self.crop_half_y = float(crop_half_y)
        self.crop_half_z = float(crop_half_z)
        self.positive_fraction = float(positive_fraction)
        self.hard_negative_fraction = float(hard_negative_fraction)
        self.hard_neg_shell_min = float(hard_neg_shell_min)
        self.hard_neg_shell_max = float(hard_neg_shell_max)
        self.hard_neg_max_fg_points = int(hard_neg_max_fg_points)
        self.aug_jitter_xyz_std = float(aug_jitter_xyz_std)
        self.aug_dropout = float(aug_dropout)
        self.aug_rotate_z_deg = float(aug_rotate_z_deg)
        self.aug_flip_x_prob = float(aug_flip_x_prob)
        self.aug_flip_y_prob = float(aug_flip_y_prob)
        self.aug_bg_dropout_prob = float(aug_bg_dropout_prob)
        self.aug_bg_dropout_min = float(aug_bg_dropout_min)
        self.aug_bg_dropout_max = float(aug_bg_dropout_max)
        self.aug_normal_noise_std = float(aug_normal_noise_std)
        self.positive_context_scale_min = float(positive_context_scale_min)
        self.positive_context_scale_max = float(max(positive_context_scale_min, positive_context_scale_max))
        self.positive_safe_margin = float(positive_safe_margin)
        self.positive_jitter_frac = float(positive_jitter_frac)
        self.positive_min_coverage = float(positive_min_coverage)
        self.hard_neg_near_obj_prob = float(hard_neg_near_obj_prob)
        self.hard_neg_context_margin = float(hard_neg_context_margin)
        self.hard_neg_exclude_margin = float(hard_neg_exclude_margin)

    def __len__(self):
        return len(self.scenes) * self.samples_per_scene

    def _crop_mask(self, xyz: np.ndarray, center: np.ndarray, half_xyz=None):
        if half_xyz is None:
            half_xyz = np.array([self.crop_half_x, self.crop_half_y, self.crop_half_z], dtype=np.float32)
        else:
            half_xyz = np.asarray(half_xyz, dtype=np.float32)
        return (
            (xyz[:, 0] >= center[0] - half_xyz[0])
            & (xyz[:, 0] <= center[0] + half_xyz[0])
            & (xyz[:, 1] >= center[1] - half_xyz[1])
            & (xyz[:, 1] <= center[1] + half_xyz[1])
            & (xyz[:, 2] >= center[2] - half_xyz[2])
            & (xyz[:, 2] <= center[2] + half_xyz[2])
        )

    def _sample_positive_crop(self, scene):
        obj = scene["inst_info"][np.random.randint(len(scene["inst_info"]))]
        base_half = np.array([self.crop_half_x, self.crop_half_y, self.crop_half_z], dtype=np.float32)
        scale = float(np.random.uniform(self.positive_context_scale_min, self.positive_context_scale_max))
        crop_half = base_half * scale

        obj_half = 0.5 * np.asarray(obj["size_xyz"], dtype=np.float32)
        crop_half = np.maximum(crop_half, obj_half + float(self.positive_safe_margin))
        avail = np.maximum(crop_half - obj_half - float(self.positive_safe_margin), 0.0)
        jitter_span = avail * float(np.clip(self.positive_jitter_frac, 0.0, 1.0))
        jitter = np.random.uniform(-jitter_span, jitter_span).astype(np.float32)
        center = obj["center"].astype(np.float32) + jitter
        return center.astype(np.float32), crop_half.astype(np.float32), obj

    def _sample_random_bg_center(self, scene):
        inst = scene["inst"]
        bg_idx = np.where(inst == 0)[0]
        if bg_idx.size == 0:
            return scene["xyz"][np.random.randint(scene["xyz"].shape[0])].astype(np.float32)
        j = np.random.choice(bg_idx)
        return scene["xyz"][j].astype(np.float32)

    def _object_coverage(self, inst: np.ndarray, mask: np.ndarray, obj_id: int):
        obj_mask = inst == int(obj_id)
        denom = int(obj_mask.sum())
        if denom <= 0:
            return 1.0
        covered = int(np.sum(mask & obj_mask))
        return float(covered / max(denom, 1))

    def _sample_near_object_bg_center(self, scene):
        xyz = scene["xyz"]
        inst = scene["inst"]
        bg_mask = inst == 0
        if not np.any(bg_mask):
            return None

        for _ in range(10):
            obj = scene["inst_info"][np.random.randint(len(scene["inst_info"]))]
            outer_min = np.asarray(obj["min_xyz"], dtype=np.float32) - float(self.hard_neg_context_margin)
            outer_max = np.asarray(obj["max_xyz"], dtype=np.float32) + float(self.hard_neg_context_margin)
            inner_min = np.asarray(obj["min_xyz"], dtype=np.float32) - float(self.hard_neg_exclude_margin)
            inner_max = np.asarray(obj["max_xyz"], dtype=np.float32) + float(self.hard_neg_exclude_margin)

            in_outer = (
                (xyz[:, 0] >= outer_min[0])
                & (xyz[:, 0] <= outer_max[0])
                & (xyz[:, 1] >= outer_min[1])
                & (xyz[:, 1] <= outer_max[1])
                & (xyz[:, 2] >= outer_min[2])
                & (xyz[:, 2] <= outer_max[2])
            )
            in_inner = (
                (xyz[:, 0] >= inner_min[0])
                & (xyz[:, 0] <= inner_max[0])
                & (xyz[:, 1] >= inner_min[1])
                & (xyz[:, 1] <= inner_max[1])
                & (xyz[:, 2] >= inner_min[2])
                & (xyz[:, 2] <= inner_max[2])
            )
            cand_idx = np.where(bg_mask & in_outer & (~in_inner))[0]
            if cand_idx.size == 0:
                continue
            pick = int(np.random.choice(cand_idx))
            cand = xyz[pick].astype(np.float32)
            mask = self._crop_mask(xyz, cand)
            if mask.sum() < 200:
                continue
            if int((inst[mask] > 0).sum()) <= self.hard_neg_max_fg_points:
                return cand
        return None

    def _sample_hard_negative_center(self, scene):
        xyz = scene["xyz"]
        inst = scene["inst"]
        if np.random.rand() < self.hard_neg_near_obj_prob:
            cand = self._sample_near_object_bg_center(scene)
            if cand is not None:
                return cand.astype(np.float32)
        for _ in range(20):
            obj = scene["inst_info"][np.random.randint(len(scene["inst_info"]))]
            center = obj["center"].copy()
            direction = np.random.normal(0.0, 1.0, size=(3,)).astype(np.float32)
            direction[2] *= 0.5
            norm = np.linalg.norm(direction) + 1e-6
            direction = direction / norm
            shell_r = np.random.uniform(self.hard_neg_shell_min, self.hard_neg_shell_max)
            cand = center + direction * shell_r
            mask = self._crop_mask(xyz, cand)
            if mask.sum() < 200:
                continue
            if int((inst[mask] > 0).sum()) <= self.hard_neg_max_fg_points:
                return cand.astype(np.float32)
        return self._sample_random_bg_center(scene)

    def _apply_point_aug(self, xyz_scene_crop, point_inputs_crop, inst_crop, extra_dim=0, use_abs_coords=False):
        xyz_scene_crop = xyz_scene_crop.astype(np.float32, copy=True)
        point_inputs_crop = point_inputs_crop.astype(np.float32, copy=True)
        inst_crop = inst_crop.astype(np.int64, copy=False)
        extra_dim = int(max(0, extra_dim))
        extra_offset = 3 + (3 if bool(use_abs_coords) else 0)

        if self.aug_flip_x_prob > 0 and np.random.rand() < self.aug_flip_x_prob:
            xyz_scene_crop[:, 0] *= -1.0
            point_inputs_crop[:, 0] *= -1.0
            if bool(use_abs_coords) and point_inputs_crop.shape[1] >= 6:
                point_inputs_crop[:, 3] *= -1.0
            if extra_dim >= 3 and point_inputs_crop.shape[1] >= extra_offset + 3:
                point_inputs_crop[:, extra_offset] *= -1.0

        if self.aug_flip_y_prob > 0 and np.random.rand() < self.aug_flip_y_prob:
            xyz_scene_crop[:, 1] *= -1.0
            point_inputs_crop[:, 1] *= -1.0
            if bool(use_abs_coords) and point_inputs_crop.shape[1] >= 6:
                point_inputs_crop[:, 4] *= -1.0
            if extra_dim >= 3 and point_inputs_crop.shape[1] >= extra_offset + 3:
                point_inputs_crop[:, extra_offset + 1] *= -1.0

        if self.aug_rotate_z_deg > 0:
            ang = float(np.deg2rad(np.random.uniform(-self.aug_rotate_z_deg, self.aug_rotate_z_deg)))
            ca = math.cos(ang)
            sa = math.sin(ang)
            rot = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
            xyz_scene_crop = xyz_scene_crop @ rot.T
            point_inputs_crop[:, :3] = xyz_scene_crop
            if bool(use_abs_coords) and point_inputs_crop.shape[1] >= 6:
                point_inputs_crop[:, 3:6] = point_inputs_crop[:, 3:6] @ rot.T
            if extra_dim >= 3 and point_inputs_crop.shape[1] >= extra_offset + 3:
                point_inputs_crop[:, extra_offset : extra_offset + 3] = point_inputs_crop[:, extra_offset : extra_offset + 3] @ rot.T

        if self.aug_jitter_xyz_std > 0:
            jitter = np.random.normal(0.0, self.aug_jitter_xyz_std, size=xyz_scene_crop.shape).astype(np.float32)
            xyz_scene_crop = xyz_scene_crop + jitter
            point_inputs_crop[:, :3] = xyz_scene_crop

        if self.aug_normal_noise_std > 0 and extra_dim >= 3 and point_inputs_crop.shape[1] >= extra_offset + 3:
            normals = point_inputs_crop[:, extra_offset : extra_offset + 3]
            normals = normals + np.random.normal(0.0, self.aug_normal_noise_std, size=normals.shape).astype(np.float32)
            normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-6)
            point_inputs_crop[:, extra_offset : extra_offset + 3] = normals.astype(np.float32)

        fg_mask = inst_crop > 0
        bg_idx = np.where(~fg_mask)[0]
        if bg_idx.size > 0:
            drop_ratio = 0.0
            if self.aug_bg_dropout_prob > 0 and np.random.rand() < self.aug_bg_dropout_prob:
                lo = max(0.0, self.aug_bg_dropout_min)
                hi = max(lo, self.aug_bg_dropout_max)
                drop_ratio = float(np.random.uniform(lo, hi))
            elif self.aug_dropout > 0:
                drop_ratio = float(np.random.uniform(0.0, self.aug_dropout))

            if drop_ratio > 0:
                keep = np.ones((xyz_scene_crop.shape[0],), dtype=bool)
                keep_bg = np.random.rand(bg_idx.size) >= drop_ratio
                keep[bg_idx] = keep_bg
                min_keep = int(max(32, int(fg_mask.sum())))
                if keep.sum() > min_keep:
                    xyz_scene_crop = xyz_scene_crop[keep]
                    point_inputs_crop = point_inputs_crop[keep]
                    return xyz_scene_crop, point_inputs_crop, keep

        return xyz_scene_crop, point_inputs_crop, None

    def __getitem__(self, idx):
        scene = self.scenes[idx % len(self.scenes)]
        xyz = scene["xyz"]
        xyz_scene = scene["xyz_scene"]
        pin = scene["point_inputs"]
        inst = scene["inst"]
        extra_dim = int(scene.get("extra_dim", max(0, pin.shape[1] - 3)))
        use_abs_coords = bool(int(scene.get("use_abs_coords", 0)))

        r = np.random.rand()
        crop_half = None
        target_obj = None
        if len(scene["inst_info"]) > 0 and r < self.positive_fraction:
            center, crop_half, target_obj = self._sample_positive_crop(scene)
        elif len(scene["inst_info"]) > 0 and r < self.positive_fraction + self.hard_negative_fraction:
            center = self._sample_hard_negative_center(scene)
        else:
            center = self._sample_random_bg_center(scene)

        mask = self._crop_mask(xyz, center, crop_half)

        if target_obj is not None and self.positive_min_coverage > 0:
            coverage = self._object_coverage(inst, mask, int(target_obj["id"]))
            if coverage < self.positive_min_coverage:
                target_half = np.maximum(
                    np.asarray(crop_half if crop_half is not None else [self.crop_half_x, self.crop_half_y, self.crop_half_z], dtype=np.float32),
                    0.5 * np.asarray(target_obj["size_xyz"], dtype=np.float32) + float(self.positive_safe_margin) + 0.5,
                )
                center = np.asarray(target_obj["center"], dtype=np.float32)
                mask = self._crop_mask(xyz, center, target_half)

        if mask.sum() < 200:
            center = xyz[np.random.randint(xyz.shape[0])].astype(np.float32)
            mask = self._crop_mask(xyz, center)

        xyz_crop = xyz[mask]
        xyz_scene_crop = xyz_scene[mask]
        pin_crop = pin[mask]
        inst_crop = inst[mask]

        keep_idx = sample_keep_indices_with_fg_priority(inst_crop, self.max_points)
        xyz_crop = xyz_crop[keep_idx]
        xyz_scene_crop = xyz_scene_crop[keep_idx]
        pin_crop = pin_crop[keep_idx]
        inst_crop = inst_crop[keep_idx]

        xyz_scene_crop, pin_crop, keep_aug = self._apply_point_aug(
            xyz_scene_crop,
            pin_crop,
            inst_crop,
            extra_dim=extra_dim,
            use_abs_coords=use_abs_coords,
        )
        if keep_aug is not None:
            xyz_crop = xyz_crop[keep_aug]
            inst_crop = inst_crop[keep_aug]

        fg_target = (inst_crop > 0).astype(np.float32)
        center_target, offset_target, offset_weight, size_target, size_weight = build_instance_aux_targets(
            xyz_scene_crop,
            inst_crop,
        )
        vin_crop = select_voxel_point_features(
            xyz_scene_crop,
            pin_crop,
            extra_dim=extra_dim,
            voxel_feature_mode=self.voxel_feature_mode,
        )

        vcoords, vfeats, _, p2v, keep_mask = voxelize_with_mapping(
            xyz_scene_crop,
            voxel_size=self.voxel_size,
            point_features=vin_crop,
            max_points_per_voxel=self.max_points_per_voxel,
            max_voxels=self.max_voxels,
            feature_mode=self.feature_mode,
            quantize="floor",
        )

        km = keep_mask.cpu().numpy().astype(bool)
        xyz_crop = xyz_crop[km]
        xyz_scene_crop = xyz_scene_crop[km]
        pin_crop = pin_crop[km]
        inst_crop = inst_crop[km]
        fg_target = fg_target[km]
        center_target = center_target[km]
        offset_target = offset_target[km]
        offset_weight = offset_weight[km]
        size_target = size_target[km]
        size_weight = size_weight[km]
        p2v = p2v[keep_mask]

        pts_scene_t = torch.from_numpy(xyz_scene_crop).to(torch.float32)
        raw_xyz_t = torch.from_numpy(xyz_crop).to(torch.float32)
        pin_t = torch.from_numpy(pin_crop).to(torch.float32)
        fg_t = torch.from_numpy(fg_target.astype(np.float32)).to(torch.float32)
        center_t = torch.from_numpy(center_target.astype(np.float32)).to(torch.float32)
        offset_t = torch.from_numpy(offset_target.astype(np.float32)).to(torch.float32)
        offset_w_t = torch.from_numpy(offset_weight.astype(np.float32)).to(torch.float32)
        size_t = torch.from_numpy(size_target.astype(np.float32)).to(torch.float32)
        size_w_t = torch.from_numpy(size_weight.astype(np.float32)).to(torch.float32)
        scene_scale_t = torch.full((xyz_scene_crop.shape[0],), float(scene.get("scene_scale", 1.0)), dtype=torch.float32)
        inst_t = torch.from_numpy(inst_crop.astype(np.int64)).to(torch.int64)

        return vcoords, vfeats, pts_scene_t, raw_xyz_t, pin_t, p2v, fg_t, center_t, offset_t, offset_w_t, size_t, size_w_t, scene_scale_t, inst_t, scene["name"]


class FullSceneDataset(Dataset):
    def __init__(
        self,
        scenes,
        voxel_size=0.02,
        max_points_per_voxel=50,
        max_voxels=0,
        feature_mode="safe",
        voxel_feature_mode="extra_only",
    ):
        self.scenes = scenes
        self.voxel_size = float(voxel_size)
        self.max_points_per_voxel = int(max_points_per_voxel)
        self.max_voxels = int(max_voxels)
        self.feature_mode = str(feature_mode)
        self.voxel_feature_mode = str(voxel_feature_mode)

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        scene = self.scenes[idx]
        xyz = scene["xyz"]
        xyz_scene = scene["xyz_scene"]
        pin = scene["point_inputs"]
        inst = scene["inst"]
        extra_dim = int(scene.get("extra_dim", max(0, pin.shape[1] - 3)))

        fg_target = (inst > 0).astype(np.float32)
        center_target, offset_target, offset_weight, size_target, size_weight = build_instance_aux_targets(
            xyz_scene,
            inst,
        )
        vin = select_voxel_point_features(
            xyz_scene,
            pin,
            extra_dim=extra_dim,
            voxel_feature_mode=self.voxel_feature_mode,
        )

        vcoords, vfeats, _, p2v, keep_mask = voxelize_with_mapping(
            xyz_scene,
            voxel_size=self.voxel_size,
            point_features=vin,
            max_points_per_voxel=self.max_points_per_voxel,
            max_voxels=self.max_voxels,
            feature_mode=self.feature_mode,
            quantize="floor",
        )

        km = keep_mask.cpu().numpy().astype(bool)
        xyz = xyz[km]
        xyz_scene = xyz_scene[km]
        pin = pin[km]
        inst = inst[km]
        fg_target = fg_target[km]
        center_target = center_target[km]
        offset_target = offset_target[km]
        offset_weight = offset_weight[km]
        size_target = size_target[km]
        size_weight = size_weight[km]
        p2v = p2v[keep_mask]

        pts_scene_t = torch.from_numpy(xyz_scene).to(torch.float32)
        raw_xyz_t = torch.from_numpy(xyz).to(torch.float32)
        pin_t = torch.from_numpy(pin).to(torch.float32)
        fg_t = torch.from_numpy(fg_target.astype(np.float32)).to(torch.float32)
        center_t = torch.from_numpy(center_target.astype(np.float32)).to(torch.float32)
        offset_t = torch.from_numpy(offset_target.astype(np.float32)).to(torch.float32)
        offset_w_t = torch.from_numpy(offset_weight.astype(np.float32)).to(torch.float32)
        size_t = torch.from_numpy(size_target.astype(np.float32)).to(torch.float32)
        size_w_t = torch.from_numpy(size_weight.astype(np.float32)).to(torch.float32)
        scene_scale_t = torch.full((xyz_scene.shape[0],), float(scene.get("scene_scale", 1.0)), dtype=torch.float32)
        inst_t = torch.from_numpy(inst.astype(np.int64)).to(torch.int64)

        return vcoords, vfeats, pts_scene_t, raw_xyz_t, pin_t, p2v, fg_t, center_t, offset_t, offset_w_t, size_t, size_w_t, scene_scale_t, inst_t, scene["name"]


def collate_fn(batch):
    coords_list, feats_list = [], []
    pts_scene_list, raw_xyz_list, pin_list, p2v_list, pb_list = [], [], [], [], []
    fg_list, center_list, offset_list, offset_w_list, size_list, size_w_list, scene_scale_list, inst_list = [], [], [], [], [], [], [], []
    names = []

    voxel_base = 0
    for b, (vcoords, vfeats, pts_scene, raw_xyz, pin, p2v, fg, center, offset, offset_w, size_t, size_w, scene_scale, inst, name) in enumerate(batch):
        vcoords_b = torch.cat([torch.full((vcoords.shape[0], 1), b, dtype=torch.int32), vcoords], dim=1)
        coords_list.append(vcoords_b)
        feats_list.append(vfeats)

        pts_scene_list.append(pts_scene)
        raw_xyz_list.append(raw_xyz)
        pin_list.append(pin)
        fg_list.append(fg)
        center_list.append(center)
        offset_list.append(offset)
        offset_w_list.append(offset_w)
        size_list.append(size_t)
        size_w_list.append(size_w)
        scene_scale_list.append(scene_scale)
        inst_list.append(inst)

        p2v_list.append(p2v + voxel_base)
        pb_list.append(torch.full((pts_scene.shape[0],), b, dtype=torch.int64))

        voxel_base += vcoords.shape[0]
        names.append(name)

    coords_b = torch.cat(coords_list, dim=0)
    feats_b = torch.cat(feats_list, dim=0)
    pts_scene_b = torch.cat(pts_scene_list, dim=0)
    raw_xyz_b = torch.cat(raw_xyz_list, dim=0)
    pin_b = torch.cat(pin_list, dim=0)
    p2v_b = torch.cat(p2v_list, dim=0)
    pb_b = torch.cat(pb_list, dim=0)
    fg_b = torch.cat(fg_list, dim=0)
    center_b = torch.cat(center_list, dim=0)
    offset_b = torch.cat(offset_list, dim=0)
    offset_w_b = torch.cat(offset_w_list, dim=0)
    size_b = torch.cat(size_list, dim=0)
    size_w_b = torch.cat(size_w_list, dim=0)
    scene_scale_b = torch.cat(scene_scale_list, dim=0)
    inst_b = torch.cat(inst_list, dim=0)

    return coords_b, feats_b, pts_scene_b, raw_xyz_b, pin_b, p2v_b, pb_b, fg_b, center_b, offset_b, offset_w_b, size_b, size_w_b, scene_scale_b, inst_b, names


def make_norm(num_channels: int, norm: str = "gn"):
    norm = str(norm).lower()
    if norm == "bn":
        return nn.BatchNorm1d(num_channels, eps=1e-3, momentum=0.01)
    for g in (16, 8, 4, 2, 1):
        if num_channels % g == 0:
            return nn.GroupNorm(g, num_channels, eps=1e-5, affine=True)
    return nn.GroupNorm(1, num_channels, eps=1e-5, affine=True)


def _bias_from_prior(prior: float):
    p = float(np.clip(prior, 1e-4, 1.0 - 1e-4))
    return math.log(p / (1.0 - p))


def scatter_scene_feature_stats(point_feat: torch.Tensor, point_batch_ids: torch.Tensor, batch_size: int):
    feat_dim = int(point_feat.shape[1])
    mean = point_feat.new_zeros((batch_size, feat_dim))
    sq = point_feat.new_zeros((batch_size, feat_dim))
    count = point_feat.new_zeros((batch_size, 1))

    ones = point_feat.new_ones((point_feat.shape[0], 1))
    mean.index_add_(0, point_batch_ids, point_feat)
    sq.index_add_(0, point_batch_ids, point_feat * point_feat)
    count.index_add_(0, point_batch_ids, ones)

    count_safe = count.clamp_min(1.0)
    mean = mean / count_safe
    var = (sq / count_safe) - mean * mean
    std = torch.sqrt(var.clamp_min(0.0) + 1e-6)

    max_feat = point_feat.new_zeros((batch_size, feat_dim))
    for b in range(int(batch_size)):
        mask = point_batch_ids == int(b)
        if bool(mask.any()):
            max_feat[b] = point_feat[mask].max(dim=0)[0]

    return mean, max_feat, std


def unpack_model_outputs(outputs):
    if isinstance(outputs, dict):
        return outputs

    fg_logits = outputs
    zeros = fg_logits.new_zeros(fg_logits.shape)
    offset_zeros = fg_logits.new_zeros((fg_logits.shape[0], 3))
    size_zeros = fg_logits.new_zeros((fg_logits.shape[0], 3))
    return {
        "fg_logits": fg_logits,
        "center_logits": zeros,
        "offset_pred": offset_zeros,
        "size_pred": size_zeros,
    }


class SparseResBlock(nn.Module):
    def __init__(self, channels: int, norm: str = "gn", indice_key: str = "subm", dropout: float = 0.0, dilation: int = 1):
        super().__init__()
        dilation = int(max(1, dilation))
        padding = dilation
        self.conv1 = spconv.SubMConv3d(
            channels,
            channels,
            3,
            padding=padding,
            dilation=dilation,
            bias=False,
            indice_key=indice_key,
        )
        self.norm1 = make_norm(channels, norm)
        self.conv2 = spconv.SubMConv3d(
            channels,
            channels,
            3,
            padding=padding,
            dilation=dilation,
            bias=False,
            indice_key=indice_key,
        )
        self.norm2 = make_norm(channels, norm)
        self.dropout = nn.Dropout(p=float(dropout)) if float(dropout) > 0 else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x):
        identity = x.features
        out = self.conv1(x)
        out = out.replace_feature(self.act(self.norm1(out.features)))
        out = self.conv2(out)
        feat = self.dropout(self.norm2(out.features))
        out = out.replace_feature(self.act(feat + identity))
        return out


class SparseFGNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        point_in_dim: int,
        init_channels: int = 40,
        norm: str = "gn",
        dropout: float = 0.1,
        point_refine_dim: int = 96,
        fg_prior: float = 0.02,
        center_prior: float = 0.02,
        enable_deep_stage: int = 0,
        context_stage_max_voxels: int = 280000,
        deep_stage_max_voxels: int = 320000,
    ):
        super().__init__()
        C = int(init_channels)
        self.point_in_dim = int(point_in_dim)
        self.use_deep_stage = bool(int(enable_deep_stage))
        self.context_stage_max_voxels = int(context_stage_max_voxels)
        self.deep_stage_max_voxels = int(deep_stage_max_voxels)

        def subm(in_c, out_c, key):
            return spconv.SubMConv3d(in_c, out_c, 3, padding=1, bias=False, indice_key=key)

        def spc(in_c, out_c, key, stride=2):
            return spconv.SparseConv3d(in_c, out_c, 3, stride=stride, padding=1, bias=False, indice_key=key)

        def inv(in_c, out_c, key):
            return spconv.SparseInverseConv3d(in_c, out_c, 3, bias=False, indice_key=key)

        self.backbone = nn.ModuleDict(
            {
                "conv1": spconv.SparseSequential(
                    subm(in_channels, C, "subm1"),
                    make_norm(C, norm),
                    nn.ReLU(inplace=True),
                    subm(C, C, "subm1"),
                    make_norm(C, norm),
                    nn.ReLU(inplace=True),
                ),
                "conv2": spconv.SparseSequential(
                    spc(C, C * 2, "spconv2", stride=2),
                    make_norm(C * 2, norm),
                    nn.ReLU(inplace=True),
                    subm(C * 2, C * 2, "subm2"),
                    make_norm(C * 2, norm),
                    nn.ReLU(inplace=True),
                ),
                "conv3": spconv.SparseSequential(
                    spc(C * 2, C * 4, "spconv3", stride=2),
                    make_norm(C * 4, norm),
                    nn.ReLU(inplace=True),
                    subm(C * 4, C * 4, "subm3"),
                    make_norm(C * 4, norm),
                    nn.ReLU(inplace=True),
                ),
                "up2": spconv.SparseSequential(
                    inv(C * 4, C * 2, "spconv3"),
                    make_norm(C * 2, norm),
                    nn.ReLU(inplace=True),
                ),
                "dec2": spconv.SparseSequential(
                    subm(C * 4, C * 2, "subm2_dec"),
                    make_norm(C * 2, norm),
                    nn.ReLU(inplace=True),
                ),
                "up1": spconv.SparseSequential(
                    inv(C * 2, C, "spconv2"),
                    make_norm(C, norm),
                    nn.ReLU(inplace=True),
                ),
                "dec1": spconv.SparseSequential(
                    subm(C * 2, C, "subm1_dec"),
                    make_norm(C, norm),
                    nn.ReLU(inplace=True),
                ),
            }
        )
        self.backbone_extra = nn.ModuleDict(
            {
                "res1": nn.Sequential(
                    SparseResBlock(C, norm=norm, indice_key="subm1", dropout=dropout * 0.5),
                    SparseResBlock(C, norm=norm, indice_key="subm1", dropout=dropout * 0.5),
                ),
                "res2": nn.Sequential(
                    SparseResBlock(C * 2, norm=norm, indice_key="subm2", dropout=dropout * 0.5),
                    SparseResBlock(C * 2, norm=norm, indice_key="subm2", dropout=dropout * 0.5),
                ),
                "res3": nn.Sequential(
                    SparseResBlock(C * 4, norm=norm, indice_key="subm3", dropout=dropout * 0.5),
                    SparseResBlock(C * 4, norm=norm, indice_key="subm3", dropout=dropout * 0.5),
                ),
                "ctx2": nn.Sequential(
                    SparseResBlock(C * 2, norm=norm, indice_key="subm2_ctx_d2", dropout=dropout * 0.5, dilation=2),
                ),
                "ctx3": nn.Sequential(
                    SparseResBlock(C * 4, norm=norm, indice_key="subm3_ctx_d2", dropout=dropout * 0.5, dilation=2),
                    SparseResBlock(C * 4, norm=norm, indice_key="subm3_ctx_d3", dropout=dropout * 0.5, dilation=3),
                ),
                "conv4": spconv.SparseSequential(
                    spc(C * 4, C * 8, "spconv4", stride=2),
                    make_norm(C * 8, norm),
                    nn.ReLU(inplace=True),
                ),
                "res4": nn.Sequential(
                    SparseResBlock(C * 8, norm=norm, indice_key="subm4", dropout=dropout * 0.5),
                    SparseResBlock(C * 8, norm=norm, indice_key="subm4", dropout=dropout * 0.5),
                ),
                "up3": spconv.SparseSequential(
                    inv(C * 8, C * 4, "spconv4"),
                    make_norm(C * 4, norm),
                    nn.ReLU(inplace=True),
                ),
                "dec3": spconv.SparseSequential(
                    subm(C * 8, C * 4, "subm3_dec"),
                    make_norm(C * 4, norm),
                    nn.ReLU(inplace=True),
                ),
                "resd3": nn.Sequential(
                    SparseResBlock(C * 4, norm=norm, indice_key="subm3_dec", dropout=dropout * 0.5),
                ),
                "resd2": nn.Sequential(
                    SparseResBlock(C * 2, norm=norm, indice_key="subm2_dec", dropout=dropout * 0.5),
                ),
                "resd1": nn.Sequential(
                    SparseResBlock(C, norm=norm, indice_key="subm1_dec", dropout=dropout * 0.5),
                ),
            }
        )

        self.voxel_mlp = nn.Sequential(
            nn.Linear(C, C),
            nn.LayerNorm(C),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)) if dropout > 0 else nn.Identity(),
            nn.Linear(C, C),
            nn.LayerNorm(C),
            nn.ReLU(inplace=True),
        )

        P = int(point_refine_dim)
        self.point_stem = nn.Sequential(
            nn.Linear(self.point_in_dim, P),
            nn.LayerNorm(P),
            nn.GELU(),
            nn.Dropout(p=float(dropout)) if dropout > 0 else nn.Identity(),
            nn.Linear(P, P),
            nn.LayerNorm(P),
            nn.GELU(),
        )
        self.scale_projs = nn.ModuleDict(
            {
                "x1": nn.Sequential(nn.Linear(C, P), nn.LayerNorm(P), nn.GELU()),
                "x2": nn.Sequential(nn.Linear(C * 2, P), nn.LayerNorm(P), nn.GELU()),
                "x3": nn.Sequential(nn.Linear(C * 4, P), nn.LayerNorm(P), nn.GELU()),
                "out": nn.Sequential(nn.Linear(C, P), nn.LayerNorm(P), nn.GELU()),
            }
        )
        self.scale_fuse = nn.Sequential(
            nn.Linear(P * 5, P * 2),
            nn.LayerNorm(P * 2),
            nn.GELU(),
            nn.Dropout(p=float(dropout)) if dropout > 0 else nn.Identity(),
            nn.Linear(P * 2, P),
            nn.LayerNorm(P),
            nn.GELU(),
        )
        self.scale_gate = nn.Sequential(
            nn.Linear(P * 5, P),
            nn.LayerNorm(P),
            nn.GELU(),
            nn.Linear(P, P),
            nn.Sigmoid(),
        )
        self.point_refine = nn.Sequential(
            nn.Linear(P, P),
            nn.LayerNorm(P),
            nn.GELU(),
            nn.Dropout(p=float(dropout)) if dropout > 0 else nn.Identity(),
            nn.Linear(P, P),
            nn.LayerNorm(P),
            nn.GELU(),
        )

        ctx_dim = P * 4 + self.point_in_dim
        self.context_fuse = nn.Sequential(
            nn.Linear(ctx_dim, P * 2),
            nn.LayerNorm(P * 2),
            nn.GELU(),
            nn.Dropout(p=float(dropout)) if dropout > 0 else nn.Identity(),
            nn.Linear(P * 2, P),
            nn.LayerNorm(P),
            nn.GELU(),
        )
        self.context_gate = nn.Sequential(
            nn.Linear(ctx_dim, P),
            nn.LayerNorm(P),
            nn.GELU(),
            nn.Linear(P, P),
            nn.Sigmoid(),
        )
        self.object_trunk = nn.Sequential(
            nn.Linear(P, P),
            nn.LayerNorm(P),
            nn.GELU(),
            nn.Dropout(p=float(dropout)) if dropout > 0 else nn.Identity(),
            nn.Linear(P, P),
            nn.LayerNorm(P),
            nn.GELU(),
        )

        head_hidden = P

        def make_head(out_dim: int):
            return nn.Sequential(
                nn.Linear(P, head_hidden),
                nn.LayerNorm(head_hidden),
                nn.GELU(),
                nn.Linear(head_hidden, out_dim),
            )

        self.fg_head = nn.Linear(P, 1)
        self.center_head = make_head(1)
        self.offset_head = make_head(3)
        self.size_head = make_head(3)

        nn.init.constant_(self.scale_fuse[4].weight, 0.0)
        nn.init.constant_(self.scale_fuse[4].bias, 0.0)
        nn.init.constant_(self.point_refine[4].weight, 0.0)
        nn.init.constant_(self.point_refine[4].bias, 0.0)
        nn.init.constant_(self.context_fuse[4].weight, 0.0)
        nn.init.constant_(self.context_fuse[4].bias, 0.0)
        nn.init.constant_(self.object_trunk[4].weight, 0.0)
        nn.init.constant_(self.object_trunk[4].bias, 0.0)
        nn.init.normal_(self.fg_head.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.fg_head.bias, _bias_from_prior(fg_prior))
        nn.init.normal_(self.center_head[-1].weight, mean=0.0, std=0.01)
        nn.init.constant_(self.center_head[-1].bias, _bias_from_prior(center_prior))
        nn.init.normal_(self.offset_head[-1].weight, mean=0.0, std=0.01)
        nn.init.constant_(self.offset_head[-1].bias, 0.0)
        nn.init.normal_(self.size_head[-1].weight, mean=0.0, std=0.01)
        nn.init.constant_(self.size_head[-1].bias, 0.0)

    def forward(self, coords_b, feats_b, batch_size, spatial_shape, point2voxel, point_inputs, point_batch_ids=None):
        x = spconv.SparseConvTensor(feats_b, coords_b, spatial_shape=spatial_shape, batch_size=batch_size)

        x1 = self.backbone["conv1"](x)
        x1 = self.backbone_extra["res1"](x1)
        x2 = self.backbone["conv2"](x1)
        x2 = self.backbone_extra["res2"](x2)
        x2 = self.backbone_extra["ctx2"](x2)
        x3 = self.backbone["conv3"](x2)
        x3 = self.backbone_extra["res3"](x3)
        use_ctx3 = (self.context_stage_max_voxels <= 0) or (int(x3.features.shape[0]) <= self.context_stage_max_voxels)
        if use_ctx3:
            x3 = self.backbone_extra["ctx3"](x3)

        use_deep_stage = self.use_deep_stage and ((self.deep_stage_max_voxels <= 0) or (int(x3.features.shape[0]) <= self.deep_stage_max_voxels))
        if use_deep_stage:
            x4 = self.backbone_extra["conv4"](x3)
            x4 = self.backbone_extra["res4"](x4)

            up3 = self.backbone_extra["up3"](x4)
            cat3 = up3.replace_feature(torch.cat([up3.features, x3.features], dim=1))
            d3 = self.backbone_extra["dec3"](cat3)
            d3 = self.backbone_extra["resd3"](d3)
            up2_in = d3
        else:
            up2_in = x3

        up2 = self.backbone["up2"](up2_in)
        cat2 = up2.replace_feature(torch.cat([up2.features, x2.features], dim=1))
        d2 = self.backbone["dec2"](cat2)
        d2 = self.backbone_extra["resd2"](d2)

        up1 = self.backbone["up1"](d2)
        cat1 = up1.replace_feature(torch.cat([up1.features, x1.features], dim=1))
        out = self.backbone["dec1"](cat1)
        out = self.backbone_extra["resd1"](out)

        voxel_feat = self.voxel_mlp(out.features)
        point_base_coords = coords_b[point2voxel].to(torch.int64)
        point_seed = self.point_stem(point_inputs)
        point_x1 = self.scale_projs["x1"](gather_sparse_to_points(x1, point_base_coords, stride_div=1))
        point_x2 = self.scale_projs["x2"](gather_sparse_to_points(x2, point_base_coords, stride_div=2))
        point_x3 = self.scale_projs["x3"](gather_sparse_to_points(x3, point_base_coords, stride_div=4))
        point_out = self.scale_projs["out"](voxel_feat[point2voxel])

        local_ctx = torch.cat([point_seed, point_x1, point_x2, point_x3, point_out], dim=1)
        local_delta = self.scale_fuse(local_ctx)
        local_gate = self.scale_gate(local_ctx)
        point_local = point_seed + point_out + local_gate * local_delta
        point_local = point_local + self.point_refine(point_local)

        if point_batch_ids is None:
            point_batch_ids = torch.zeros((point_local.shape[0],), dtype=torch.long, device=point_local.device)

        scene_mean, scene_max, scene_std = scatter_scene_feature_stats(point_local, point_batch_ids, batch_size)
        ctx = torch.cat(
            [
                point_local,
                scene_mean[point_batch_ids],
                scene_max[point_batch_ids],
                scene_std[point_batch_ids],
                point_inputs,
            ],
            dim=1,
        )
        ctx_delta = self.context_fuse(ctx)
        ctx_gate = self.context_gate(ctx)
        fused = point_local + ctx_gate * ctx_delta
        fused = fused + self.object_trunk(fused)

        fg_logits = self.fg_head(fused).squeeze(1)
        center_logits = self.center_head(fused).squeeze(1)
        offset_pred = self.offset_head(fused)
        size_pred = self.size_head(fused)
        return {
            "fg_logits": fg_logits,
            "center_logits": center_logits,
            "offset_pred": offset_pred,
            "size_pred": size_pred,
        }


def calculate_spatial_shape(coords_b, padding=20):
    xyz = coords_b[:, 1:]
    maxc = xyz.max(dim=0)[0]
    return (maxc + padding).tolist()


def focal_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.75,
    gamma: float = 2.0,
    reduction: str = "mean",
):
    targets = targets.to(dtype=logits.dtype)
    prob = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = prob * targets + (1.0 - prob) * (1.0 - targets)
    alpha_factor = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    modulating = (1.0 - p_t).clamp(0.0, 1.0) ** gamma
    loss = alpha_factor * modulating * ce
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def hard_negative_mined_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.75,
    gamma: float = 2.0,
    neg_pos_ratio: float = 3.0,
    min_neg: int = 1024,
    max_neg: int = 16384,
):
    per_loss = focal_loss_with_logits(logits, targets, alpha=alpha, gamma=gamma, reduction="none")
    pos_mask = targets > 0.5
    neg_mask = ~pos_mask

    pos_loss = per_loss[pos_mask]
    neg_loss = per_loss[neg_mask]

    n_pos = int(pos_mask.sum().item())
    base = max(n_pos, 1)
    k = max(int(min_neg), int(round(float(neg_pos_ratio) * base)))
    if int(max_neg) > 0:
        k = min(k, int(max_neg))
    k = min(k, int(neg_loss.numel()))
    if k > 0:
        neg_keep = torch.topk(neg_loss, k=k, largest=True, sorted=False).values
    else:
        neg_keep = neg_loss[:0]

    denom = max(int(pos_loss.numel() + neg_keep.numel()), 1)
    return (pos_loss.sum() + neg_keep.sum()) / float(denom)


def tversky_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.3,
    beta: float = 0.7,
    smooth: float = 1.0,
):
    prob = torch.sigmoid(logits)
    targets = targets.to(dtype=prob.dtype)
    tp = (prob * targets).sum()
    fp = (prob * (1.0 - targets)).sum()
    fn = ((1.0 - prob) * targets).sum()
    tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return 1.0 - tversky


class FGLoss(nn.Module):
    def __init__(
        self,
        w_focal=1.0,
        w_tversky=1.0,
        w_center=0.25,
        w_offset=0.70,
        w_size=0.25,
        focal_alpha=0.75,
        focal_gamma=2.0,
        hnm_ratio=3.0,
        hnm_min_neg=1024,
        hnm_max_neg=16384,
        tversky_alpha=0.3,
        tversky_beta=0.7,
        center_bg_weight=0.20,
        center_fg_weight=1.00,
        center_peak_weight=1.50,
        offset_weight_power=1.0,
    ):
        super().__init__()
        self.w_focal = float(w_focal)
        self.w_tversky = float(w_tversky)
        self.w_center = float(w_center)
        self.w_offset = float(w_offset)
        self.w_size = float(w_size)
        self.focal_alpha = float(focal_alpha)
        self.focal_gamma = float(focal_gamma)
        self.hnm_ratio = float(hnm_ratio)
        self.hnm_min_neg = int(hnm_min_neg)
        self.hnm_max_neg = int(hnm_max_neg)
        self.tversky_alpha = float(tversky_alpha)
        self.tversky_beta = float(tversky_beta)
        self.center_bg_weight = float(center_bg_weight)
        self.center_fg_weight = float(center_fg_weight)
        self.center_peak_weight = float(center_peak_weight)
        self.offset_weight_power = float(offset_weight_power)

    def forward(self, outputs, fg_target, center_target=None, offset_target=None, offset_weight=None, size_target=None, size_weight=None):
        outputs = unpack_model_outputs(outputs)
        fg_logits = outputs["fg_logits"]
        center_logits = outputs["center_logits"]
        offset_pred = outputs["offset_pred"]
        size_pred = outputs.get("size_pred")

        focal = hard_negative_mined_focal_loss(
            fg_logits,
            fg_target,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
            neg_pos_ratio=self.hnm_ratio,
            min_neg=self.hnm_min_neg,
            max_neg=self.hnm_max_neg,
        )
        tv = tversky_loss_with_logits(
            fg_logits,
            fg_target,
            alpha=self.tversky_alpha,
            beta=self.tversky_beta,
        )

        zero = fg_logits.sum() * 0.0
        center_loss = zero
        offset_loss = zero
        size_loss = zero

        if center_target is not None:
            center_target = center_target.to(dtype=center_logits.dtype)
            fg_mask = (fg_target > 0.5).to(dtype=center_logits.dtype)
            center_weights = torch.full_like(center_target, self.center_bg_weight)
            center_weights = center_weights + fg_mask * self.center_fg_weight
            center_weights = center_weights + center_target * self.center_peak_weight
            center_bce = F.binary_cross_entropy_with_logits(center_logits, center_target, reduction="none")
            center_loss = (center_bce * center_weights).sum() / center_weights.sum().clamp_min(1.0)

        if offset_target is not None:
            fg_mask = fg_target > 0.5
            if bool(fg_mask.any()):
                off_pred = offset_pred[fg_mask]
                off_tgt = offset_target[fg_mask].to(dtype=off_pred.dtype)
                off_w = offset_weight[fg_mask].to(dtype=off_pred.dtype) if offset_weight is not None else torch.ones_like(off_pred[:, 0])
                off_w = off_w.clamp_min(1e-3).pow(self.offset_weight_power).unsqueeze(1)
                off_l1 = F.smooth_l1_loss(off_pred, off_tgt, reduction="none")
                offset_loss = (off_l1 * off_w).sum() / (off_w.sum().clamp_min(1.0) * float(off_l1.shape[1]))

        if size_target is not None and size_pred is not None:
            fg_mask = fg_target > 0.5
            if bool(fg_mask.any()):
                size_est = size_pred[fg_mask]
                size_tgt = size_target[fg_mask].to(dtype=size_est.dtype)
                size_w = size_weight[fg_mask].to(dtype=size_est.dtype) if size_weight is not None else torch.ones_like(size_est[:, 0])
                size_w = size_w.clamp_min(1e-3).unsqueeze(1)
                size_l1 = F.smooth_l1_loss(size_est, size_tgt, reduction="none")
                size_loss = (size_l1 * size_w).sum() / (size_w.sum().clamp_min(1.0) * float(size_l1.shape[1]))

        total = self.w_focal * focal + self.w_tversky * tv + self.w_center * center_loss + self.w_offset * offset_loss + self.w_size * size_loss
        stats = {
            "focal": focal.detach(),
            "tversky": tv.detach(),
            "center": center_loss.detach(),
            "offset": offset_loss.detach(),
            "size": size_loss.detach(),
        }
        return total, stats


@torch.no_grad()
def binary_metrics_from_prob(prob: np.ndarray, target01: np.ndarray, thr: float = 0.5):
    pred = (prob >= thr).astype(np.int64)
    tgt = (target01 > 0.5).astype(np.int64)

    tp = int(np.sum((pred == 1) & (tgt == 1)))
    fp = int(np.sum((pred == 1) & (tgt == 0)))
    fn = int(np.sum((pred == 0) & (tgt == 1)))
    tn = int(np.sum((pred == 0) & (tgt == 0)))

    prec = tp / (tp + fp + 1e-6)
    rec = tp / (tp + fn + 1e-6)
    iou = tp / (tp + fp + fn + 1e-6)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-6)
    return {"prec": prec, "rec": rec, "iou": iou, "acc": acc, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def empty_eval_counts():
    return {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "tn": 0,
        "gt_hit": 0,
        "gt_total": 0,
        "pred_hit": 0,
        "pred_total": 0,
        "scene_hit": 0,
        "scene_total": 0,
        "bbox_match_total": 0,
        "bbox_good": 0,
        "bbox_rel_mean_sum": 0.0,
        "bbox_rel_max_sum": 0.0,
        "bbox_center_dist_sum": 0.0,
        "bbox_iou_sum": 0.0,
        "gt_cover_sum": 0.0,
        "gt_cover_good": 0,
        "pred_purity_sum": 0.0,
    }


def add_eval_counts(dst: dict, src: dict):
    for k, v in src.items():
        dst[k] = dst.get(k, 0) + v
    return dst


def summarize_eval_counts(counts: dict):
    tp = float(counts.get("tp", 0))
    fp = float(counts.get("fp", 0))
    fn = float(counts.get("fn", 0))
    tn = float(counts.get("tn", 0))

    fg_prec = tp / (tp + fp + 1e-6)
    fg_rec = tp / (tp + fn + 1e-6)
    fg_iou = tp / (tp + fp + fn + 1e-6)
    fg_f1 = (2.0 * fg_prec * fg_rec) / (fg_prec + fg_rec + 1e-6)
    fg_acc = (tp + tn) / (tp + tn + fp + fn + 1e-6)

    obj_rec = float(counts.get("gt_hit", 0)) / max(float(counts.get("gt_total", 0)), 1.0)
    obj_prec = float(counts.get("pred_hit", 0)) / max(float(counts.get("pred_total", 0)), 1.0)
    obj_f1 = (2.0 * obj_prec * obj_rec) / (obj_prec + obj_rec + 1e-6)
    scene_rec = float(counts.get("scene_hit", 0)) / max(float(counts.get("scene_total", 0)), 1.0)
    obj_cover_mean = float(counts.get("gt_cover_sum", 0.0)) / max(float(counts.get("gt_total", 0)), 1.0)
    obj_cover_good_ratio = float(counts.get("gt_cover_good", 0)) / max(float(counts.get("gt_total", 0)), 1.0)
    pred_purity_mean = float(counts.get("pred_purity_sum", 0.0)) / max(float(counts.get("pred_total", 0)), 1.0)

    bbox_match_total = float(counts.get("bbox_match_total", 0))
    bbox_match_cov = bbox_match_total / max(float(counts.get("gt_total", 0)), 1.0)
    if bbox_match_total > 0:
        bbox_good_ratio = float(counts.get("bbox_good", 0)) / bbox_match_total
        bbox_rel_mean = float(counts.get("bbox_rel_mean_sum", 0.0)) / bbox_match_total
        bbox_rel_max = float(counts.get("bbox_rel_max_sum", 0.0)) / bbox_match_total
        bbox_center_dist = float(counts.get("bbox_center_dist_sum", 0.0)) / bbox_match_total
        bbox_iou = float(counts.get("bbox_iou_sum", 0.0)) / bbox_match_total
        bbox_size_score = float(np.exp(-1.25 * bbox_rel_mean) * np.exp(-0.45 * bbox_rel_max))
        bbox_center_score = float(np.exp(-bbox_center_dist / 4.0))
        bbox_shape_score = 0.40 * bbox_size_score + 0.20 * bbox_center_score + 0.25 * bbox_iou + 0.15 * bbox_good_ratio
        bbox_score = float(bbox_match_cov * bbox_shape_score)
    else:
        bbox_good_ratio = 0.0
        bbox_rel_mean = 0.0
        bbox_rel_max = 0.0
        bbox_center_dist = 0.0
        bbox_iou = 0.0
        bbox_score = 0.0

    # Prefer models that find complete targets, not just tiny fragments on a background-heavy scene.
    score = (
        0.24 * obj_f1
        + 0.18 * obj_cover_mean
        + 0.10 * obj_cover_good_ratio
        + 0.12 * obj_rec
        + 0.08 * obj_prec
        + 0.10 * pred_purity_mean
        + 0.12 * bbox_score
        + 0.04 * fg_f1
        + 0.02 * fg_iou
    )
    return {
        "score": float(score),
        "fg_prec": float(fg_prec),
        "fg_rec": float(fg_rec),
        "fg_f1": float(fg_f1),
        "fg_iou": float(fg_iou),
        "fg_acc": float(fg_acc),
        "obj_rec": float(obj_rec),
        "obj_prec": float(obj_prec),
        "obj_f1": float(obj_f1),
        "obj_cover_mean": float(obj_cover_mean),
        "obj_cover_good_ratio": float(obj_cover_good_ratio),
        "pred_purity_mean": float(pred_purity_mean),
        "scene_rec": float(scene_rec),
        "bbox_match_cov": float(bbox_match_cov),
        "bbox_good_ratio": float(bbox_good_ratio),
        "bbox_rel_mean": float(bbox_rel_mean),
        "bbox_rel_max": float(bbox_rel_max),
        "bbox_center_dist": float(bbox_center_dist),
        "bbox_iou": float(bbox_iou),
        "bbox_score": float(bbox_score),
    }


def bounds_iou_3d(a_min: np.ndarray, a_max: np.ndarray, b_min: np.ndarray, b_max: np.ndarray):
    inter = np.maximum(0.0, np.minimum(a_max, b_max) - np.maximum(a_min, b_min))
    inter_vol = float(np.prod(inter))
    va = float(np.prod(np.maximum(a_max - a_min, 0.0)))
    vb = float(np.prod(np.maximum(b_max - b_min, 0.0)))
    return inter_vol / max(va + vb - inter_vol, 1e-6)


def instance_box_summary(points_xyz: np.ndarray):
    pts = np.asarray(points_xyz, dtype=np.float32)
    if pts.shape[0] == 0:
        return {
            "center": np.zeros((3,), dtype=np.float32),
            "dims": np.zeros((3,), dtype=np.float32),
            "min_xyz": np.zeros((3,), dtype=np.float32),
            "max_xyz": np.zeros((3,), dtype=np.float32),
        }
    obox = oriented_box_stats_xyz(pts)
    return {
        "center": obox["center"].astype(np.float32),
        "dims": obox["dims"].astype(np.float32),
        "min_xyz": obox["min_xyz"].astype(np.float32),
        "max_xyz": obox["max_xyz"].astype(np.float32),
    }


def evaluate_pred_vs_gt(
    pred_inst: np.ndarray,
    gt_inst: np.ndarray,
    xyz_raw: np.ndarray = None,
    obj_hit_ratio: float = 0.15,
    min_hit_points: int = 24,
):
    pred_inst = np.asarray(pred_inst, dtype=np.int64)
    gt_inst = np.asarray(gt_inst, dtype=np.int64)
    pred_fg = pred_inst > 0
    gt_fg = gt_inst > 0

    counts = {
        "tp": int(np.sum(pred_fg & gt_fg)),
        "fp": int(np.sum(pred_fg & (~gt_fg))),
        "fn": int(np.sum((~pred_fg) & gt_fg)),
        "tn": int(np.sum((~pred_fg) & (~gt_fg))),
        "gt_hit": 0,
        "gt_total": 0,
        "pred_hit": 0,
        "pred_total": 0,
        "scene_hit": int(np.any(pred_fg & gt_fg)),
        "scene_total": 1,
        "bbox_match_total": 0,
        "bbox_good": 0,
        "bbox_rel_mean_sum": 0.0,
        "bbox_rel_max_sum": 0.0,
        "bbox_center_dist_sum": 0.0,
        "bbox_iou_sum": 0.0,
        "gt_cover_sum": 0.0,
        "gt_cover_good": 0,
        "pred_purity_sum": 0.0,
    }

    gt_ids = np.unique(gt_inst[gt_inst > 0]).astype(np.int64)
    pred_ids = np.unique(pred_inst[pred_inst > 0]).astype(np.int64)

    counts["gt_total"] = int(gt_ids.size)
    counts["pred_total"] = int(pred_ids.size)

    gt_masks = {int(gid): (gt_inst == int(gid)) for gid in gt_ids.tolist()}
    pred_masks = {int(pid): (pred_inst == int(pid)) for pid in pred_ids.tolist()}

    for gid in gt_ids:
        gmask = gt_masks[int(gid)]
        overlap = int(np.sum(pred_fg[gmask]))
        gt_cover = overlap / max(int(gmask.sum()), 1)
        counts["gt_cover_sum"] += float(gt_cover)
        if gt_cover >= 0.60:
            counts["gt_cover_good"] += 1
        need = int(max(min_hit_points, math.ceil(float(gmask.sum()) * float(obj_hit_ratio))))
        need = min(need, int(gmask.sum()))
        if overlap >= max(1, need):
            counts["gt_hit"] += 1

    for pid in pred_ids:
        pmask = pred_masks[int(pid)]
        overlap_best = 0
        for gid in gt_ids:
            overlap_best = max(overlap_best, int(np.sum(pmask & gt_masks[int(gid)])))
        counts["pred_purity_sum"] += float(overlap_best / max(int(pmask.sum()), 1))
        if pmask.sum() > 0:
            need = int(max(min_hit_points, math.ceil(float(pmask.sum()) * 0.08)))
            need = min(need, int(pmask.sum()))
            if overlap_best >= max(1, need):
                counts["pred_hit"] += 1

    pair_candidates = []
    for gid in gt_ids.tolist():
        gmask = gt_masks[int(gid)]
        for pid in pred_ids.tolist():
            pmask = pred_masks[int(pid)]
            overlap = int(np.sum(gmask & pmask))
            if overlap <= 0:
                continue
            union = int(np.sum(gmask | pmask))
            iou = overlap / max(union, 1)
            pair_candidates.append((iou, overlap, int(gid), int(pid)))

    pair_candidates.sort(reverse=True)
    used_gt = set()
    used_pred = set()
    for iou, overlap, gid, pid in pair_candidates:
        if gid in used_gt or pid in used_pred:
            continue
        used_gt.add(gid)
        used_pred.add(pid)

        if xyz_raw is not None:
            xyz_arr = np.asarray(xyz_raw, dtype=np.float32)
            gt_box = instance_box_summary(xyz_arr[gt_masks[gid]])
            pred_box = instance_box_summary(xyz_arr[pred_masks[pid]])
        else:
            gt_box = instance_box_summary(np.stack(np.where(gt_masks[gid]), axis=1).astype(np.float32))
            pred_box = instance_box_summary(np.stack(np.where(pred_masks[pid]), axis=1).astype(np.float32))

        dims_gt = np.maximum(np.sort(gt_box["dims"]), 1e-6)
        dims_pred = np.maximum(np.sort(pred_box["dims"]), 1e-6)
        rel = np.abs(dims_pred - dims_gt) / dims_gt
        center_dist = float(np.linalg.norm(pred_box["center"] - gt_box["center"]))
        bbox_iou = bounds_iou_3d(gt_box["min_xyz"], gt_box["max_xyz"], pred_box["min_xyz"], pred_box["max_xyz"])

        counts["bbox_match_total"] += 1
        counts["bbox_rel_mean_sum"] += float(np.mean(rel))
        counts["bbox_rel_max_sum"] += float(np.max(rel))
        counts["bbox_center_dist_sum"] += float(center_dist)
        counts["bbox_iou_sum"] += float(bbox_iou)
        if float(np.max(rel)) <= 0.45 and center_dist <= 8.0:
            counts["bbox_good"] += 1

    return counts


def dbscan_bfs(feat: np.ndarray, eps: float, min_samples: int):
    M = int(feat.shape[0])
    if M == 0:
        return np.zeros((0,), dtype=np.int32)

    tree = cKDTree(feat)
    labels = -np.ones((M,), dtype=np.int32)
    cluster_id = 0

    for i in range(M):
        if labels[i] != -1:
            continue

        neigh = tree.query_ball_point(feat[i], r=eps)
        if len(neigh) < int(min_samples):
            labels[i] = -2
            continue

        labels[i] = cluster_id
        queue = list(neigh)
        for j in queue:
            labels[j] = cluster_id

        while queue:
            j = queue.pop()
            neigh_j = tree.query_ball_point(feat[j], r=eps)
            if len(neigh_j) >= int(min_samples):
                for nb in neigh_j:
                    if labels[nb] in (-1, -2):
                        labels[nb] = cluster_id
                        queue.append(nb)
        cluster_id += 1

    labels = np.where(labels == -2, -1, labels)
    return labels


def build_postproc_priors(meta):
    if not isinstance(meta, dict) or len(meta) == 0:
        return {
            "size_x_lo": 2.5,
            "size_x_hi": 8.0,
            "size_y_lo": 2.0,
            "size_y_hi": 8.0,
            "size_z_lo": 2.5,
            "size_z_hi": 8.5,
            "pts_lo": 300,
            "pts_hi": 10000,
            "final_min_points": 80,
            "merge_small_cluster_points": 360,
            "merge_xy_gap": 1.6,
            "merge_z_gap": 2.8,
            "merge_xy_overlap": 0.25,
            "support_radius": 0.90,
            "keep_topk": 3,
            "expected_max_instances": 2,
            "fg_points_lo": 120,
            "fg_points_hi": 5000,
            "fg_ratio_lo": 0.003,
            "fg_ratio_hi": 0.050,
            "fg_ratio_ref": 0.010,
            "fg_points_ref": 1200.0,
            "binary_cluster_eps": 0.42,
            "binary_cluster_min_samples": 3,
            "binary_min_component_points": 48,
            "binary_max_component_points": 12000,
        }

    def _get(k, d):
        return float(meta.get(k, d))

    size_x_p10 = _get("size_x_p10", 4.0)
    size_x_p50 = _get("size_x_p50", 4.5)
    size_x_p90 = _get("size_x_p90", 5.5)
    size_y_p10 = _get("size_y_p10", 3.5)
    size_y_p50 = _get("size_y_p50", 4.0)
    size_y_p90 = _get("size_y_p90", 4.8)
    size_z_p10 = _get("size_z_p10", 4.5)
    size_z_p50 = _get("size_z_p50", 5.0)
    size_z_p90 = _get("size_z_p90", 6.2)
    pts_p10 = _get("num_points_p10", 1200.0)
    pts_p50 = _get("num_points_p50", 1700.0)
    pts_p90 = _get("num_points_p90", 2600.0)
    fg_points_p10 = _get("fg_points_p10", 800.0)
    fg_points_p50 = _get("fg_points_p50", 1400.0)
    fg_points_p90 = _get("fg_points_p90", 2400.0)
    fg_ratio_p10 = _get("fg_ratio_p10", 0.006)
    fg_ratio_p50 = _get("fg_ratio_p50", 0.010)
    fg_ratio_p90 = _get("fg_ratio_p90", 0.018)

    xy_short_p10 = min(size_x_p10, size_y_p10)
    xy_short_p50 = min(size_x_p50, size_y_p50)
    xy_short_p90 = min(size_x_p90, size_y_p90)
    xy_long_p10 = max(size_x_p10, size_y_p10)
    xy_long_p50 = max(size_x_p50, size_y_p50)
    xy_long_p90 = max(size_x_p90, size_y_p90)

    pri = {
        "size_x_lo": max(1.5, 0.55 * size_x_p10),
        "size_x_hi": 1.70 * size_x_p90,
        "size_y_lo": max(1.5, 0.55 * size_y_p10),
        "size_y_hi": 1.70 * size_y_p90,
        "xy_short_lo": max(1.5, 0.55 * xy_short_p10),
        "xy_short_hi": 1.70 * xy_short_p90,
        "xy_long_lo": max(1.8, 0.55 * xy_long_p10),
        "xy_long_hi": 1.70 * xy_long_p90,
        "size_z_lo": max(2.0, 0.60 * size_z_p10),
        "size_z_hi": 1.70 * size_z_p90,
        "pts_lo": max(120.0, 0.20 * pts_p10),
        "pts_hi": 3.50 * pts_p90,
        "final_min_points": max(50.0, 0.08 * pts_p10),
        "merge_small_cluster_points": max(180.0, 0.25 * pts_p50),
        "merge_xy_gap": max(1.2, 0.40 * xy_short_p50),
        "merge_z_gap": max(1.6, 0.55 * size_z_p50),
        "merge_xy_overlap": 0.22,
        "support_radius": max(0.75, 0.18 * xy_long_p50),
        "keep_topk": max(2, min(4, int(round(_get("inst_per_scene_p90", 2.0) + 1.0)))),
        "expected_max_instances": max(1, int(round(_get("inst_per_scene_p90", 2.0)))),
        "fg_points_lo": max(32.0, 0.18 * fg_points_p10),
        "fg_points_hi": max(256.0, 2.80 * fg_points_p90),
        "fg_ratio_lo": max(0.0015, 0.55 * fg_ratio_p10),
        "fg_ratio_hi": min(0.0600, max(0.0100, 2.40 * fg_ratio_p90)),
        "fg_ratio_ref": float(np.clip(fg_ratio_p50, 0.0015, 0.0500)),
        "fg_points_ref": max(32.0, fg_points_p50),
        "binary_cluster_eps": max(0.24, 0.12 * xy_long_p50),
        "binary_cluster_min_samples": 3,
        "binary_min_component_points": max(24.0, 0.035 * pts_p10),
        "binary_max_component_points": max(300.0, 4.50 * pts_p90),
    }
    return pri


def soft_range_score(v, lo, hi, margin_ratio=0.25):
    lo = float(lo)
    hi = float(hi)
    if lo <= v <= hi:
        return 1.0

    span = max(hi - lo, 1e-6)
    margin = max(span * margin_ratio, 1e-6)

    if v < lo:
        d = (lo - v) / margin
    else:
        d = (v - hi) / margin
    return float(np.exp(-(d * d)))


def bbox_gap_1d(a_min, a_max, b_min, b_max):
    if a_max < b_min:
        return float(b_min - a_max)
    if b_max < a_min:
        return float(a_min - b_max)
    return 0.0


def bbox_xy_overlap_ratio(a_min, a_max, b_min, b_max):
    inter_x = max(0.0, min(float(a_max[0]), float(b_max[0])) - max(float(a_min[0]), float(b_min[0])))
    inter_y = max(0.0, min(float(a_max[1]), float(b_max[1])) - max(float(a_min[1]), float(b_min[1])))
    inter = inter_x * inter_y
    area_a = max(float(a_max[0] - a_min[0]), 1e-6) * max(float(a_max[1] - a_min[1]), 1e-6)
    area_b = max(float(b_max[0] - b_min[0]), 1e-6) * max(float(b_max[1] - b_min[1]), 1e-6)
    return float(inter / (min(area_a, area_b) + 1e-6))


def make_cluster_from_indices(global_idx, xyz_raw, score_prob, seed_mask=None):
    gidx = np.asarray(global_idx, dtype=np.int64)
    cpts = xyz_raw[gidx]
    min_xyz = cpts.min(axis=0).astype(np.float32)
    max_xyz = cpts.max(axis=0).astype(np.float32)
    size_xyz = (max_xyz - min_xyz).astype(np.float32)
    center = cpts.mean(axis=0).astype(np.float32)
    obox = oriented_box_stats_xyz(cpts)
    d3_center = np.linalg.norm(cpts - center[None, :], axis=1)
    return {
        "global_idx": gidx,
        "num_points": int(gidx.size),
        "min_xyz": min_xyz,
        "max_xyz": max_xyz,
        "size_xyz": size_xyz,
        "center": center,
        "obox_dims": obox["dims"].astype(np.float32),
        "obox_center": obox["center"].astype(np.float32),
        "prob_mean": float(np.mean(score_prob[gidx])),
        "prob_max": float(np.max(score_prob[gidx])),
        "rad90": float(np.percentile(d3_center, 90)) if gidx.size > 0 else 0.0,
        "seed_count": int(seed_mask[gidx].sum()) if seed_mask is not None else 0,
    }


def build_hysteresis_clusters(
    xyz_raw: np.ndarray,
    fg_prob: np.ndarray,
    seed_prob: np.ndarray,
    score_prob: np.ndarray,
    cluster_xyz: np.ndarray,
    seed_threshold: float,
    grow_threshold: float,
    cluster_eps: float,
    grow_cluster_min_samples: int,
    seed_min_points: int,
):
    low_mask = fg_prob >= float(grow_threshold)
    if int(low_mask.sum()) == 0:
        return []

    seed_mask = seed_prob >= float(seed_threshold)
    low_idx = np.where(low_mask)[0]
    low_pts = cluster_xyz[low_mask].astype(np.float32)

    labels = dbscan_bfs(low_pts, eps=float(cluster_eps), min_samples=int(grow_cluster_min_samples))
    clusters = []

    if labels.size > 0 and labels.max() >= 0:
        for cid in range(int(labels.max()) + 1):
            local = np.where(labels == cid)[0]
            if local.size == 0:
                continue
            gidx = low_idx[local]
            seed_count = int(seed_mask[gidx].sum())
            if seed_count < int(seed_min_points):
                continue
            clusters.append(make_cluster_from_indices(gidx, xyz_raw, score_prob, seed_mask=seed_mask))

    if len(clusters) == 0:
        high_idx = np.where(seed_mask)[0]
        if high_idx.size > 0:
            clusters.append(make_cluster_from_indices(high_idx, xyz_raw, score_prob, seed_mask=seed_mask))

    return clusters


def should_merge_clusters(a, b, priors, merge_small_cluster_points, merge_xy_gap, merge_z_gap, merge_xy_overlap):
    if a["num_points"] >= b["num_points"]:
        big, small = a, b
    else:
        big, small = b, a

    xy_dist = float(np.linalg.norm(big["center"][:2] - small["center"][:2]))
    x_gap = bbox_gap_1d(big["min_xyz"][0], big["max_xyz"][0], small["min_xyz"][0], small["max_xyz"][0])
    y_gap = bbox_gap_1d(big["min_xyz"][1], big["max_xyz"][1], small["min_xyz"][1], small["max_xyz"][1])
    z_gap = bbox_gap_1d(big["min_xyz"][2], big["max_xyz"][2], small["min_xyz"][2], small["max_xyz"][2])
    xy_overlap = bbox_xy_overlap_ratio(big["min_xyz"], big["max_xyz"], small["min_xyz"], small["max_xyz"])

    small_like_fragment = (
        (small["num_points"] <= float(merge_small_cluster_points))
        or (small["size_xyz"][2] < 0.70 * float(priors["size_z_lo"]))
        or (
            (small["size_xyz"][0] < 0.80 * float(priors["size_x_lo"]))
            and (small["size_xyz"][1] < 0.80 * float(priors["size_y_lo"]))
        )
    )

    if xy_overlap >= float(merge_xy_overlap) and z_gap <= float(merge_z_gap):
        return True

    if small_like_fragment and xy_dist <= float(merge_xy_gap) and z_gap <= float(merge_z_gap) * 1.20:
        return True

    if small_like_fragment and x_gap <= float(merge_xy_gap) and y_gap <= float(merge_xy_gap) and z_gap <= float(merge_z_gap) * 1.30:
        return True

    return False


def merge_fragmented_clusters(
    clusters,
    xyz_raw: np.ndarray,
    score_prob: np.ndarray,
    priors: dict,
    merge_small_cluster_points: float,
    merge_xy_gap: float,
    merge_z_gap: float,
    merge_xy_overlap: float,
    max_rounds: int = 4,
):
    if len(clusters) <= 1:
        return clusters

    merged = list(clusters)
    for _ in range(int(max_rounds)):
        n = len(merged)
        parent = np.arange(n, dtype=np.int64)

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        any_merge = False
        for i in range(n):
            for j in range(i + 1, n):
                if should_merge_clusters(
                    merged[i],
                    merged[j],
                    priors=priors,
                    merge_small_cluster_points=merge_small_cluster_points,
                    merge_xy_gap=merge_xy_gap,
                    merge_z_gap=merge_z_gap,
                    merge_xy_overlap=merge_xy_overlap,
                ):
                    union(i, j)
                    any_merge = True

        if not any_merge:
            break

        groups = {}
        for i in range(n):
            r = find(i)
            groups.setdefault(r, []).append(i)

        new_clusters = []
        for idxs in groups.values():
            all_idx = np.concatenate([merged[k]["global_idx"] for k in idxs], axis=0)
            all_idx = np.unique(all_idx)
            new_clusters.append(make_cluster_from_indices(all_idx, xyz_raw, score_prob))

        merged = new_clusters

    return merged


def expand_clusters_by_support_points(
    clusters,
    xyz_raw: np.ndarray,
    support_prob: np.ndarray,
    score_prob: np.ndarray,
    support_threshold: float,
    support_radius: float,
):
    if len(clusters) == 0:
        return clusters

    pred = np.zeros((xyz_raw.shape[0],), dtype=np.int64)
    cluster_pts = []
    cluster_ids = []
    for cid, c in enumerate(clusters, start=1):
        pred[c["global_idx"]] = cid
        cluster_pts.append(xyz_raw[c["global_idx"]])
        cluster_ids.append(np.full((c["global_idx"].shape[0],), cid, dtype=np.int64))

    if len(cluster_pts) == 0:
        return clusters

    cluster_pts = np.concatenate(cluster_pts, axis=0).astype(np.float32)
    cluster_ids = np.concatenate(cluster_ids, axis=0).astype(np.int64)

    support_mask = (support_prob >= float(support_threshold)) & (pred == 0)
    support_idx = np.where(support_mask)[0]
    if support_idx.size == 0:
        return clusters

    tree = cKDTree(cluster_pts)
    dist, nn = tree.query(xyz_raw[support_idx].astype(np.float32), k=1)
    ok = dist <= float(support_radius)
    if np.any(ok):
        pred[support_idx[ok]] = cluster_ids[nn[ok]].astype(np.int64)

    out = []
    for cid in sorted(np.unique(pred[pred > 0]).tolist()):
        gidx = np.where(pred == cid)[0]
        out.append(make_cluster_from_indices(gidx, xyz_raw, score_prob))
    return out


def score_cluster_for_keep(c, priors):
    obox_dims = np.asarray(c.get("obox_dims", c["size_xyz"]), dtype=np.float32)
    xy_sorted = np.sort(obox_dims[:2])
    short_score = soft_range_score(float(xy_sorted[0]), priors.get("xy_short_lo", priors["size_y_lo"]), priors.get("xy_short_hi", priors["size_y_hi"]))
    long_score = soft_range_score(float(xy_sorted[1]), priors.get("xy_long_lo", priors["size_x_lo"]), priors.get("xy_long_hi", priors["size_x_hi"]))
    sz_score = soft_range_score(float(obox_dims[2]), priors["size_z_lo"], priors["size_z_hi"])
    pt_score = soft_range_score(c["num_points"], priors["pts_lo"], priors["pts_hi"])

    prior_score = (short_score * long_score * sz_score * pt_score) ** 0.25
    prob_score = 0.60 * float(c["prob_mean"]) + 0.40 * float(c["prob_max"])
    final_score = prob_score * prior_score

    axis_z = float(c["size_xyz"][2])
    if axis_z > max(float(obox_dims[2]) * 1.8, float(priors["size_z_hi"]) * 1.35):
        final_score *= 0.35
    if float(obox_dims[2]) > float(priors["size_z_hi"]) * 1.45:
        final_score *= 0.45

    if c["num_points"] < float(priors["final_min_points"]):
        final_score *= 0.20
    if float(obox_dims[2]) < 0.65 * float(priors["size_z_lo"]):
        final_score *= 0.55

    c = dict(c)
    c["prior_score"] = float(prior_score)
    c["final_score"] = float(final_score)
    return c


def cluster_points_to_instances(
    xyz_raw: np.ndarray,
    fg_prob: np.ndarray,
    seed_prob: np.ndarray,
    score_prob: np.ndarray,
    cluster_xyz: np.ndarray,
    seed_threshold: float,
    grow_threshold: float,
    support_threshold: float,
    cluster_eps: float,
    grow_cluster_min_samples: int,
    seed_min_points: int,
    support_radius: float,
    postproc_priors: dict,
    min_cluster_score: float = 0.12,
    keep_topk: int = 3,
    fallback_keep_best: bool = True,
    merge_small_cluster_points: float = 0.0,
    merge_xy_gap: float = 0.0,
    merge_z_gap: float = 0.0,
    merge_xy_overlap: float = 0.22,
):
    pred_inst = np.zeros((xyz_raw.shape[0],), dtype=np.int64)
    if xyz_raw.shape[0] == 0:
        return pred_inst, []

    priors = dict(postproc_priors)
    if merge_small_cluster_points <= 0:
        merge_small_cluster_points = float(priors.get("merge_small_cluster_points", 300.0))
    if merge_xy_gap <= 0:
        merge_xy_gap = float(priors.get("merge_xy_gap", 1.6))
    if merge_z_gap <= 0:
        merge_z_gap = float(priors.get("merge_z_gap", 2.8))
    if support_radius <= 0:
        support_radius = float(priors.get("support_radius", 0.90))
    if keep_topk <= 0:
        keep_topk = int(priors.get("keep_topk", 3))

    clusters = build_hysteresis_clusters(
        xyz_raw=xyz_raw,
        fg_prob=fg_prob,
        seed_prob=seed_prob,
        score_prob=score_prob,
        cluster_xyz=cluster_xyz,
        seed_threshold=seed_threshold,
        grow_threshold=grow_threshold,
        cluster_eps=cluster_eps,
        grow_cluster_min_samples=grow_cluster_min_samples,
        seed_min_points=seed_min_points,
    )
    if len(clusters) == 0:
        return pred_inst, []

    clusters = merge_fragmented_clusters(
        clusters,
        xyz_raw=xyz_raw,
        score_prob=score_prob,
        priors=priors,
        merge_small_cluster_points=merge_small_cluster_points,
        merge_xy_gap=merge_xy_gap,
        merge_z_gap=merge_z_gap,
        merge_xy_overlap=merge_xy_overlap,
        max_rounds=4,
    )

    clusters = expand_clusters_by_support_points(
        clusters,
        xyz_raw=xyz_raw,
        support_prob=fg_prob,
        score_prob=score_prob,
        support_threshold=support_threshold,
        support_radius=support_radius,
    )

    clusters = merge_fragmented_clusters(
        clusters,
        xyz_raw=xyz_raw,
        score_prob=score_prob,
        priors=priors,
        merge_small_cluster_points=merge_small_cluster_points,
        merge_xy_gap=merge_xy_gap,
        merge_z_gap=merge_z_gap,
        merge_xy_overlap=merge_xy_overlap,
        max_rounds=2,
    )

    scored = [score_cluster_for_keep(c, priors) for c in clusters]
    scored = sorted(scored, key=lambda x: x["final_score"], reverse=True)
    kept = [c for c in scored if c["final_score"] >= float(min_cluster_score)]

    if len(kept) == 0 and fallback_keep_best:
        kept = scored[:1]

    kept = kept[: int(max(1, keep_topk))]
    for i, c in enumerate(kept, start=1):
        pred_inst[c["global_idx"]] = i

    return pred_inst, kept


def adapt_fg_threshold(
    fg_prob: np.ndarray,
    base_thr: float,
    priors: dict,
    thr_min: float = 0.05,
    thr_max: float = 0.95,
    num_steps: int = 41,
):
    fg_prob = np.asarray(fg_prob, dtype=np.float32)
    if fg_prob.size == 0:
        return float(base_thr)

    base_thr = float(np.clip(base_thr, thr_min, thr_max))
    N = int(fg_prob.size)

    ratio_lo = float(priors.get("fg_ratio_lo", 0.003))
    ratio_hi = float(priors.get("fg_ratio_hi", 0.050))
    ratio_ref = float(np.clip(priors.get("fg_ratio_ref", 0.010), ratio_lo, ratio_hi))
    points_lo = int(max(1, round(priors.get("fg_points_lo", max(24.0, ratio_lo * N)))))
    points_hi = int(max(points_lo, round(priors.get("fg_points_hi", max(points_lo, ratio_hi * N)))))
    points_ref = float(np.clip(priors.get("fg_points_ref", max(32.0, ratio_ref * N)), points_lo, points_hi))

    base_count = int(np.sum(fg_prob >= base_thr))
    base_ratio = base_count / max(N, 1)
    if ratio_lo <= base_ratio <= ratio_hi and points_lo <= base_count <= points_hi:
        return float(base_thr)

    grid = np.linspace(thr_min, thr_max, int(max(5, num_steps))).astype(np.float32)
    counts = np.array([int(np.sum(fg_prob >= float(t))) for t in grid], dtype=np.int64)
    ratios = counts.astype(np.float32) / max(N, 1)

    score = np.abs(np.log((ratios + 1e-6) / (ratio_ref + 1e-6)))
    score = score + 0.45 * np.abs(np.log((counts.astype(np.float32) + 1.0) / (points_ref + 1.0)))
    score = score + 0.08 * np.abs(grid - base_thr)

    inside = (ratios >= ratio_lo) & (ratios <= ratio_hi) & (counts >= points_lo) & (counts <= points_hi)
    if np.any(inside):
        idx = int(np.argmin(np.where(inside, score, score + 1e6)))
    else:
        idx = int(np.argmin(score))
    return float(grid[idx])


def mask_to_connected_instances(
    xyz_raw: np.ndarray,
    fg_mask: np.ndarray,
    cluster_eps: float,
    cluster_min_samples: int,
    keep_topk: int = 0,
):
    fg_mask = np.asarray(fg_mask, dtype=bool)
    pred_inst = np.zeros((fg_mask.shape[0],), dtype=np.int64)
    idx = np.where(fg_mask)[0]
    if idx.size == 0:
        return pred_inst

    pts = xyz_raw[idx].astype(np.float32)
    labels = dbscan_bfs(pts, eps=float(cluster_eps), min_samples=max(1, int(cluster_min_samples)))
    valid = labels >= 0
    if not np.any(valid):
        pred_inst[idx] = 1
        return pred_inst

    order = []
    for cid in range(int(labels.max()) + 1):
        local = np.where(labels == cid)[0]
        if local.size > 0:
            order.append((local.size, cid))
    order.sort(reverse=True)
    if keep_topk > 0:
        order = order[: int(keep_topk)]

    for out_id, (_, cid) in enumerate(order, start=1):
        pred_inst[idx[labels == int(cid)]] = int(out_id)
    return pred_inst


def cleanup_binary_mask(
    xyz_raw: np.ndarray,
    fg_prob: np.ndarray,
    fg_mask: np.ndarray,
    cluster_eps: float,
    cluster_min_samples: int,
    min_component_points: int,
    max_component_points: int = 0,
    postproc_priors: dict = None,
    keep_topk: int = 1,
    fallback_keep_largest: bool = True,
):
    fg_mask = np.asarray(fg_mask, dtype=bool)
    keep_mask = np.zeros_like(fg_mask, dtype=bool)
    idx = np.where(fg_mask)[0]
    if idx.size == 0:
        return keep_mask

    priors = dict(postproc_priors or {})
    pts = xyz_raw[idx].astype(np.float32)
    labels = dbscan_bfs(pts, eps=float(cluster_eps), min_samples=max(1, int(cluster_min_samples)))
    best_score = -1e18
    best_comp = None
    scored_kept = []

    if labels.size > 0 and labels.max() >= 0:
        for cid in range(int(labels.max()) + 1):
            local = np.where(labels == cid)[0]
            if local.size == 0:
                continue
            gidx = idx[local]
            comp_n = int(gidx.size)
            cluster = score_cluster_for_keep(make_cluster_from_indices(gidx, xyz_raw, fg_prob), priors) if priors else None
            score = float(cluster["final_score"]) if cluster is not None else (float(np.mean(fg_prob[gidx])) + 0.0004 * comp_n)
            if score > best_score:
                best_score = score
                best_comp = gidx

            if comp_n < int(max(1, min_component_points)):
                continue
            if int(max_component_points) > 0 and comp_n > int(max_component_points):
                continue
            if cluster is not None:
                scored_kept.append(cluster)
            else:
                keep_mask[gidx] = True

    if len(scored_kept) > 0:
        scored_kept = sorted(scored_kept, key=lambda x: x["final_score"], reverse=True)
        limit = int(max(1, keep_topk))
        for cluster in scored_kept[:limit]:
            keep_mask[cluster["global_idx"]] = True
        return keep_mask
    if np.any(keep_mask):
        return keep_mask

    if fallback_keep_largest:
        if best_comp is not None:
            keep_mask[best_comp] = True
            return keep_mask
        keep_mask[idx] = True
    return keep_mask


def predict_scene_labels(
    xyz_raw: np.ndarray,
    fg_prob: np.ndarray,
    base_thr: float,
    postproc_priors: dict,
    seed_prob: np.ndarray = None,
    score_prob: np.ndarray = None,
    vote_xyz: np.ndarray = None,
    output_mode: str = "binary_cc",
    adaptive_threshold: bool = True,
    binary_cluster_eps: float = -1.0,
    binary_cluster_min_samples: int = -1,
    binary_min_component_points: int = -1,
    binary_max_component_points: int = -1,
    seed_threshold: float = -1.0,
    grow_threshold: float = -1.0,
    support_threshold: float = -1.0,
    cluster_eps: float = 0.60,
    grow_cluster_min_samples: int = 6,
    seed_min_points: int = 24,
    support_radius: float = -1.0,
    min_cluster_score: float = 0.12,
    keep_topk: int = 3,
    fallback_keep_best: bool = True,
    merge_small_cluster_points: float = -1.0,
    merge_xy_gap: float = -1.0,
    merge_z_gap: float = -1.0,
    merge_xy_overlap: float = 0.22,
):
    priors = dict(postproc_priors or {})
    output_mode = str(output_mode).lower()
    if output_mode == "auto":
        output_mode = "binary_cc"

    seed_prob = np.asarray(seed_prob if seed_prob is not None else fg_prob, dtype=np.float32)
    score_prob = np.asarray(score_prob if score_prob is not None else seed_prob, dtype=np.float32)
    cluster_xyz = np.asarray(vote_xyz if vote_xyz is not None else xyz_raw, dtype=np.float32)
    base_thr = float(np.clip(base_thr, 0.01, 0.99))
    scene_thr = adapt_fg_threshold(score_prob, base_thr, priors) if adaptive_threshold else base_thr

    if binary_cluster_eps <= 0:
        binary_cluster_eps = float(priors.get("binary_cluster_eps", max(0.28, 0.80 * float(cluster_eps))))
    if binary_cluster_min_samples <= 0:
        binary_cluster_min_samples = int(priors.get("binary_cluster_min_samples", 3))
    if binary_min_component_points <= 0:
        binary_min_component_points = int(round(priors.get("binary_min_component_points", 48.0)))
    if binary_max_component_points <= 0:
        binary_max_component_points = int(round(priors.get("binary_max_component_points", 0.0)))

    if output_mode == "instance":
        seed_thr = float(seed_threshold) if seed_threshold > 0 else max(scene_thr, base_thr, 0.48)
        grow_thr = float(grow_threshold) if grow_threshold > 0 else max(0.06, min(seed_thr - 0.22, 0.34))
        sup_thr = float(support_threshold) if support_threshold > 0 else max(0.04, grow_thr - 0.08)
        pred_inst, kept = cluster_points_to_instances(
            xyz_raw=xyz_raw,
            fg_prob=fg_prob,
            seed_prob=seed_prob,
            score_prob=score_prob,
            cluster_xyz=cluster_xyz,
            seed_threshold=seed_thr,
            grow_threshold=grow_thr,
            support_threshold=sup_thr,
            cluster_eps=cluster_eps,
            grow_cluster_min_samples=grow_cluster_min_samples,
            seed_min_points=seed_min_points,
            support_radius=support_radius,
            postproc_priors=priors,
            min_cluster_score=min_cluster_score,
            keep_topk=keep_topk,
            fallback_keep_best=fallback_keep_best,
            merge_small_cluster_points=merge_small_cluster_points,
            merge_xy_gap=merge_xy_gap,
            merge_z_gap=merge_z_gap,
            merge_xy_overlap=merge_xy_overlap,
        )
        if int(pred_inst.max()) > 0:
            return pred_inst, {
                "output_mode": output_mode,
                "scene_thr": float(scene_thr),
                "seed_thr": float(seed_thr),
                "grow_thr": float(grow_thr),
                "support_thr": float(sup_thr),
                "num_instances": int(pred_inst.max()),
                "num_fg_points": int((pred_inst > 0).sum()),
            }

    raw_mask = score_prob >= float(scene_thr)
    clean_mask = cleanup_binary_mask(
        xyz_raw=xyz_raw,
        fg_prob=score_prob,
        fg_mask=raw_mask,
        cluster_eps=float(binary_cluster_eps),
        cluster_min_samples=int(binary_cluster_min_samples),
        min_component_points=int(binary_min_component_points),
        max_component_points=int(binary_max_component_points),
        postproc_priors=priors,
        keep_topk=int(max(1, min(int(max(1, keep_topk)), int(max(1, priors.get("expected_max_instances", 1)))))),
        fallback_keep_largest=bool(fallback_keep_best),
    )

    if output_mode == "binary":
        pred_inst = clean_mask.astype(np.int64)
    else:
        pred_inst = mask_to_connected_instances(
            xyz_raw=xyz_raw,
            fg_mask=clean_mask,
            cluster_eps=float(binary_cluster_eps),
            cluster_min_samples=max(1, int(binary_cluster_min_samples)),
            keep_topk=int(max(0, keep_topk)),
        )

    return pred_inst, {
        "output_mode": output_mode,
        "scene_thr": float(scene_thr),
        "num_instances": int(pred_inst.max()) if output_mode != "binary" else int(np.any(pred_inst > 0)),
        "num_fg_points": int((pred_inst > 0).sum()),
        "binary_cluster_eps": float(binary_cluster_eps),
        "binary_min_component_points": int(binary_min_component_points),
    }


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_metric, best_thr, meta=None):
    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "best_metric": best_metric,
        "best_thr": best_thr,
        "meta": meta or {},
    }
    torch.save(ckpt, path)


def update_checkpoint_postproc(path, best_postproc: dict, meta_updates: dict = None, logger=None):
    if not os.path.exists(path):
        return
    ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict):
        return
    ckpt["best_postproc"] = dict(best_postproc or {})
    meta = ckpt.get("meta", {}) if isinstance(ckpt.get("meta", {}), dict) else {}
    if meta_updates:
        meta.update(meta_updates)
    if best_postproc:
        meta["best_postproc"] = dict(best_postproc)
    ckpt["meta"] = meta
    torch.save(ckpt, path)
    if logger is not None:
        logger.info(f"Updated checkpoint postproc: {path} -> {best_postproc}")


def load_state_dict_flexible(model, state_dict, logger=None):
    model_sd = model.state_dict()
    remapped = dict(state_dict)
    legacy_map = {
        "fg_head.weight": "fg_head.3.weight",
        "fg_head.bias": "fg_head.3.bias",
    }
    for src, dst in legacy_map.items():
        if src in remapped and dst in model_sd and hasattr(remapped[src], "shape") and remapped[src].shape == model_sd[dst].shape:
            remapped[dst] = remapped[src]

    filtered = {}
    skipped = []
    for k, v in remapped.items():
        if (k in model_sd) and (hasattr(v, "shape")) and (model_sd[k].shape == v.shape):
            filtered[k] = v
        else:
            skipped.append(k)

    incompat = model.load_state_dict(filtered, strict=False)
    if logger is not None:
        logger.info(f"State dict: loaded {len(filtered)}/{len(state_dict)} tensors.")
        if skipped:
            logger.warning(f"State dict: skipped {len(skipped)} keys due to mismatch/not found. Example: {skipped[:8]}")
        if getattr(incompat, "missing_keys", None):
            mk = incompat.missing_keys
            if mk:
                logger.warning(f"State dict: missing keys {len(mk)}. Example: {mk[:8]}")
        if getattr(incompat, "unexpected_keys", None):
            uk = incompat.unexpected_keys
            if uk:
                logger.warning(f"State dict: unexpected keys {len(uk)}. Example: {uk[:8]}")


def load_checkpoint(path, model, device, optimizer=None, scheduler=None, logger=None):
    ckpt = torch.load(path, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        load_state_dict_flexible(model, ckpt["model_state_dict"], logger=logger)

        if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception as e:
                if logger is not None:
                    logger.warning(f"Optimizer state not loaded (mismatch): {e}")

        if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
            try:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            except Exception as e:
                if logger is not None:
                    logger.warning(f"Scheduler state not loaded (mismatch): {e}")

        epoch = int(ckpt.get("epoch", 0))
        best_metric = float(ckpt.get("best_metric", -1.0))
        best_thr = float(ckpt.get("best_thr", 0.5))
        meta = ckpt.get("meta", {})
        return epoch, best_metric, best_thr, meta

    load_state_dict_flexible(model, ckpt, logger=logger)
    return 0, -1.0, 0.5, {}


def set_module_trainable(module: nn.Module, enabled: bool):
    if module is None:
        return
    flag = bool(enabled)
    for p in module.parameters():
        p.requires_grad = flag


def configure_trainable_modules(model: nn.Module, args, logger=None):
    freeze_backbone = bool(int(getattr(args, "freeze_backbone", 0)))
    freeze_voxel_mlp = bool(int(getattr(args, "freeze_voxel_mlp", 0)))
    freeze_point_refine = bool(int(getattr(args, "freeze_point_refine", 0)))
    freeze_fg_head = bool(int(getattr(args, "freeze_fg_head", 0)))
    freeze_context_fuse = bool(int(getattr(args, "freeze_context_fuse", 0)))
    freeze_center_head = bool(int(getattr(args, "freeze_center_head", 0)))
    freeze_offset_head = bool(int(getattr(args, "freeze_offset_head", 0)))

    set_module_trainable(getattr(model, "backbone", None), not freeze_backbone)
    set_module_trainable(getattr(model, "backbone_extra", None), not freeze_backbone)
    set_module_trainable(getattr(model, "voxel_mlp", None), not freeze_voxel_mlp)
    set_module_trainable(getattr(model, "point_stem", None), not freeze_point_refine)
    set_module_trainable(getattr(model, "scale_projs", None), not freeze_point_refine)
    set_module_trainable(getattr(model, "scale_fuse", None), not freeze_point_refine)
    set_module_trainable(getattr(model, "scale_gate", None), not freeze_point_refine)
    set_module_trainable(getattr(model, "point_refine", None), not freeze_point_refine)
    set_module_trainable(getattr(model, "context_fuse", None), not freeze_context_fuse)
    set_module_trainable(getattr(model, "context_gate", None), not freeze_context_fuse)
    set_module_trainable(getattr(model, "object_trunk", None), not freeze_context_fuse)
    set_module_trainable(getattr(model, "fg_head", None), not freeze_fg_head)
    set_module_trainable(getattr(model, "center_head", None), not freeze_center_head)
    set_module_trainable(getattr(model, "offset_head", None), not freeze_offset_head)
    set_module_trainable(getattr(model, "size_head", None), not freeze_offset_head)

    trainable = sum(int(p.numel()) for p in model.parameters() if p.requires_grad)
    total = sum(int(p.numel()) for p in model.parameters())
    if logger is not None:
        logger.info(
            f"Trainable modules | backbone={int(not freeze_backbone)} | voxel_mlp={int(not freeze_voxel_mlp)} | "
            f"point_refine={int(not freeze_point_refine)} | context={int(not freeze_context_fuse)} | "
            f"fg_head={int(not freeze_fg_head)} | center_head={int(not freeze_center_head)} | "
            f"offset_head={int(not freeze_offset_head)} | "
            f"params={trainable}/{total}"
        )


@torch.no_grad()
def infer_prob_one_batch(
    model,
    device,
    batch,
    center_prob_mix: float = 0.15,
    center_prob_power: float = 0.50,
    vote_max_offset_norm: float = 4.0,
):
    coords_b, feats_b, pts_scene_b, raw_xyz_b, pin_b, p2v_b, pb_b, fg_b, center_b, offset_b, offset_w_b, size_b, size_w_b, scene_scale_b, inst_b, names = batch

    coords_b = coords_b.to(device)
    feats_b = feats_b.to(device)
    pin_b = pin_b.to(device)
    p2v_b = p2v_b.to(device)
    pb_b = pb_b.to(device)

    feats_b = torch_safe_nan_to_num(feats_b, nan=0.0, posinf=0.0, neginf=0.0)
    pin_b = torch_safe_nan_to_num(pin_b, nan=0.0, posinf=0.0, neginf=0.0)

    spatial_shape = calculate_spatial_shape(coords_b)
    batch_size = int(coords_b[:, 0].max().item() + 1)

    use_amp = bool(device.type == "cuda")
    with torch.cuda.amp.autocast(enabled=use_amp):
        outputs = model(
            coords_b,
            feats_b,
            batch_size=batch_size,
            spatial_shape=spatial_shape,
            point2voxel=p2v_b,
            point_inputs=pin_b,
            point_batch_ids=pb_b,
        )
    outputs = unpack_model_outputs(outputs)
    fg_prob = torch.sigmoid(outputs["fg_logits"]).detach().cpu().numpy().astype(np.float32)
    center_prob = torch.sigmoid(outputs["center_logits"]).detach().cpu().numpy().astype(np.float32)
    score_prob = fuse_fg_center_prob_np(
        fg_prob,
        center_prob,
        center_mix=center_prob_mix,
        center_power=center_prob_power,
    )
    offset_scene = outputs["offset_pred"].detach().cpu().numpy().astype(np.float32)
    offset_scene = clip_vector_norm_np(offset_scene, max_norm=float(vote_max_offset_norm))
    raw_xyz = raw_xyz_b.detach().cpu().numpy().astype(np.float32)
    scene_scale = scene_scale_b.detach().cpu().numpy().astype(np.float32)
    vote_xyz = raw_xyz + offset_scene * scene_scale[:, None]
    target = fg_b.detach().cpu().numpy().astype(np.float32)
    inst = inst_b.detach().cpu().numpy().astype(np.int64)
    return fg_prob, center_prob, score_prob, vote_xyz.astype(np.float32), raw_xyz, target, inst, names


@torch.no_grad()
def collect_scene_inference_cache(
    model,
    loader,
    device,
    center_prob_mix: float = 0.15,
    center_prob_power: float = 0.50,
    vote_max_offset_norm: float = 4.0,
):
    cache = []
    model.eval()
    for batch in loader:
        fg_prob, center_prob, score_prob, vote_xyz, raw_xyz, target, inst, names = infer_prob_one_batch(
            model,
            device,
            batch,
            center_prob_mix=center_prob_mix,
            center_prob_power=center_prob_power,
            vote_max_offset_norm=vote_max_offset_norm,
        )
        cache.append(
            {
                "name": names[0],
                "raw_xyz": raw_xyz.astype(np.float32),
                "fg_prob": fg_prob.astype(np.float32),
                "center_prob": center_prob.astype(np.float32),
                "score_prob": score_prob.astype(np.float32),
                "vote_xyz": vote_xyz.astype(np.float32),
                "target": target.astype(np.float32),
                "inst": inst.astype(np.int64),
            }
    )
    return cache


@torch.no_grad()
def collect_scene_inference_cache_from_scenes(
    model,
    scenes,
    device,
    voxel_size: float,
    max_points_per_voxel: int,
    max_voxels: int,
    feature_mode: str,
    voxel_feature_mode: str,
    center_prob_mix: float = 0.15,
    center_prob_power: float = 0.50,
    vote_max_offset_norm: float = 4.0,
    chunk_max_points: int = 260000,
    chunk_min_points: int = 32000,
    chunk_overlap: float = 2.0,
    chunk_max_span_x: float = 192.0,
    chunk_max_span_y: float = 192.0,
    chunk_max_span_z: float = 96.0,
    logger=None,
):
    cache = []
    model.eval()
    for scene in scenes:
        fg_prob, center_prob, score_prob, vote_xyz, raw_xyz, target, inst, names = infer_scene_prob_chunked(
            model,
            device,
            scene,
            voxel_size=voxel_size,
            max_points_per_voxel=max_points_per_voxel,
            max_voxels=max_voxels,
            feature_mode=feature_mode,
            voxel_feature_mode=voxel_feature_mode,
            center_prob_mix=center_prob_mix,
            center_prob_power=center_prob_power,
            vote_max_offset_norm=vote_max_offset_norm,
            chunk_max_points=chunk_max_points,
            chunk_min_points=chunk_min_points,
            chunk_overlap=chunk_overlap,
            chunk_max_span_x=chunk_max_span_x,
            chunk_max_span_y=chunk_max_span_y,
            chunk_max_span_z=chunk_max_span_z,
        )
        cache.append(
            {
                "name": names[0],
                "raw_xyz": raw_xyz.astype(np.float32),
                "fg_prob": fg_prob.astype(np.float32),
                "center_prob": center_prob.astype(np.float32),
                "score_prob": score_prob.astype(np.float32),
                "vote_xyz": vote_xyz.astype(np.float32),
                "target": target.astype(np.float32),
                "inst": inst.astype(np.int64),
            }
        )
    return cache


def build_eval_postproc_cfg(args, postproc_priors: dict, base_thr: float, output_mode: str):
    priors = dict(postproc_priors or {})
    raw_keep_topk = int(getattr(args, "keep_topk", 0))
    keep_topk = int(max(1, raw_keep_topk if raw_keep_topk > 0 else priors.get("keep_topk", 1)))
    adaptive_threshold = int(getattr(args, "adaptive_threshold", -1))
    cfg = {
        "output_mode": str(output_mode),
        "base_thr": float(base_thr),
        "adaptive_threshold": int(1 if adaptive_threshold < 0 else adaptive_threshold),
        "binary_cluster_eps": float(getattr(args, "binary_cluster_eps", -1.0)),
        "binary_cluster_min_samples": int(getattr(args, "binary_cluster_min_samples", -1)),
        "binary_min_component_points": int(getattr(args, "binary_min_component_points", -1)),
        "binary_max_component_points": int(getattr(args, "binary_max_component_points", -1)),
        "cluster_eps": float(getattr(args, "cluster_eps", 0.60)),
        "grow_cluster_min_samples": int(getattr(args, "grow_cluster_min_samples", 6)),
        "seed_min_points": int(getattr(args, "seed_min_points", 24)),
        "support_radius": float(getattr(args, "support_radius", -1.0)),
        "min_cluster_score": float(getattr(args, "min_cluster_score", 0.12)),
        "keep_topk": int(keep_topk),
        "fallback_keep_best": int(bool(getattr(args, "fallback_keep_best", 1))),
        "merge_small_cluster_points": float(getattr(args, "merge_small_cluster_points", -1.0)),
        "merge_xy_gap": float(getattr(args, "merge_xy_gap", -1.0)),
        "merge_z_gap": float(getattr(args, "merge_z_gap", -1.0)),
        "merge_xy_overlap": float(getattr(args, "merge_xy_overlap", 0.22)),
    }
    return cfg


def evaluate_cached_predictions(cache, postproc_cfg: dict, postproc_priors: dict):
    counts = empty_eval_counts()
    per_scene = []
    for item in cache:
        pred_inst, info = predict_scene_labels(
            xyz_raw=item["raw_xyz"],
            fg_prob=item["fg_prob"],
            base_thr=float(postproc_cfg.get("base_thr", 0.35)),
            postproc_priors=postproc_priors,
            seed_prob=item.get("score_prob"),
            score_prob=item.get("score_prob"),
            vote_xyz=item.get("vote_xyz"),
            output_mode=str(postproc_cfg.get("output_mode", "binary_cc")),
            adaptive_threshold=bool(int(postproc_cfg.get("adaptive_threshold", 1))),
            binary_cluster_eps=float(postproc_cfg.get("binary_cluster_eps", -1.0)),
            binary_cluster_min_samples=int(postproc_cfg.get("binary_cluster_min_samples", -1)),
            binary_min_component_points=int(postproc_cfg.get("binary_min_component_points", -1)),
            binary_max_component_points=int(postproc_cfg.get("binary_max_component_points", -1)),
            seed_threshold=float(postproc_cfg.get("seed_threshold", -1.0)),
            grow_threshold=float(postproc_cfg.get("grow_threshold", -1.0)),
            support_threshold=float(postproc_cfg.get("support_threshold", -1.0)),
            cluster_eps=float(postproc_cfg.get("cluster_eps", 0.60)),
            grow_cluster_min_samples=int(postproc_cfg.get("grow_cluster_min_samples", 6)),
            seed_min_points=int(postproc_cfg.get("seed_min_points", 24)),
            support_radius=float(postproc_cfg.get("support_radius", -1.0)),
            min_cluster_score=float(postproc_cfg.get("min_cluster_score", 0.12)),
            keep_topk=int(postproc_cfg.get("keep_topk", 3)),
            fallback_keep_best=bool(int(postproc_cfg.get("fallback_keep_best", 1))),
            merge_small_cluster_points=float(postproc_cfg.get("merge_small_cluster_points", -1.0)),
            merge_xy_gap=float(postproc_cfg.get("merge_xy_gap", -1.0)),
            merge_z_gap=float(postproc_cfg.get("merge_z_gap", -1.0)),
            merge_xy_overlap=float(postproc_cfg.get("merge_xy_overlap", 0.22)),
        )
        scene_counts = evaluate_pred_vs_gt(pred_inst, item["inst"], xyz_raw=item["raw_xyz"])
        add_eval_counts(counts, scene_counts)
        per_scene.append({"name": item["name"], "info": info, "counts": scene_counts})

    summary = summarize_eval_counts(counts)
    return summary, counts, per_scene


def calibrate_postproc(cache, postproc_priors: dict, base_thr: float, args, logger=None):
    if len(cache) == 0:
        return {}, {}

    cache_calib = list(cache)
    limit = int(getattr(args, "calibration_scene_limit", 0))
    if limit > 0 and len(cache_calib) > limit:
        idx = np.linspace(0, len(cache_calib) - 1, num=limit, dtype=np.int64)
        cache_calib = [cache_calib[int(i)] for i in idx.tolist()]
        if logger is not None:
            logger.info(f"Calibration subset: {len(cache_calib)}/{len(cache)} scenes")

    keep_topk = int(max(1, args.keep_topk))
    keep_topk_grid = sorted(set([1, keep_topk]))
    fallback_grid = [0, 1]
    min_comp_ref = int(max(8, round(postproc_priors.get("binary_min_component_points", 48.0))))
    max_comp_ref = int(max(0, round(postproc_priors.get("binary_max_component_points", 0.0))))
    bin_eps_ref = float(postproc_priors.get("binary_cluster_eps", max(0.28, float(args.cluster_eps) * 0.8)))

    thr_hi_cap = max(float(getattr(args, "val_thr_max", 0.80)), 0.90)
    thr_lo = max(args.val_thr_min, float(base_thr) - 0.22, 0.10)
    thr_hi = min(thr_hi_cap, float(base_thr) + 0.24, 0.95)
    if thr_hi <= thr_lo:
        thr_lo = max(args.val_thr_min, 0.10)
        thr_hi = min(thr_hi_cap, 0.90)
    thr_grid = np.linspace(thr_lo, thr_hi, num=min(11, max(7, int(args.val_thr_num)))).astype(np.float32)
    min_comp_grid = sorted(set([max(8, int(round(min_comp_ref * s))) for s in (0.75, 1.25)]))
    bin_eps_grid = sorted(set([round(max(0.20, bin_eps_ref * s), 4) for s in (0.95, 1.10)]))
    inst_score_grid = [0.05, 0.08, 0.14]

    best_summary = None
    best_cfg = None

    def _maybe_update(cfg):
        nonlocal best_summary, best_cfg
        summary, _, _ = evaluate_cached_predictions(cache_calib, cfg, postproc_priors)
        if (best_summary is None) or (summary["score"] > best_summary["score"]):
            best_summary = dict(summary)
            best_cfg = dict(cfg)

    for thr in thr_grid.tolist():
        for min_comp in min_comp_grid:
            for bin_eps in bin_eps_grid:
                for topk in keep_topk_grid:
                    for fallback_keep_best in fallback_grid:
                        cfg = {
                            "output_mode": "binary_cc",
                            "base_thr": float(thr),
                            "adaptive_threshold": 1,
                            "binary_cluster_eps": float(bin_eps),
                            "binary_cluster_min_samples": int(max(1, args.binary_cluster_min_samples if args.binary_cluster_min_samples > 0 else 3)),
                            "binary_min_component_points": int(min_comp),
                            "binary_max_component_points": int(max_comp_ref),
                            "keep_topk": int(topk),
                            "fallback_keep_best": int(fallback_keep_best),
                            "cluster_eps": float(args.cluster_eps),
                            "grow_cluster_min_samples": int(args.grow_cluster_min_samples),
                            "seed_min_points": int(args.seed_min_points),
                        }
                        _maybe_update(cfg)

    for thr in thr_grid.tolist():
        for min_comp in min_comp_grid:
            for fallback_keep_best in fallback_grid:
                cfg = {
                    "output_mode": "binary",
                    "base_thr": float(thr),
                    "adaptive_threshold": 1,
                    "binary_cluster_eps": float(bin_eps_ref),
                    "binary_cluster_min_samples": int(max(1, args.binary_cluster_min_samples if args.binary_cluster_min_samples > 0 else 3)),
                    "binary_min_component_points": int(min_comp),
                    "binary_max_component_points": int(max_comp_ref),
                    "keep_topk": 1,
                    "fallback_keep_best": int(fallback_keep_best),
                    "cluster_eps": float(args.cluster_eps),
                    "grow_cluster_min_samples": int(args.grow_cluster_min_samples),
                    "seed_min_points": int(args.seed_min_points),
                }
                _maybe_update(cfg)

    for thr in thr_grid.tolist():
        for min_cluster_score in inst_score_grid:
            for topk in keep_topk_grid:
                for fallback_keep_best in fallback_grid:
                    cfg = {
                        "output_mode": "instance",
                        "base_thr": float(thr),
                        "adaptive_threshold": 1,
                        "cluster_eps": float(args.cluster_eps),
                        "grow_cluster_min_samples": int(args.grow_cluster_min_samples),
                        "seed_min_points": int(args.seed_min_points),
                        "support_radius": float(args.support_radius),
                        "min_cluster_score": float(min_cluster_score),
                        "keep_topk": int(topk),
                        "fallback_keep_best": int(fallback_keep_best),
                        "merge_small_cluster_points": float(args.merge_small_cluster_points),
                        "merge_xy_gap": float(args.merge_xy_gap),
                        "merge_z_gap": float(args.merge_z_gap),
                        "merge_xy_overlap": float(args.merge_xy_overlap),
                    }
                    _maybe_update(cfg)

    if logger is not None and best_summary is not None:
        logger.info(f"Best calibrated postproc: {best_cfg}")
        logger.info(
            f"Calibrated metrics | score={best_summary['score']:.4f} | obj_rec={best_summary['obj_rec']:.4f} | "
            f"obj_prec={best_summary['obj_prec']:.4f} | obj_f1={best_summary['obj_f1']:.4f} | "
            f"bbox_score={best_summary['bbox_score']:.4f} | bbox_rel_max={best_summary['bbox_rel_max']:.4f} | "
            f"fg_f1={best_summary['fg_f1']:.4f} | fg_iou={best_summary['fg_iou']:.4f}"
        )

    if best_cfg and len(cache_calib) != len(cache):
        full_summary, _, _ = evaluate_cached_predictions(cache, best_cfg, postproc_priors)
        if logger is not None:
            logger.info(
                f"Full-val calibrated metrics | score={full_summary['score']:.4f} | obj_rec={full_summary['obj_rec']:.4f} | "
                f"obj_prec={full_summary['obj_prec']:.4f} | obj_f1={full_summary['obj_f1']:.4f} | "
                f"bbox_score={full_summary['bbox_score']:.4f} | bbox_rel_max={full_summary['bbox_rel_max']:.4f} | "
                f"fg_f1={full_summary['fg_f1']:.4f} | fg_iou={full_summary['fg_iou']:.4f}"
            )
        return best_cfg, full_summary

    return best_cfg or {}, best_summary or {}


def evaluate_val(model, val_loader, device, thr_candidates, args=None, postproc_priors=None, logger=None, val_scenes=None):
    if val_scenes is not None:
        cache = collect_scene_inference_cache_from_scenes(
            model,
            val_scenes,
            device,
            voxel_size=float(getattr(args, "voxel_size", 0.02)) if args is not None else 0.02,
            max_points_per_voxel=int(getattr(args, "max_points_per_voxel", 50)) if args is not None else 50,
            max_voxels=int(getattr(args, "max_voxels", 0)) if args is not None else 0,
            feature_mode=str(getattr(args, "feature_mode", "safe")) if args is not None else "safe",
            voxel_feature_mode=str(getattr(args, "voxel_feature_mode", "extra_only")) if args is not None else "extra_only",
            center_prob_mix=float(getattr(args, "center_prob_mix", 0.15)) if args is not None else 0.15,
            center_prob_power=float(getattr(args, "center_prob_power", 0.50)) if args is not None else 0.50,
            vote_max_offset_norm=float(getattr(args, "vote_max_offset_norm", 4.0)) if args is not None else 4.0,
            chunk_max_points=int(getattr(args, "infer_chunk_max_points", 260000)) if args is not None else 260000,
            chunk_min_points=int(getattr(args, "infer_chunk_min_points", 32000)) if args is not None else 32000,
            chunk_overlap=float(getattr(args, "infer_chunk_overlap", 2.0)) if args is not None else 2.0,
            chunk_max_span_x=float(getattr(args, "infer_chunk_max_span_x", 192.0)) if args is not None else 192.0,
            chunk_max_span_y=float(getattr(args, "infer_chunk_max_span_y", 192.0)) if args is not None else 192.0,
            chunk_max_span_z=float(getattr(args, "infer_chunk_max_span_z", 96.0)) if args is not None else 96.0,
            logger=logger,
        )
    else:
        cache = collect_scene_inference_cache(
            model,
            val_loader,
            device,
            center_prob_mix=float(getattr(args, "center_prob_mix", 0.15)) if args is not None else 0.15,
            center_prob_power=float(getattr(args, "center_prob_power", 0.50)) if args is not None else 0.50,
            vote_max_offset_norm=float(getattr(args, "vote_max_offset_norm", 4.0)) if args is not None else 4.0,
        )
    n = len(cache)
    if n == 0:
        return {
            "metric_name": "fg_iou",
            "val_metric": 0.0,
            "val_fg_iou": 0.0,
            "val_prec": 0.0,
            "val_rec": 0.0,
            "best_thr": 0.5,
        }

    metric_name = str(getattr(args, "val_metric", "score")).lower() if args is not None else "fg_iou"
    if metric_name != "score":
        best_iou = -1.0
        best_thr = 0.5
        best_prec = 0.0
        best_rec = 0.0
        for thr in thr_candidates:
            ious = []
            precs = []
            recs = []
            for item in cache:
                m = binary_metrics_from_prob(item.get("score_prob", item["fg_prob"]), item["target"], thr=float(thr))
                ious.append(m["iou"])
                precs.append(m["prec"])
                recs.append(m["rec"])

            miou = float(np.mean(ious))
            if miou > best_iou:
                best_iou = float(miou)
                best_thr = float(thr)
                best_prec = float(np.mean(precs))
                best_rec = float(np.mean(recs))

        if logger is not None:
            logger.info(
                f"Val sweep | metric=fg_iou | best_thr={best_thr:.3f} | fgIoU={best_iou:.4f} | "
                f"prec={best_prec:.4f} | rec={best_rec:.4f}"
            )
        return {
            "metric_name": "fg_iou",
            "val_metric": float(best_iou),
            "val_fg_iou": float(best_iou),
            "val_prec": float(best_prec),
            "val_rec": float(best_rec),
            "best_thr": float(best_thr),
        }

    priors = dict(postproc_priors or {})
    thr_grid = np.asarray(thr_candidates, dtype=np.float32)
    thr_limit = int(max(1, getattr(args, "val_eval_thr_limit", len(thr_grid))))
    if thr_grid.size > thr_limit:
        idx = np.linspace(0, thr_grid.size - 1, num=thr_limit, dtype=np.int64)
        thr_grid = thr_grid[idx]

    mode_text = str(getattr(args, "val_output_modes", "binary,binary_cc")).strip()
    modes = [m.strip().lower() for m in mode_text.split(",") if m.strip()]
    modes = [m for m in modes if m in ("binary", "binary_cc", "instance")]
    if len(modes) == 0:
        modes = ["binary", "binary_cc"]

    best_summary = None
    best_cfg = None
    for mode in modes:
        for thr in thr_grid.tolist():
            cfg = build_eval_postproc_cfg(args, priors, base_thr=float(thr), output_mode=mode)
            cfg["adaptive_threshold"] = 0
            summary, _, _ = evaluate_cached_predictions(cache, cfg, priors)
            if (best_summary is None) or (summary["score"] > best_summary["score"]):
                best_summary = dict(summary)
                best_cfg = dict(cfg)

    if best_summary is None:
        best_summary = {
            "score": 0.0,
            "fg_iou": 0.0,
            "fg_prec": 0.0,
            "fg_rec": 0.0,
            "obj_prec": 0.0,
            "obj_rec": 0.0,
            "obj_f1": 0.0,
            "obj_cover_mean": 0.0,
            "pred_purity_mean": 0.0,
            "bbox_score": 0.0,
        }
        best_cfg = {"base_thr": 0.5, "output_mode": "binary"}

    if logger is not None:
        logger.info(
            f"Val sweep | metric=score | mode={best_cfg['output_mode']} | best_thr={float(best_cfg['base_thr']):.3f} | "
            f"score={best_summary['score']:.4f} | obj_f1={best_summary['obj_f1']:.4f} | "
            f"obj_cover={best_summary['obj_cover_mean']:.4f} | pred_purity={best_summary['pred_purity_mean']:.4f} | "
            f"bbox_score={best_summary['bbox_score']:.4f} | fg_iou={best_summary['fg_iou']:.4f}"
        )

    return {
        "metric_name": "score",
        "val_metric": float(best_summary["score"]),
        "val_fg_iou": float(best_summary["fg_iou"]),
        "val_prec": float(best_summary["fg_prec"]),
        "val_rec": float(best_summary["fg_rec"]),
        "val_obj_prec": float(best_summary["obj_prec"]),
        "val_obj_rec": float(best_summary["obj_rec"]),
        "val_obj_f1": float(best_summary["obj_f1"]),
        "val_obj_cover_mean": float(best_summary["obj_cover_mean"]),
        "val_pred_purity_mean": float(best_summary["pred_purity_mean"]),
        "val_bbox_score": float(best_summary["bbox_score"]),
        "best_thr": float(best_cfg.get("base_thr", 0.5)),
        "best_postproc": dict(best_cfg),
    }


def train(args):
    logger, _ = setup_logging()
    set_seed(args.seed)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    logger.info(f"Device: {device}")

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    files = glob.glob(os.path.join(args.data_dir, "*.csv"))
    if not files:
        raise ValueError(f"No csv found in: {args.data_dir}")

    extra_train_files = []
    if args.extra_train_dir:
        extra_train_files = glob.glob(os.path.join(args.extra_train_dir, "*.csv"))
        if not extra_train_files:
            raise ValueError(f"No csv found in extra_train_dir: {args.extra_train_dir}")

    if args.val_dir:
        val_files = glob.glob(os.path.join(args.val_dir, "*.csv"))
        if not val_files:
            raise ValueError(f"No csv found in val_dir: {args.val_dir}")
        train_files = sorted(files)
        train_files = deduplicate_files(train_files, logger=logger)
        logger.info(
            f"Using external validation set | train_dir={args.data_dir} ({len(train_files)} files after dedup) | "
            f"val_dir={args.val_dir} ({len(val_files)} files)"
        )
    else:
        files_dedup = deduplicate_files(sorted(files), logger=logger)
        train_files, val_files = split_files(files_dedup, train_ratio=args.train_split, seed=args.seed)
        logger.info(f"Total files: {len(files)} (dedup: {len(files_dedup)}) | Train: {len(train_files)} | Val: {len(val_files)}")

    if len(extra_train_files) > 0:
        extra_train_files = deduplicate_files(sorted(extra_train_files), logger=logger)
        logger.info(f"Using extra labeled train dir | extra_train_dir={args.extra_train_dir} ({len(extra_train_files)} files after dedup)")

    train_scenes = load_scenes(
        train_files + extra_train_files,
        input_dim=args.input_dim,
        normalize_mode=args.normalize_mode,
        scale_quantile=args.scale_quantile,
        scale_clip_min=args.scale_clip_min,
        scale_clip_max=args.scale_clip_max,
        abs_coord_scale=args.abs_coord_scale,
        use_abs_coords=bool(args.use_abs_coords),
        coord_unit_scale=args.coord_unit_scale,
        auto_unit_normalize=bool(args.auto_unit_normalize),
        auto_mm_threshold=args.auto_mm_threshold,
        pre_downsample_voxel=args.pre_downsample_voxel,
        logger=logger,
    )
    val_scenes = load_scenes(
        val_files,
        input_dim=args.input_dim,
        normalize_mode=args.normalize_mode,
        scale_quantile=args.scale_quantile,
        scale_clip_min=args.scale_clip_min,
        scale_clip_max=args.scale_clip_max,
        abs_coord_scale=args.abs_coord_scale,
        use_abs_coords=bool(args.use_abs_coords),
        coord_unit_scale=args.coord_unit_scale,
        auto_unit_normalize=bool(args.auto_unit_normalize),
        auto_mm_threshold=args.auto_mm_threshold,
        pre_downsample_voxel=args.pre_downsample_voxel,
        logger=logger,
    )

    gt_priors = estimate_gt_priors(train_scenes)
    logger.info(f"GT priors: {gt_priors}")

    point_in_dim = train_scenes[0]["point_inputs"].shape[1]
    voxel_point_dim = select_voxel_point_features(
        train_scenes[0]["xyz_scene"],
        train_scenes[0]["point_inputs"],
        extra_dim=int(train_scenes[0].get("extra_dim", max(0, point_in_dim - 3))),
        voxel_feature_mode=args.voxel_feature_mode,
    ).shape[1]
    in_channels = int(voxel_point_dim) * 3 + 2
    logger.info(
        f"Derived point_in_dim={point_in_dim} | voxel_point_dim={int(voxel_point_dim)} | "
        f"voxel in_channels={in_channels}"
    )

    train_ds = CropTrainDataset(
        train_scenes,
        voxel_size=args.voxel_size,
        max_points_per_voxel=args.max_points_per_voxel,
        max_voxels=args.max_voxels,
        max_points=args.max_points,
        feature_mode=args.feature_mode,
        voxel_feature_mode=args.voxel_feature_mode,
        samples_per_scene=args.samples_per_scene,
        crop_half_x=args.crop_half_x,
        crop_half_y=args.crop_half_y,
        crop_half_z=args.crop_half_z,
        positive_fraction=args.positive_fraction,
        hard_negative_fraction=args.hard_negative_fraction,
        hard_neg_shell_min=args.hard_neg_shell_min,
        hard_neg_shell_max=args.hard_neg_shell_max,
        hard_neg_max_fg_points=args.hard_neg_max_fg_points,
        aug_jitter_xyz_std=args.aug_jitter_xyz_std,
        aug_dropout=args.aug_dropout,
        aug_rotate_z_deg=args.aug_rotate_z_deg,
        aug_flip_x_prob=args.aug_flip_x_prob,
        aug_flip_y_prob=args.aug_flip_y_prob,
        aug_bg_dropout_prob=args.aug_bg_dropout_prob,
        aug_bg_dropout_min=args.aug_bg_dropout_min,
        aug_bg_dropout_max=args.aug_bg_dropout_max,
        aug_normal_noise_std=args.aug_normal_noise_std,
        positive_context_scale_min=args.positive_context_scale_min,
        positive_context_scale_max=args.positive_context_scale_max,
        positive_safe_margin=args.positive_safe_margin,
        positive_jitter_frac=args.positive_jitter_frac,
        positive_min_coverage=args.positive_min_coverage,
        hard_neg_near_obj_prob=args.hard_neg_near_obj_prob,
        hard_neg_context_margin=args.hard_neg_context_margin,
        hard_neg_exclude_margin=args.hard_neg_exclude_margin,
    )

    val_ds = FullSceneDataset(
        val_scenes,
        voxel_size=args.voxel_size,
        max_points_per_voxel=args.max_points_per_voxel,
        max_voxels=args.max_voxels,
        feature_mode=args.feature_mode,
        voxel_feature_mode=args.voxel_feature_mode,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=0)

    model = SparseFGNet(
        in_channels=in_channels,
        point_in_dim=point_in_dim,
        init_channels=args.init_channels,
        norm=args.norm,
        dropout=args.dropout,
        point_refine_dim=args.point_refine_dim,
        fg_prior=args.fg_prior,
        center_prior=args.center_prior,
        enable_deep_stage=args.enable_deep_stage,
        context_stage_max_voxels=args.context_stage_max_voxels,
        deep_stage_max_voxels=args.deep_stage_max_voxels,
    ).to(device)

    criterion = FGLoss(
        w_focal=args.w_focal,
        w_tversky=args.w_tversky,
        w_center=args.w_center,
        w_offset=args.w_offset,
        w_size=args.w_size,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        hnm_ratio=args.hnm_ratio,
        hnm_min_neg=args.hnm_min_neg,
        hnm_max_neg=args.hnm_max_neg,
        tversky_alpha=args.tversky_alpha,
        tversky_beta=args.tversky_beta,
        center_bg_weight=args.center_bg_weight,
        center_fg_weight=args.center_fg_weight,
        center_peak_weight=args.center_peak_weight,
        offset_weight_power=args.offset_weight_power,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup_epochs = int(getattr(args, "warmup_epochs", 5))
    if warmup_epochs > 0 and args.epochs > warmup_epochs:
        warmup_sched = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
        )
        cosine_sched = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, args.epochs - warmup_epochs), eta_min=args.lr * 0.05
        )
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs]
        )
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.05)

    use_amp = bool(args.amp == 1) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    logger.info(f"AMP enabled: {use_amp}")
    metric_label = str(args.val_metric).lower()

    start_epoch = 0
    best_metric = -1.0
    best_thr = float(args.default_fg_threshold)

    if args.resume_path:
        logger.info(f"Resuming from checkpoint: {args.resume_path}")
        ep, bm, bt, meta = load_checkpoint(args.resume_path, model, device, optimizer=optimizer, scheduler=scheduler, logger=logger)
        start_epoch = int(ep)
        best_metric = float(bm)
        best_thr = float(bt)
        logger.info(f"Resume: epoch={start_epoch}, best_fg_iou={best_metric:.4f}, best_thr={best_thr:.3f}, meta={meta}")
    elif args.pretrained_weights:
        logger.info(f"Loading pretrained weights: {args.pretrained_weights}")
        ep, bm, bt, meta = load_checkpoint(args.pretrained_weights, model, device, optimizer=None, scheduler=None, logger=logger)
        logger.info(f"Loaded weights from epoch={ep}, meta={meta}")

    configure_trainable_modules(model, args, logger=logger)

    patience_counter = 0
    best_path = args.model_save_path.replace(".pth", "_best.pth")

    ckpt_meta = {
        "input_dim": int(args.input_dim),
        "point_in_dim": int(point_in_dim),
        "voxel_point_dim": int(voxel_point_dim),
        "in_channels": int(in_channels),
        "init_channels": int(args.init_channels),
        "point_refine_dim": int(args.point_refine_dim),
        "norm": str(args.norm),
        "dropout": float(args.dropout),
        "normalize_mode": str(args.normalize_mode),
        "scale_quantile": float(args.scale_quantile),
        "scale_clip_min": float(args.scale_clip_min),
        "scale_clip_max": float(args.scale_clip_max),
        "abs_coord_scale": float(args.abs_coord_scale),
        "use_abs_coords": int(bool(args.use_abs_coords)),
        "coord_unit_scale": float(args.coord_unit_scale),
        "auto_unit_normalize": int(bool(args.auto_unit_normalize)),
        "auto_mm_threshold": float(args.auto_mm_threshold),
        "pre_downsample_voxel": float(args.pre_downsample_voxel),
        "voxel_size": float(args.voxel_size),
        "max_points_per_voxel": int(args.max_points_per_voxel),
        "feature_mode": str(args.feature_mode),
        "voxel_feature_mode": str(args.voxel_feature_mode),
        "fg_prior": float(args.fg_prior),
        "center_prior": float(args.center_prior),
        "center_prob_mix": float(args.center_prob_mix),
        "center_prob_power": float(args.center_prob_power),
        "vote_max_offset_norm": float(args.vote_max_offset_norm),
        "enable_deep_stage": int(args.enable_deep_stage),
        "context_stage_max_voxels": int(args.context_stage_max_voxels),
        "deep_stage_max_voxels": int(args.deep_stage_max_voxels),
        "infer_chunk_max_points": int(args.infer_chunk_max_points),
        "infer_chunk_min_points": int(args.infer_chunk_min_points),
        "infer_chunk_overlap": float(args.infer_chunk_overlap),
        "infer_chunk_max_span_x": float(args.infer_chunk_max_span_x),
        "infer_chunk_max_span_y": float(args.infer_chunk_max_span_y),
        "infer_chunk_max_span_z": float(args.infer_chunk_max_span_z),
        "crop_half_x": float(args.crop_half_x),
        "crop_half_y": float(args.crop_half_y),
        "crop_half_z": float(args.crop_half_z),
        "aug_rotate_z_deg": float(args.aug_rotate_z_deg),
        "aug_flip_x_prob": float(args.aug_flip_x_prob),
        "aug_flip_y_prob": float(args.aug_flip_y_prob),
        "aug_bg_dropout_prob": float(args.aug_bg_dropout_prob),
        "aug_bg_dropout_min": float(args.aug_bg_dropout_min),
        "aug_bg_dropout_max": float(args.aug_bg_dropout_max),
        "aug_normal_noise_std": float(args.aug_normal_noise_std),
        "positive_context_scale_min": float(args.positive_context_scale_min),
        "positive_context_scale_max": float(args.positive_context_scale_max),
        "positive_safe_margin": float(args.positive_safe_margin),
        "positive_jitter_frac": float(args.positive_jitter_frac),
        "positive_min_coverage": float(args.positive_min_coverage),
        "hard_neg_near_obj_prob": float(args.hard_neg_near_obj_prob),
        "hard_neg_context_margin": float(args.hard_neg_context_margin),
        "hard_neg_exclude_margin": float(args.hard_neg_exclude_margin),
        "cluster_eps": float(args.cluster_eps),
        "grow_cluster_min_samples": int(args.grow_cluster_min_samples),
        "seed_min_points": int(args.seed_min_points),
        "support_radius": float(args.support_radius),
        "merge_small_cluster_points": float(args.merge_small_cluster_points),
        "merge_xy_gap": float(args.merge_xy_gap),
        "merge_z_gap": float(args.merge_z_gap),
        "merge_xy_overlap": float(args.merge_xy_overlap),
        "min_cluster_score": float(args.min_cluster_score),
        "keep_topk": int(args.keep_topk),
        "val_metric": str(args.val_metric),
        "freeze_backbone": int(args.freeze_backbone),
        "freeze_voxel_mlp": int(args.freeze_voxel_mlp),
        "freeze_point_refine": int(args.freeze_point_refine),
        "freeze_context_fuse": int(args.freeze_context_fuse),
        "freeze_fg_head": int(args.freeze_fg_head),
        "freeze_center_head": int(args.freeze_center_head),
        "freeze_offset_head": int(args.freeze_offset_head),
        "w_size": float(args.w_size),
    }
    ckpt_meta.update(gt_priors)
    val_postproc_priors = build_postproc_priors(ckpt_meta)

    thr_candidates = np.linspace(args.val_thr_min, args.val_thr_max, args.val_thr_num).astype(np.float32)
    last_val_metrics = {
        "metric_name": metric_label,
        "val_metric": 0.0,
        "val_fg_iou": 0.0,
        "val_prec": 0.0,
        "val_rec": 0.0,
        "val_obj_prec": 0.0,
        "val_obj_rec": 0.0,
        "val_obj_f1": 0.0,
        "val_bbox_score": 0.0,
        "best_thr": best_thr,
    }

    for epoch in range(start_epoch, args.epochs):
        model.train()
        tr_total = tr_focal = tr_tversky = tr_center = tr_offset = tr_size = 0.0
        tr_pred_ratio = tr_gt_ratio = tr_center_pos = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", disable=bool(int(getattr(args, "disable_tqdm", 0))))
        for batch in pbar:
            coords_b, feats_b, pts_scene_b, raw_xyz_b, pin_b, p2v_b, pb_b, fg_b, center_b, offset_b, offset_w_b, size_b, size_w_b, scene_scale_b, inst_b, names = batch

            coords_b = coords_b.to(device)
            feats_b = feats_b.to(device)
            pin_b = pin_b.to(device)
            p2v_b = p2v_b.to(device)
            pb_b = pb_b.to(device)
            fg_b = fg_b.to(device)
            center_b = center_b.to(device)
            offset_b = offset_b.to(device)
            offset_w_b = offset_w_b.to(device)
            size_b = size_b.to(device)
            size_w_b = size_w_b.to(device)

            feats_b = torch_safe_nan_to_num(feats_b, nan=0.0, posinf=0.0, neginf=0.0)
            pin_b = torch_safe_nan_to_num(pin_b, nan=0.0, posinf=0.0, neginf=0.0)

            spatial_shape = calculate_spatial_shape(coords_b)
            batch_size = int(coords_b[:, 0].max().item() + 1)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(
                    coords_b,
                    feats_b,
                    batch_size=batch_size,
                    spatial_shape=spatial_shape,
                    point2voxel=p2v_b,
                    point_inputs=pin_b,
                    point_batch_ids=pb_b,
                )
                total, loss_stats = criterion(
                    outputs,
                    fg_b,
                    center_target=center_b,
                    offset_target=offset_b,
                    offset_weight=offset_w_b,
                    size_target=size_b,
                    size_weight=size_w_b,
                )

            if not torch.isfinite(total):
                logger.warning("Non-finite loss detected. Skipping step.")
                continue

            if use_amp:
                scaler.scale(total).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()

            tr_total += float(total.item())
            tr_focal += float(loss_stats["focal"].item())
            tr_tversky += float(loss_stats["tversky"].item())
            tr_center += float(loss_stats["center"].item())
            tr_offset += float(loss_stats["offset"].item())
            tr_size += float(loss_stats["size"].item())

            with torch.no_grad():
                outputs = unpack_model_outputs(outputs)
                fg_prob = torch.sigmoid(outputs["fg_logits"])
                center_prob = torch.sigmoid(outputs["center_logits"])
                score_prob = fuse_fg_center_prob_torch(
                    fg_prob,
                    center_prob,
                    center_mix=args.center_prob_mix,
                    center_power=args.center_prob_power,
                )
                pred_ratio = float((score_prob >= 0.5).float().mean().item())
                gt_ratio = float((fg_b > 0.5).float().mean().item())
                center_pos = float(center_prob[fg_b > 0.5].mean().item()) if bool((fg_b > 0.5).any()) else 0.0
                tr_pred_ratio += pred_ratio
                tr_gt_ratio += gt_ratio
                tr_center_pos += center_pos

            pbar.set_postfix(
                {
                    "loss": f"{total.item():.4f}",
                    "focal": f"{loss_stats['focal'].item():.4f}",
                    "tv": f"{loss_stats['tversky'].item():.4f}",
                    "ctr": f"{loss_stats['center'].item():.4f}",
                    "off": f"{loss_stats['offset'].item():.4f}",
                    "size": f"{loss_stats['size'].item():.4f}",
                    "pred@0.5": f"{pred_ratio:.3f}",
                    "center+": f"{center_pos:.3f}",
                    "gt": f"{gt_ratio:.3f}",
                }
            )

        tr_den = max(1, len(train_loader))
        tr_total /= tr_den
        tr_focal /= tr_den
        tr_tversky /= tr_den
        tr_center /= tr_den
        tr_offset /= tr_den
        tr_size /= tr_den
        tr_pred_ratio /= tr_den
        tr_gt_ratio /= tr_den
        tr_center_pos /= tr_den

        do_val = (len(val_scenes) > 0) and (
            ((epoch + 1) % int(max(1, args.val_interval)) == 0)
            or ((epoch + 1) == args.epochs)
        )
        if do_val:
            model.eval()
            if device.type == "cuda":
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            val_metrics = evaluate_val(
                model,
                val_loader,
                device,
                thr_candidates,
                args=args,
                postproc_priors=val_postproc_priors,
                logger=logger,
                val_scenes=val_scenes,
            )
            last_val_metrics = dict(val_metrics)
        else:
            val_metrics = dict(last_val_metrics)

        scheduler.step()

        if do_val:
            if str(val_metrics.get("metric_name", metric_label)).lower() == "score":
                logger.info(
                    f"Epoch {epoch + 1}: Train loss={tr_total:.5f} (focal={tr_focal:.5f}, tversky={tr_tversky:.5f}, center={tr_center:.5f}, offset={tr_offset:.5f}) | "
                    f"size={tr_size:.5f} | pred@0.5={tr_pred_ratio:.4f} | gt={tr_gt_ratio:.4f} | center+={tr_center_pos:.4f} | "
                    f"Val score={val_metrics['val_metric']:.4f} | obj_f1={val_metrics['val_obj_f1']:.4f} | "
                    f"obj_cover={val_metrics['val_obj_cover_mean']:.4f} | bbox_score={val_metrics['val_bbox_score']:.4f} | "
                    f"fgIoU={val_metrics['val_fg_iou']:.4f} | "
                    f"best_thr={val_metrics['best_thr']:.3f} | LR={scheduler.get_last_lr()[0]:.6f}"
                )
            else:
                logger.info(
                    f"Epoch {epoch + 1}: Train loss={tr_total:.5f} (focal={tr_focal:.5f}, tversky={tr_tversky:.5f}, center={tr_center:.5f}, offset={tr_offset:.5f}, size={tr_size:.5f}) | "
                    f"pred@0.5={tr_pred_ratio:.4f} | gt={tr_gt_ratio:.4f} | center+={tr_center_pos:.4f} | "
                    f"Val fgIoU={val_metrics['val_fg_iou']:.4f} | "
                    f"Val prec={val_metrics['val_prec']:.4f} | Val rec={val_metrics['val_rec']:.4f} | "
                    f"best_thr={val_metrics['best_thr']:.3f} | LR={scheduler.get_last_lr()[0]:.6f}"
                )
        else:
            logger.info(
                f"Epoch {epoch + 1}: Train loss={tr_total:.5f} (focal={tr_focal:.5f}, tversky={tr_tversky:.5f}, center={tr_center:.5f}, offset={tr_offset:.5f}, size={tr_size:.5f}) | "
                f"pred@0.5={tr_pred_ratio:.4f} | gt={tr_gt_ratio:.4f} | center+={tr_center_pos:.4f} | "
                f"Val skipped (interval={int(max(1, args.val_interval))}) | "
                f"last {last_val_metrics.get('metric_name', metric_label)}={last_val_metrics.get('val_metric', last_val_metrics['val_fg_iou']):.4f} | "
                f"last_thr={last_val_metrics['best_thr']:.3f} | "
                f"LR={scheduler.get_last_lr()[0]:.6f}"
            )

        save_meta = dict(ckpt_meta)
        if isinstance(last_val_metrics.get("best_postproc"), dict):
            save_meta["best_postproc"] = dict(last_val_metrics["best_postproc"])
        save_checkpoint(args.model_save_path, model, optimizer, scheduler, epoch + 1, best_metric, best_thr, meta=save_meta)
        logger.info(f"Saved last: {args.model_save_path}")

        if do_val:
            cur_iou = float(val_metrics.get("val_metric", val_metrics["val_fg_iou"]))
            cur_thr = float(val_metrics["best_thr"])
            if cur_iou > best_metric:
                best_metric = cur_iou
                best_thr = cur_thr
                best_meta = dict(save_meta)
                if isinstance(val_metrics.get("best_postproc"), dict):
                    best_meta["best_postproc"] = dict(val_metrics["best_postproc"])
                save_checkpoint(best_path, model, optimizer, scheduler, epoch + 1, best_metric, best_thr, meta=best_meta)
                logger.info(f"NEW BEST {val_metrics.get('metric_name', metric_label)}={best_metric:.4f} | thr={best_thr:.3f} -> {best_path}")
                patience_counter = 0
            else:
                patience_counter += 1
        if (epoch + 1) >= int(max(1, args.min_epochs_before_stop)) and patience_counter >= int(args.patience):
            logger.info(f"Early stopping. Best {metric_label}={best_metric:.4f} | best_thr={best_thr:.3f}")
            break

    if int(args.calibrate_postproc) == 1 and len(val_scenes) > 0:
        logger.info("Calibrating output postproc on validation scenes...")
        if os.path.exists(best_path):
            load_checkpoint(best_path, model, device, optimizer=None, scheduler=None, logger=logger)
        model.eval()
        if device.type == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        cache = collect_scene_inference_cache_from_scenes(
            model,
            val_scenes,
            device,
            voxel_size=float(args.voxel_size),
            max_points_per_voxel=int(args.max_points_per_voxel),
            max_voxels=int(args.max_voxels),
            feature_mode=str(args.feature_mode),
            voxel_feature_mode=str(args.voxel_feature_mode),
            center_prob_mix=float(args.center_prob_mix),
            center_prob_power=float(args.center_prob_power),
            vote_max_offset_norm=float(args.vote_max_offset_norm),
            chunk_max_points=int(args.infer_chunk_max_points),
            chunk_min_points=int(args.infer_chunk_min_points),
            chunk_overlap=float(args.infer_chunk_overlap),
            chunk_max_span_x=float(args.infer_chunk_max_span_x),
            chunk_max_span_y=float(args.infer_chunk_max_span_y),
            chunk_max_span_z=float(args.infer_chunk_max_span_z),
            logger=logger,
        )
        postproc_priors = build_postproc_priors(ckpt_meta)
        best_postproc, best_post_metrics = calibrate_postproc(cache, postproc_priors, best_thr, args, logger=logger)
        if best_postproc:
            ckpt_meta["best_postproc"] = dict(best_postproc)
            update_checkpoint_postproc(best_path, best_postproc, meta_updates=ckpt_meta, logger=logger)
            update_checkpoint_postproc(args.model_save_path, best_postproc, meta_updates=ckpt_meta, logger=logger)
            logger.info(f"Calibration summary: {best_post_metrics}")


def predict(args):
    logger, _ = setup_logging()
    set_seed(args.seed)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    logger.info(f"Device: {device}")

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(args.model_path)

    files = sorted(glob.glob(os.path.join(args.input_dir, "*.csv")))
    if not files:
        raise ValueError(f"No csv in: {args.input_dir}")

    ckpt = torch.load(args.model_path, map_location=device)
    meta = ckpt.get("meta", {}) if isinstance(ckpt, dict) else {}

    default_point_in_dim = int(args.input_dim + (3 if bool(args.use_abs_coords) else 0))
    point_in_dim = int(meta.get("point_in_dim", default_point_in_dim))
    default_voxel_point_dim = max(1, int(meta.get("voxel_point_dim", max(3, args.input_dim - 3))))
    in_channels = int(meta.get("in_channels", default_voxel_point_dim * 3 + 2))
    fg_prior = float(meta.get("fg_prior", args.fg_prior))
    center_prior = float(meta.get("center_prior", args.center_prior))
    best_thr = float(ckpt.get("best_thr", args.default_fg_threshold)) if isinstance(ckpt, dict) else float(args.default_fg_threshold)

    model = SparseFGNet(
        in_channels=in_channels,
        point_in_dim=point_in_dim,
        init_channels=int(meta.get("init_channels", args.init_channels)),
        norm=str(meta.get("norm", args.norm)),
        dropout=float(meta.get("dropout", args.dropout)),
        point_refine_dim=int(meta.get("point_refine_dim", args.point_refine_dim)),
        fg_prior=fg_prior,
        center_prior=center_prior,
        enable_deep_stage=int(meta.get("enable_deep_stage", args.enable_deep_stage)),
        context_stage_max_voxels=int(meta.get("context_stage_max_voxels", args.context_stage_max_voxels)),
        deep_stage_max_voxels=int(meta.get("deep_stage_max_voxels", args.deep_stage_max_voxels)),
    ).to(device)

    ep, bm, bt, meta = load_checkpoint(args.model_path, model, device, optimizer=None, scheduler=None, logger=logger)
    model.eval()
    if bt > 0:
        best_thr = bt
    logger.info(f"Loaded model: epoch={ep}, best_fg_iou={bm:.4f}, best_thr={best_thr:.3f}, meta={meta}")

    norm_mode = str(meta.get("normalize_mode", args.normalize_mode))
    scale_quantile = float(meta.get("scale_quantile", args.scale_quantile))
    scale_clip_min = float(meta.get("scale_clip_min", args.scale_clip_min))
    scale_clip_max = float(meta.get("scale_clip_max", args.scale_clip_max))
    abs_coord_scale = float(meta.get("abs_coord_scale", args.abs_coord_scale))
    use_abs_coords = bool(int(meta.get("use_abs_coords", args.use_abs_coords)))
    coord_unit_scale = float(meta.get("coord_unit_scale", args.coord_unit_scale))
    auto_unit_normalize = bool(int(meta.get("auto_unit_normalize", args.auto_unit_normalize)))
    auto_mm_threshold = float(meta.get("auto_mm_threshold", args.auto_mm_threshold))
    pre_downsample_voxel = float(meta.get("pre_downsample_voxel", args.pre_downsample_voxel))
    voxel_size = float(meta.get("voxel_size", args.voxel_size))
    feature_mode = str(meta.get("feature_mode", args.feature_mode))
    voxel_feature_mode = str(meta.get("voxel_feature_mode", args.voxel_feature_mode))
    max_points_per_voxel = int(meta.get("max_points_per_voxel", args.max_points_per_voxel))
    center_prob_mix = float(meta.get("center_prob_mix", args.center_prob_mix))
    center_prob_power = float(meta.get("center_prob_power", args.center_prob_power))
    vote_max_offset_norm = float(meta.get("vote_max_offset_norm", args.vote_max_offset_norm))
    infer_chunk_max_points = int(meta.get("infer_chunk_max_points", args.infer_chunk_max_points))
    infer_chunk_min_points = int(meta.get("infer_chunk_min_points", args.infer_chunk_min_points))
    infer_chunk_overlap = float(meta.get("infer_chunk_overlap", args.infer_chunk_overlap))
    infer_chunk_max_span_x = float(meta.get("infer_chunk_max_span_x", args.infer_chunk_max_span_x))
    infer_chunk_max_span_y = float(meta.get("infer_chunk_max_span_y", args.infer_chunk_max_span_y))
    infer_chunk_max_span_z = float(meta.get("infer_chunk_max_span_z", args.infer_chunk_max_span_z))

    saved_postproc = ckpt.get("best_postproc", meta.get("best_postproc", {})) if isinstance(ckpt, dict) else {}
    postproc_priors = build_postproc_priors(meta)
    logger.info(f"Postproc priors: {postproc_priors}")
    if saved_postproc:
        logger.info(f"Loaded saved postproc: {saved_postproc}")

    os.makedirs(args.output_dir, exist_ok=True)
    use_amp = bool(args.infer_amp) and device.type == "cuda"
    save_fmt = ["%.6f", "%.6f", "%.6f", "%.0f"]
    overall_counts = empty_eval_counts()
    labeled_scene_count = 0

    resolved_output_mode = str(args.output_mode).lower()
    if resolved_output_mode == "auto":
        resolved_output_mode = str(saved_postproc.get("output_mode", "binary_cc"))

    adaptive_threshold = bool(
        int(args.adaptive_threshold)
        if int(args.adaptive_threshold) >= 0
        else int(saved_postproc.get("adaptive_threshold", 1))
    )
    base_thr = float(args.fg_threshold) if args.fg_threshold > 0 else float(saved_postproc.get("base_thr", best_thr))
    resolved_binary_cluster_eps = float(args.binary_cluster_eps) if args.binary_cluster_eps > 0 else float(saved_postproc.get("binary_cluster_eps", -1.0))
    resolved_binary_cluster_min_samples = (
        int(args.binary_cluster_min_samples) if args.binary_cluster_min_samples > 0 else int(saved_postproc.get("binary_cluster_min_samples", -1))
    )
    resolved_binary_min_component_points = (
        int(args.binary_min_component_points) if args.binary_min_component_points > 0 else int(saved_postproc.get("binary_min_component_points", -1))
    )
    resolved_binary_max_component_points = (
        int(args.binary_max_component_points) if args.binary_max_component_points > 0 else int(saved_postproc.get("binary_max_component_points", -1))
    )
    resolved_cluster_eps = float(args.cluster_eps) if args.cluster_eps > 0 else float(saved_postproc.get("cluster_eps", meta.get("cluster_eps", 0.60)))
    resolved_grow_cluster_min_samples = (
        int(args.grow_cluster_min_samples)
        if args.grow_cluster_min_samples > 0
        else int(saved_postproc.get("grow_cluster_min_samples", meta.get("grow_cluster_min_samples", 6)))
    )
    resolved_seed_min_points = (
        int(args.seed_min_points)
        if args.seed_min_points > 0
        else int(saved_postproc.get("seed_min_points", meta.get("seed_min_points", 24)))
    )
    resolved_support_radius = (
        float(args.support_radius)
        if args.support_radius > 0
        else float(saved_postproc.get("support_radius", postproc_priors.get("support_radius", 0.90)))
    )
    resolved_min_cluster_score = (
        float(args.min_cluster_score)
        if args.min_cluster_score > 0
        else float(saved_postproc.get("min_cluster_score", meta.get("min_cluster_score", 0.12)))
    )
    resolved_keep_topk = int(args.keep_topk) if args.keep_topk > 0 else int(saved_postproc.get("keep_topk", postproc_priors.get("keep_topk", 3)))
    resolved_merge_small_cluster_points = (
        float(args.merge_small_cluster_points)
        if args.merge_small_cluster_points > 0
        else float(saved_postproc.get("merge_small_cluster_points", postproc_priors.get("merge_small_cluster_points", 300.0)))
    )
    resolved_merge_xy_gap = (
        float(args.merge_xy_gap) if args.merge_xy_gap > 0 else float(saved_postproc.get("merge_xy_gap", postproc_priors.get("merge_xy_gap", 1.6)))
    )
    resolved_merge_z_gap = (
        float(args.merge_z_gap) if args.merge_z_gap > 0 else float(saved_postproc.get("merge_z_gap", postproc_priors.get("merge_z_gap", 2.8)))
    )
    resolved_seed_threshold = float(args.seed_threshold) if args.seed_threshold > 0 else float(saved_postproc.get("seed_threshold", -1.0))
    resolved_grow_threshold = float(args.grow_threshold) if args.grow_threshold > 0 else float(saved_postproc.get("grow_threshold", -1.0))
    resolved_support_threshold = float(args.support_threshold) if args.support_threshold > 0 else float(saved_postproc.get("support_threshold", -1.0))

    for fp in tqdm(files, desc="Predicting"):
        xyz_orig, feats_raw_orig, gt_inst = read_csv_points(fp, input_dim=args.input_dim, require_label=False)
        xyz, feats_raw, _, prep_meta = prepare_scene_points(
            xyz_orig,
            feats_raw_orig,
            inst=gt_inst,
            coord_unit_scale=coord_unit_scale,
            auto_unit_normalize=auto_unit_normalize,
            auto_mm_threshold=auto_mm_threshold,
            pre_downsample_voxel=pre_downsample_voxel,
        )

        xyz_scene, point_inputs, norm_meta = build_point_inputs(
            xyz,
            feats_raw,
            input_dim=args.input_dim,
            normalize_mode=norm_mode,
            scale_quantile=scale_quantile,
            scale_clip_min=scale_clip_min,
            scale_clip_max=scale_clip_max,
            abs_coord_scale=abs_coord_scale,
            use_abs_coords=use_abs_coords,
        )
        out_path = os.path.join(args.output_dir, os.path.basename(fp))

        scene = {
            "name": os.path.basename(fp),
            "xyz": xyz.astype(np.float32),
            "xyz_scene": xyz_scene.astype(np.float32),
            "point_inputs": point_inputs.astype(np.float32),
            "inst": gt_inst.astype(np.int64) if gt_inst is not None else None,
            "scene_scale": float(norm_meta.get("scale", 1.0)),
            "extra_dim": int(norm_meta.get("extra_dim", max(0, args.input_dim - 3))),
        }

        if xyz.shape[0] == 0:
            out_arr = np.zeros((xyz_orig.shape[0], 4), dtype=np.float32)
            out_arr[:, :3] = xyz_orig.astype(np.float32)
            out_arr[:, 3] = 0
            np.savetxt(out_path, out_arr, delimiter=",", fmt=save_fmt)
            if gt_inst is not None:
                add_eval_counts(
                    overall_counts,
                    evaluate_pred_vs_gt(np.zeros_like(gt_inst, dtype=np.int64), gt_inst, xyz_raw=xyz_orig),
                )
                labeled_scene_count += 1
            continue

        fg_prob, center_prob, score_prob, vote_xyz, xyz_kept, _, _, _ = infer_scene_prob_chunked(
            model,
            device,
            scene,
            voxel_size=voxel_size,
            max_points_per_voxel=max_points_per_voxel,
            max_voxels=int(args.max_voxels),
            feature_mode=feature_mode,
            voxel_feature_mode=voxel_feature_mode,
            center_prob_mix=center_prob_mix,
            center_prob_power=center_prob_power,
            vote_max_offset_norm=vote_max_offset_norm,
            chunk_max_points=infer_chunk_max_points,
            chunk_min_points=infer_chunk_min_points,
            chunk_overlap=infer_chunk_overlap,
            chunk_max_span_x=infer_chunk_max_span_x,
            chunk_max_span_y=infer_chunk_max_span_y,
            chunk_max_span_z=infer_chunk_max_span_z,
        )

        pred_inst_kept, pred_info = predict_scene_labels(
            xyz_raw=xyz_kept,
            fg_prob=fg_prob,
            base_thr=base_thr,
            postproc_priors=postproc_priors,
            seed_prob=score_prob,
            score_prob=score_prob,
            vote_xyz=vote_xyz,
            output_mode=resolved_output_mode,
            adaptive_threshold=adaptive_threshold,
            binary_cluster_eps=resolved_binary_cluster_eps,
            binary_cluster_min_samples=resolved_binary_cluster_min_samples,
            binary_min_component_points=resolved_binary_min_component_points,
            binary_max_component_points=resolved_binary_max_component_points,
            seed_threshold=resolved_seed_threshold,
            grow_threshold=resolved_grow_threshold,
            support_threshold=resolved_support_threshold,
            cluster_eps=resolved_cluster_eps,
            grow_cluster_min_samples=resolved_grow_cluster_min_samples,
            seed_min_points=resolved_seed_min_points,
            support_radius=resolved_support_radius,
            min_cluster_score=resolved_min_cluster_score,
            keep_topk=resolved_keep_topk,
            fallback_keep_best=bool(args.fallback_keep_best),
            merge_small_cluster_points=resolved_merge_small_cluster_points,
            merge_xy_gap=resolved_merge_xy_gap,
            merge_z_gap=resolved_merge_z_gap,
            merge_xy_overlap=args.merge_xy_overlap,
        )

        logger.info(
            f"{os.path.basename(fp)} | mode={pred_info.get('output_mode', resolved_output_mode)} | "
            f"scene_thr={pred_info.get('scene_thr', base_thr):.3f} | fg_ratio={(pred_inst_kept > 0).mean():.4f} | "
            f"fg_mean={float(fg_prob.mean()):.4f} | center_mean={float(center_prob.mean()):.4f} | "
            f"instances={pred_info.get('num_instances', int(pred_inst_kept.max()))} | fg_points={pred_info.get('num_fg_points', int((pred_inst_kept > 0).sum()))}"
        )

        pred_inst_orig = pred_inst_kept[prep_meta["inverse_map"]]

        out_arr = np.zeros((xyz_orig.shape[0], 4), dtype=np.float32)
        out_arr[:, :3] = xyz_orig.astype(np.float32)
        out_arr[:, 3] = pred_inst_orig.astype(np.float32)
        np.savetxt(out_path, out_arr, delimiter=",", fmt=save_fmt)

        if gt_inst is not None:
            add_eval_counts(overall_counts, evaluate_pred_vs_gt(pred_inst_orig, gt_inst, xyz_raw=xyz_orig))
            labeled_scene_count += 1

    if labeled_scene_count > 0:
        summary = summarize_eval_counts(overall_counts)
        logger.info(
            f"Labeled prediction summary | scenes={labeled_scene_count} | score={summary['score']:.4f} | "
            f"obj_rec={summary['obj_rec']:.4f} | obj_prec={summary['obj_prec']:.4f} | obj_f1={summary['obj_f1']:.4f} | "
            f"obj_cover={summary['obj_cover_mean']:.4f} | pred_purity={summary['pred_purity_mean']:.4f} | "
            f"bbox_score={summary['bbox_score']:.4f} | bbox_rel_mean={summary['bbox_rel_mean']:.4f} | bbox_rel_max={summary['bbox_rel_max']:.4f} | "
            f"fg_f1={summary['fg_f1']:.4f} | fg_iou={summary['fg_iou']:.4f} | fg_prec={summary['fg_prec']:.4f} | fg_rec={summary['fg_rec']:.4f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Sparse-scene FG-first target recognition / segmentation")

    parser.add_argument("mode", choices=["train", "predict"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    _script_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--input_dim", type=int, default=6)
    parser.add_argument("--data_dir", type=str, default=os.path.join(_script_dir, "data", "train"))
    parser.add_argument("--val_dir", type=str, default=os.path.join(_script_dir, "data", "test"))
    parser.add_argument("--extra_train_dir", type=str, default=None)
    parser.add_argument("--input_dir", type=str, default=os.path.join(_script_dir, "data", "test"))
    parser.add_argument("--output_dir", type=str, default=os.path.join(_script_dir, "predictions"))
    parser.add_argument("--train_split", type=float, default=0.9)

    parser.add_argument("--normalize_mode", type=str, default="center", choices=["none", "center", "center_scale"])
    parser.add_argument("--scale_quantile", type=float, default=95.0)
    parser.add_argument("--scale_clip_min", type=float, default=0.0)
    parser.add_argument("--scale_clip_max", type=float, default=16.0)
    parser.add_argument("--abs_coord_scale", type=float, default=100.0)
    parser.add_argument("--use_abs_coords", type=int, default=0)
    parser.add_argument("--coord_unit_scale", type=float, default=0.0)
    parser.add_argument("--auto_unit_normalize", type=int, default=1)
    parser.add_argument("--auto_mm_threshold", type=float, default=200.0)
    parser.add_argument("--pre_downsample_voxel", type=float, default=0.10)

    parser.add_argument("--voxel_size", type=float, default=0.02)
    parser.add_argument("--max_points_per_voxel", type=int, default=50)
    parser.add_argument("--max_voxels", type=int, default=0)
    parser.add_argument("--feature_mode", type=str, default="safe", choices=["safe", "legacy"])
    parser.add_argument("--voxel_feature_mode", type=str, default="extra_only", choices=["extra_only", "scene_xyz_extra", "point_inputs"])

    parser.add_argument("--samples_per_scene", type=int, default=8)
    parser.add_argument("--crop_half_x", type=float, default=8.0)
    parser.add_argument("--crop_half_y", type=float, default=8.0)
    parser.add_argument("--crop_half_z", type=float, default=8.0)
    parser.add_argument("--positive_fraction", type=float, default=0.45)
    parser.add_argument("--hard_negative_fraction", type=float, default=0.35)
    parser.add_argument("--hard_neg_shell_min", type=float, default=5.0)
    parser.add_argument("--hard_neg_shell_max", type=float, default=16.0)
    parser.add_argument("--hard_neg_max_fg_points", type=int, default=24)
    parser.add_argument("--positive_context_scale_min", type=float, default=1.20)
    parser.add_argument("--positive_context_scale_max", type=float, default=1.80)
    parser.add_argument("--positive_safe_margin", type=float, default=0.35)
    parser.add_argument("--positive_jitter_frac", type=float, default=0.45)
    parser.add_argument("--positive_min_coverage", type=float, default=0.95)
    parser.add_argument("--hard_neg_near_obj_prob", type=float, default=0.65)
    parser.add_argument("--hard_neg_context_margin", type=float, default=2.80)
    parser.add_argument("--hard_neg_exclude_margin", type=float, default=0.70)
    parser.add_argument("--max_points", type=int, default=24000)
    parser.add_argument("--aug_jitter_xyz_std", type=float, default=0.010)
    parser.add_argument("--aug_dropout", type=float, default=0.02)
    parser.add_argument("--aug_rotate_z_deg", type=float, default=180.0)
    parser.add_argument("--aug_flip_x_prob", type=float, default=0.25)
    parser.add_argument("--aug_flip_y_prob", type=float, default=0.25)
    parser.add_argument("--aug_bg_dropout_prob", type=float, default=0.10)
    parser.add_argument("--aug_bg_dropout_min", type=float, default=0.02)
    parser.add_argument("--aug_bg_dropout_max", type=float, default=0.06)
    parser.add_argument("--aug_normal_noise_std", type=float, default=0.005)

    parser.add_argument("--init_channels", type=int, default=48)
    parser.add_argument("--norm", type=str, default="gn", choices=["gn", "bn"])
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--point_refine_dim", type=int, default=128)
    parser.add_argument("--fg_prior", type=float, default=0.01)
    parser.add_argument("--center_prior", type=float, default=0.02)
    parser.add_argument("--center_prob_mix", type=float, default=0.15)
    parser.add_argument("--center_prob_power", type=float, default=0.50)
    parser.add_argument("--vote_max_offset_norm", type=float, default=4.0)
    parser.add_argument("--enable_deep_stage", type=int, default=1)
    parser.add_argument("--context_stage_max_voxels", type=int, default=280000)
    parser.add_argument("--deep_stage_max_voxels", type=int, default=320000)
    parser.add_argument("--infer_chunk_max_points", type=int, default=260000)
    parser.add_argument("--infer_chunk_min_points", type=int, default=32000)
    parser.add_argument("--infer_chunk_overlap", type=float, default=2.0)
    parser.add_argument("--infer_chunk_max_span_x", type=float, default=192.0)
    parser.add_argument("--infer_chunk_max_span_y", type=float, default=192.0)
    parser.add_argument("--infer_chunk_max_span_z", type=float, default=96.0)

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min_epochs_before_stop", type=int, default=120)
    parser.add_argument("--val_interval", type=int, default=5)
    parser.add_argument("--val_metric", type=str, default="score", choices=["score", "fg_iou"])
    parser.add_argument("--val_output_modes", type=str, default="binary,binary_cc,instance")
    parser.add_argument("--val_eval_thr_limit", type=int, default=9)
    parser.add_argument("--amp", type=int, default=1)
    parser.add_argument("--disable_tqdm", type=int, default=0)
    parser.add_argument("--model_save_path", type=str, default=os.path.join(_script_dir, "tapa_model.pth"))
    parser.add_argument("--resume_path", type=str, default=None)
    parser.add_argument("--pretrained_weights", type=str, default=None)
    parser.add_argument("--calibrate_postproc", type=int, default=0)
    parser.add_argument("--calibration_scene_limit", type=int, default=8)
    parser.add_argument("--freeze_backbone", type=int, default=0)
    parser.add_argument("--freeze_voxel_mlp", type=int, default=0)
    parser.add_argument("--freeze_point_refine", type=int, default=0)
    parser.add_argument("--freeze_context_fuse", type=int, default=0)
    parser.add_argument("--freeze_fg_head", type=int, default=0)
    parser.add_argument("--freeze_center_head", type=int, default=0)
    parser.add_argument("--freeze_offset_head", type=int, default=0)

    parser.add_argument("--w_focal", type=float, default=1.0)
    parser.add_argument("--w_tversky", type=float, default=1.0)
    parser.add_argument("--w_center", type=float, default=0.25)
    parser.add_argument("--w_offset", type=float, default=0.70)
    parser.add_argument("--w_size", type=float, default=0.25)
    parser.add_argument("--focal_alpha", type=float, default=0.55)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--hnm_ratio", type=float, default=3.0)
    parser.add_argument("--hnm_min_neg", type=int, default=1024)
    parser.add_argument("--hnm_max_neg", type=int, default=16384)
    parser.add_argument("--tversky_alpha", type=float, default=0.65)
    parser.add_argument("--tversky_beta", type=float, default=0.35)
    parser.add_argument("--center_bg_weight", type=float, default=0.20)
    parser.add_argument("--center_fg_weight", type=float, default=1.00)
    parser.add_argument("--center_peak_weight", type=float, default=1.50)
    parser.add_argument("--offset_weight_power", type=float, default=1.00)

    parser.add_argument("--val_thr_min", type=float, default=0.10)
    parser.add_argument("--val_thr_max", type=float, default=0.80)
    parser.add_argument("--val_thr_num", type=int, default=21)
    parser.add_argument("--default_fg_threshold", type=float, default=0.30)

    parser.add_argument("--model_path", type=str, default=os.path.join(_script_dir, "tapa_model_best.pth"))
    parser.add_argument("--infer_amp", type=int, default=1)
    parser.add_argument("--fg_threshold", type=float, default=-1.0)
    parser.add_argument("--output_mode", type=str, default="auto", choices=["auto", "instance", "binary", "binary_cc"])
    parser.add_argument("--adaptive_threshold", type=int, default=-1)
    parser.add_argument("--binary_cluster_eps", type=float, default=-1.0)
    parser.add_argument("--binary_cluster_min_samples", type=int, default=-1)
    parser.add_argument("--binary_min_component_points", type=int, default=-1)
    parser.add_argument("--binary_max_component_points", type=int, default=-1)
    parser.add_argument("--seed_threshold", type=float, default=-1.0)
    parser.add_argument("--grow_threshold", type=float, default=-1.0)
    parser.add_argument("--support_threshold", type=float, default=-1.0)
    parser.add_argument("--cluster_eps", type=float, default=0.60)
    parser.add_argument("--grow_cluster_min_samples", type=int, default=6)
    parser.add_argument("--seed_min_points", type=int, default=24)
    parser.add_argument("--support_radius", type=float, default=-1.0)
    parser.add_argument("--merge_small_cluster_points", type=float, default=-1.0)
    parser.add_argument("--merge_xy_gap", type=float, default=-1.0)
    parser.add_argument("--merge_z_gap", type=float, default=-1.0)
    parser.add_argument("--merge_xy_overlap", type=float, default=0.22)
    parser.add_argument("--min_cluster_score", type=float, default=0.12)
    parser.add_argument("--keep_topk", type=int, default=3)
    parser.add_argument("--fallback_keep_best", type=int, default=1)

    args = parser.parse_args()

    if args.mode == "train":
        if not args.data_dir or not os.path.isdir(args.data_dir):
            print(f"Error: --data_dir not found or not a directory: {args.data_dir}")
            return
        train(args)
    else:
        if not args.input_dir or not args.output_dir:
            print("Error: --input_dir and --output_dir required for prediction")
            return
        predict(args)


if __name__ == "__main__":
    main()


    """
    python tap.py predict --input_dir "D:\company\0125\门机\datarec\grabarti_intsega" --output_dir "D:\company\0125\门机\datarec\grabarti_intsegb" --model_path fginst_sparsecrop_v1_best.pth
    
    python tap.py predict   --input_dir "D:\company\0125\门机\datarec\grabarti_intsega"   --output_dir "D:\company\0125\门机\datarec\grabarti_intsegb"   --model_path fginst_sparsecrop_v1_best.pth   --seed_threshold 0.55   --grow_threshold 0.20  --support_threshold 0.08 --cluster_eps 0.60 --grow_cluster_min_samples 6 --seed_min_points 20 --support_radius 1.00 --merge_xy_gap 1.80 --merge_z_gap 3.20 --merge_xy_overlap 0.20 --keep_topk 3
    python tap.py predict   --input_dir "D:\company\0125\门机\datarec\tempdatanxyz"   --output_dir "D:\company\0125\门机\datarec\tempdatanxyzb"   --model_path fginst_sparsecrop_v1_best.pth   --seed_threshold 0.55   --grow_threshold 0.20  --support_threshold 0.08 --cluster_eps 0.60 --grow_cluster_min_samples 6 --seed_min_points 20 --support_radius 1.00 --merge_xy_gap 1.80 --merge_z_gap 3.20 --merge_xy_overlap 0.20 --keep_topk 3

    D:\company\0125\门机\datarec\tempdatanxyz
    python tap.py predict --input_dir "D:\company\0125\门机\datarec\grabarti_intsega" --output_dir "D:\company\0125\门机\datarec\tempdatanxyzb" --model_path fginst_sparsecrop_v1_best.pth
    """
