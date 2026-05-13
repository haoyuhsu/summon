"""Build ContactFormer training data from humoto pickles.

Per .pkl in humoto_data/all/, produces 4 .npy files in {output_dir}/:
    vertices/<id>_verts.npy         (T, 655, 3) world-space body verts (Z-up)
    vertices_can/<id>_verts_can.npy (T, 655, 3) pelvis-centered, yaw-normalized
    semantics/<id>_cfs.npy          (T, 655) int class IDs in [0, 72]
    contacts/<id>_cf.npy            (T, 655) float32 binary contact mask

Input motion is Y-up SMPL-X; we convert to Z-up (matching SUMMON conventions)
before SMPL-X forward + contact-label generation. Contact labels come from
per-object unsigned-distance SDFs cached on disk and reused across all samples.

Class taxonomy: 0 = background (incl. floor/ground); 1..72 = sorted humoto
object directory names. Foot-on-floor frames stay class 0 by design.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from human_body_prior.body_model.body_model import BodyModel
from scipy.spatial import cKDTree
from tqdm import tqdm

# `data_utils` imports `open3d`, `pandas`, `eulerangles`, `smplx` at module
# scope, but none of the functions we need (`get_graph_params`, `ds_us`,
# `normalize_orientation`) actually use them. Stub the modules so the import
# succeeds in environments where these aren't installed (e.g. imu-humans env
# is missing pandas; open3d is broken).
for _mod in ('open3d', 'pandas', 'eulerangles', 'smplx'):
    sys.modules.setdefault(_mod, types.ModuleType(_mod))

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, '/home/haoyuyh3/Documents/maxhsu/imu-humans/humoto')

import data_utils as du  # noqa: E402
from humoto_utils import (  # noqa: E402
    HUMOTO_DATA_DIR,
    HUMOTO_OBJECTS_DIR,
    SMPLX_MODEL_PATH,
    pose7_to_rotmat_transl,
    yup_to_zup_object_pose,
    yup_to_zup_smplx,
)


# Floor/ground are explicitly NOT trainable classes. Foot-on-floor frames
# remain class 0 (background). `floor_lamp` is unrelated and stays a class.
SKIP_NAMES = {'ground', 'floor'}


def build_class_taxonomy(objects_dir: Path) -> tuple[list[str], dict[str, int]]:
    """Enumerate object directories. Returns (class_names sorted, name→id), with id starting at 1."""
    names = sorted(
        d for d in os.listdir(objects_dir)
        if (objects_dir / d).is_dir() and not d.startswith('.') and d not in SKIP_NAMES
    )
    name_to_id = {n: i + 1 for i, n in enumerate(names)}
    return names, name_to_id


def build_object_sdf_cache(
    class_names: list[str],
    objects_dir: Path,
    cache_dir: Path,
    grid_dim: int,
    pad_m: float,
    surface_samples: int,
    rebuild: bool,
) -> None:
    """Compute (D, D, D) unsigned-distance SDFs in each object's local frame, saved to disk.

    For each grid cell, the value is approximate distance to the nearest mesh surface,
    computed by uniformly sampling `surface_samples` points on the mesh and running a
    nearest-neighbour query via cKDTree. Approximation error ~ sqrt(area / n_samples),
    e.g. <5 mm for ~1 m² area at 10k samples — well below the 5 cm contact threshold.

    Grid axes are (x, y, z) in that order, matching SUMMON's read_sdf convention
    (axis 0 = x, axis 1 = y, axis 2 = z).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    todo = [n for n in class_names if rebuild or not (cache_dir / f'{n}.npz').exists()]
    if not todo:
        return
    print(f"Building SDF cache for {len(todo)}/{len(class_names)} object classes...")
    for name in tqdm(todo, desc='SDF cache'):
        mesh_path = objects_dir / name / f'{name}.obj'
        if not mesh_path.exists():
            print(f'  [skip] missing mesh: {mesh_path}')
            continue
        mesh = trimesh.load_mesh(str(mesh_path), process=False)
        bmin = mesh.bounds[0].astype(np.float32) - pad_m
        bmax = mesh.bounds[1].astype(np.float32) + pad_m
        xs = np.linspace(bmin[0], bmax[0], grid_dim, dtype=np.float32)
        ys = np.linspace(bmin[1], bmax[1], grid_dim, dtype=np.float32)
        zs = np.linspace(bmin[2], bmax[2], grid_dim, dtype=np.float32)
        # `indexing='ij'` with order (xs, ys, zs) → grid axes (x, y, z).
        gx, gy, gz = np.meshgrid(xs, ys, zs, indexing='ij')
        pts = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)

        # Sample mesh surface uniformly, then KDTree-query each grid point.
        n_samples = max(surface_samples, len(mesh.vertices))
        try:
            samples, _ = trimesh.sample.sample_surface(mesh, n_samples)
        except Exception:
            # Fallback: use raw vertices if surface sampling fails (e.g. degenerate mesh).
            samples = mesh.vertices
        tree = cKDTree(np.asarray(samples))
        dists, _ = tree.query(pts, k=1)
        sdf = dists.reshape(grid_dim, grid_dim, grid_dim).astype(np.float32)
        np.savez(cache_dir / f'{name}.npz', sdf=sdf, grid_min=bmin, grid_max=bmax)


def load_object_sdfs(
    class_names: list[str],
    cache_dir: Path,
    device: torch.device,
) -> dict[str, dict]:
    """Load all cached SDFs to device. Returns {name: {sdf:(D,D,D), grid_min, grid_max}}."""
    cache: dict[str, dict] = {}
    for name in class_names:
        path = cache_dir / f'{name}.npz'
        if not path.exists():
            continue
        d = np.load(path)
        cache[name] = {
            'sdf': torch.tensor(d['sdf'], dtype=torch.float32, device=device),       # (D,D,D)
            'grid_min': torch.tensor(d['grid_min'], dtype=torch.float32, device=device),
            'grid_max': torch.tensor(d['grid_max'], dtype=torch.float32, device=device),
        }
    return cache


def query_sdf_batched(
    sdf: torch.Tensor,         # (D, D, D)
    grid_min: torch.Tensor,    # (3,)
    grid_max: torch.Tensor,    # (3,)
    points: torch.Tensor,      # (B, N, 3) in object-local frame
) -> torch.Tensor:
    """Trilinear-sample SDF at `points`. Returns (B, N).

    Wraps F.grid_sample to broadcast a single SDF over the batch dim. Mirrors
    the axis convention used by data_utils.read_sdf: grid axis 0=x, 1=y, 2=z;
    grid_sample's (W, H, D) corresponds to (z, y, x), hence the [2,1,0] permute.
    """
    B, N = points.shape[:2]
    D = sdf.shape[0]
    sdf_5d = sdf.view(1, 1, D, D, D).expand(B, -1, -1, -1, -1)            # no copy
    norm = (points - grid_min) / (grid_max - grid_min) * 2.0 - 1.0         # (B, N, 3)
    grid = norm[..., [2, 1, 0]].view(B, N, 1, 1, 3)
    out = F.grid_sample(sdf_5d, grid, padding_mode='border',
                        mode='bilinear', align_corners=True)               # (B, 1, N, 1, 1)
    return out.view(B, N)


def compute_contact_labels(
    verts_world_655: torch.Tensor,         # (T, 655, 3) Z-up
    objects_z: dict[str, np.ndarray],      # already Y→Z converted, pose7 per object
    sdf_cache: dict[str, dict],
    name_to_id: dict[str, int],
    contact_thresh_m: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-frame, per-vertex contact label and binary mask via per-object SDF lookup.
    Returns (cfs (T,655) int64, cf (T,655) float32). Class assignment = closest object
    if its distance < contact_thresh_m, else 0 (background).
    """
    T, N = verts_world_655.shape[:2]
    all_d: list[torch.Tensor] = []
    all_cls: list[int] = []
    for name, pose7_z in objects_z.items():
        if name in SKIP_NAMES or name not in name_to_id or name not in sdf_cache:
            continue
        R_obj, t_obj = pose7_to_rotmat_transl(np.asarray(pose7_z, dtype=np.float32))
        R_obj_t = torch.tensor(R_obj, dtype=torch.float32, device=device)   # (3, 3)
        t_obj_t = torch.tensor(t_obj, dtype=torch.float32, device=device)   # (3,)
        # world → object-local: v_local = R^T (v - t) = (v - t) @ R
        v_local = (verts_world_655 - t_obj_t) @ R_obj_t                     # (T, 655, 3)
        info = sdf_cache[name]
        d = query_sdf_batched(info['sdf'], info['grid_min'], info['grid_max'], v_local)
        all_d.append(d)
        all_cls.append(name_to_id[name])

    if not all_d:
        cfs = torch.zeros((T, N), dtype=torch.long, device=device)
    else:
        dists = torch.stack(all_d, dim=0)                                   # (K, T, 655)
        cls_t = torch.tensor(all_cls, dtype=torch.long, device=device)
        argmin_k = dists.argmin(dim=0)                                      # (T, 655)
        min_d = dists.min(dim=0).values                                     # (T, 655)
        cfs = torch.where(min_d < contact_thresh_m,
                          cls_t[argmin_k], torch.zeros_like(argmin_k))
    cf = (cfs > 0).to(torch.float32)
    return cfs, cf


def process_one_pkl(
    pkl_path: Path,
    body_model: BodyModel,
    rest_pelvis: np.ndarray,
    ds_fn1, ds_fn2,
    associated_joints: torch.Tensor,
    sdf_cache: dict,
    name_to_id: dict[str, int],
    output_dir: Path,
    contact_thresh_m: float,
    device: torch.device,
    verbose: bool = False,
) -> np.ndarray:
    """Full per-pkl pipeline: load → Y→Z → SMPL-X → downsample → contact labels → save.
    Returns a (num_classes+1,) int64 histogram of class IDs in `cfs` for this pkl,
    so the caller can aggregate dataset-wide contact statistics.
    """
    seq_id = pkl_path.stem  # e.g. '0000001'

    with open(pkl_path, 'rb') as f:
        sample = pickle.load(f)

    motion = sample['motion_smpl'].astype(np.float32)        # (T, 75)
    T = motion.shape[0]
    go_y = motion[:, 0:3]
    bp = motion[:, 3:66]
    tr_y = motion[:, 72:75]

    # Y → Z conversion
    go_z, tr_z = yup_to_zup_smplx(go_y, tr_y, rest_pelvis)

    # Single batched SMPL-X forward
    with torch.no_grad():
        out = body_model(
            pose_body=torch.tensor(bp, device=device),
            root_orient=torch.tensor(go_z, device=device),
            trans=torch.tensor(tr_z, device=device),
            betas=torch.zeros((T, 10), dtype=torch.float32, device=device),
        )
    verts_world = out.v                                       # (T, 10475, 3)
    pelvis_world = out.Jtr[:, 0]                              # (T, 3)

    # Canonical body = world body minus per-frame pelvis (no second forward pass needed).
    verts_can_full = verts_world - pelvis_world.unsqueeze(1)  # (T, 10475, 3)

    # Downsample 10475 → 2619 → 655 for both
    verts_world_655 = ds_fn2(ds_fn1(verts_world))             # (T, 655, 3)
    verts_can_655 = ds_fn2(ds_fn1(verts_can_full))            # (T, 655, 3)

    # First-frame yaw normalization on canonical (assumes Z-up; we have it post-conversion).
    verts_can_norm = du.normalize_orientation(verts_can_655, associated_joints, device)

    # Convert object poses Y → Z; skip 'ground' / 'floor' / non-pose entries.
    objects_z: dict[str, np.ndarray] = {}
    for name, p7 in sample.get('objects', {}).items():
        if name in SKIP_NAMES:
            continue
        p7 = np.asarray(p7, dtype=np.float32)
        if p7.shape != (7,):
            continue
        objects_z[name] = yup_to_zup_object_pose(p7)

    # Contact labels
    cfs, cf = compute_contact_labels(
        verts_world_655, objects_z, sdf_cache, name_to_id,
        contact_thresh_m, device,
    )

    cls_hist = torch.bincount(cfs.flatten(),
                              minlength=len(name_to_id) + 1).cpu().numpy().astype(np.int64)

    if verbose:
        active = [(i, int(c)) for i, c in enumerate(cls_hist) if c > 0 and i > 0]
        n_contact_frames = (cf.sum(dim=1) > 0).sum().item()
        text = sample.get('text')
        print(f"  {seq_id}: T={T}, contact frames={n_contact_frames}/{T}, "
              f"active classes={active}, text={text}")

    # Save in the layout ProxDataset_ds expects.
    np.save(output_dir / 'vertices' / f'{seq_id}_verts.npy',
            verts_world_655.detach().cpu().numpy())
    np.save(output_dir / 'vertices_can' / f'{seq_id}_verts_can.npy',
            verts_can_norm.detach().cpu().numpy())
    np.save(output_dir / 'semantics' / f'{seq_id}_cfs.npy',
            cfs.detach().cpu().numpy().astype(np.int32))
    np.save(output_dir / 'contacts' / f'{seq_id}_cf.npy',
            cf.detach().cpu().numpy().astype(np.float32))

    return cls_hist


def main() -> None:
    """Build humoto contact dataset for ContactFormer training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_dir', type=str, default=str(HUMOTO_DATA_DIR))
    parser.add_argument('--objects_dir', type=str, default=str(HUMOTO_OBJECTS_DIR))
    parser.add_argument('--output_dir', type=str,
                        default='/home/haoyuyh3/Documents/maxhsu/imu-humans/summon/data_custom/humoto')
    parser.add_argument('--smplx_model_path', type=str, default=SMPLX_MODEL_PATH)
    parser.add_argument('--mesh_ds_us_dir', type=str,
                        default='/home/haoyuyh3/Documents/maxhsu/imu-humans/summon/mesh_ds')
    parser.add_argument('--sdf_grid_dim', type=int, default=64)
    parser.add_argument('--sdf_pad_m', type=float, default=0.10)
    parser.add_argument('--sdf_surface_samples', type=int, default=10000,
                        help='# of points sampled uniformly on each mesh for KDTree-based distance.')
    parser.add_argument('--contact_thresh_m', type=float, default=0.05)
    parser.add_argument('--max_samples', type=int, default=-1)
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=-1)
    parser.add_argument('--rebuild_sdf_cache', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    objects_dir = Path(args.objects_dir)
    output_dir = Path(args.output_dir)
    mesh_ds_dir = Path(args.mesh_ds_us_dir)
    cache_dir = output_dir / 'object_sdf_cache'

    for sub in ('vertices', 'vertices_can', 'semantics', 'contacts'):
        (output_dir / sub).mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 1. Class taxonomy (excludes ground/floor) — persisted once.
    class_names, name_to_id = build_class_taxonomy(objects_dir)
    print(f"Class taxonomy: {len(class_names)} object classes (+ class 0 = background)")
    taxonomy_path = SCRIPT_DIR / 'humoto_class_taxonomy.json'
    with open(taxonomy_path, 'w') as f:
        json.dump({'background': 0, **name_to_id}, f, indent=2)
    print(f"Saved taxonomy: {taxonomy_path}")

    # 2. Per-object SDF cache (one-time; reused across all 735 samples).
    build_object_sdf_cache(
        class_names, objects_dir, cache_dir,
        args.sdf_grid_dim, args.sdf_pad_m, args.sdf_surface_samples,
        args.rebuild_sdf_cache,
    )
    sdf_cache = load_object_sdfs(class_names, cache_dir, device)
    print(f"Loaded SDFs for {len(sdf_cache)}/{len(class_names)} classes")

    # 3. SMPL-X model + rest-pose pelvis.
    print("Loading SMPL-X model...")
    body_model = BodyModel(
        bm_fname=args.smplx_model_path, num_betas=10, model_type='smplx',
    ).to(device)
    rest_pelvis = body_model().Jtr[0, 0].detach().cpu().numpy().astype(np.float32)

    # 4. Spiral-Conv downsampling matrices for 10475 → 2619 → 655.
    _, _, D1 = du.get_graph_params(str(mesh_ds_dir), device, layer=1)
    ds1 = du.ds_us(D1).to(device)
    _, _, D2 = du.get_graph_params(str(mesh_ds_dir), device, layer=2)
    ds2 = du.ds_us(D2).to(device)

    # 5. Associated joints for normalize_orientation (downsampled SMPL-X skinning weights argmax).
    ds_weights = torch.tensor(np.load(SCRIPT_DIR / 'support_files' / 'downsampled_weights.npy'))
    associated_joints = torch.argmax(ds_weights, dim=1)

    # 6. Iterate pkls.
    pkl_files = sorted(p for p in os.listdir(data_dir) if p.endswith('.pkl'))
    end = len(pkl_files) if args.end_idx < 0 else args.end_idx
    pkl_files = pkl_files[args.start_idx:end]
    if args.max_samples > 0:
        pkl_files = pkl_files[:args.max_samples]
    print(f"Processing {len(pkl_files)} pkls → {output_dir}")

    total_hist = np.zeros(len(name_to_id) + 1, dtype=np.int64)  # class-id → vertex-frame count
    n_samples_with_contact = 0
    for f in tqdm(pkl_files, desc='Pkls', dynamic_ncols=True):
        hist = process_one_pkl(
            data_dir / f, body_model, rest_pelvis, ds1, ds2,
            associated_joints, sdf_cache, name_to_id, output_dir,
            args.contact_thresh_m, device, verbose=args.verbose,
        )
        total_hist += hist
        if hist[1:].sum() > 0:
            n_samples_with_contact += 1

    # Summary: dataset-wide contact-label statistics.
    id_to_name = {i: n for n, i in name_to_id.items()}
    total_vf = int(total_hist.sum())
    total_contact_vf = int(total_hist[1:].sum())
    print()
    print("=" * 60)
    print(f"Contact-label summary across {len(pkl_files)} pkls")
    print("=" * 60)
    print(f"  total vertex-frames:     {total_vf:,}")
    print(f"  contact vertex-frames:   {total_contact_vf:,} "
          f"({100.0 * total_contact_vf / max(total_vf, 1):.2f}%)")
    print(f"  samples with ≥1 contact: {n_samples_with_contact}/{len(pkl_files)}")
    print()
    print(f"  Per-class vertex-frame counts (descending), nonzero only:")
    nonzero = [(i, int(c)) for i, c in enumerate(total_hist) if c > 0]
    nonzero.sort(key=lambda x: -x[1])
    for cid, count in nonzero:
        name = 'background' if cid == 0 else id_to_name.get(cid, f'?{cid}')
        pct = 100.0 * count / max(total_vf, 1)
        print(f"    {cid:3d} {name:30s} {count:>12,}  ({pct:6.3f}%)")

    # Persist the histogram alongside the taxonomy for downstream analysis.
    stats_path = output_dir / 'contact_label_stats.json'
    stats = {
        'num_pkls_processed': len(pkl_files),
        'samples_with_contact': n_samples_with_contact,
        'total_vertex_frames': total_vf,
        'total_contact_vertex_frames': total_contact_vf,
        'per_class_counts': {
            ('background' if cid == 0 else id_to_name.get(cid, f'?{cid}')): int(c)
            for cid, c in enumerate(total_hist)
        },
    }
    with open(stats_path, 'w') as fh:
        json.dump(stats, fh, indent=2)
    print()
    print(f"  Saved stats: {stats_path}")


if __name__ == '__main__':
    main()
