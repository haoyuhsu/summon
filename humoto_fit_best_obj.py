"""Fit humoto object meshes to predicted contact labels for a single sequence.

Adapted from fit_best_obj.py. Differences:

- No 8→40 class remap. Humoto class IDs (1..72 per humoto_class_taxonomy.json) are used directly.
- The candidate library is a single canonical mesh per class: humoto_objects_0805/<class>/<class>.obj.
  The inner candidate-iteration loop collapses to one iteration.
- Per-class DBSCAN `eps` is built programmatically from each class's mesh bbox diagonal,
  cached at humoto_classes_eps.json on first use.
- Per-class fitting weights default to (10, 1) for grid-search and Adam respectively when not
  explicitly set in config.params["default"].
- The human SDF is built via KDTree on uniformly-sampled surface points (no mesh_to_sdf dep).
- Floor height is min-z of body verts across all frames (humoto taxonomy has no floor class).
- 73-entry color palette derived from matplotlib's tab20 with cycling (cached).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import config  # noqa: E402
from place_obj_opt import contact_loss, grid_search, optimization, penetration_loss  # noqa: E402
from utils import align_obj_to_floor, write_verts_faces_obj  # noqa: E402

try:
    import open3d as o3d
    _HAS_O3D = True
except Exception:
    _HAS_O3D = False


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Voxel-downsample a point cloud. Uses open3d if available, else a numpy fallback."""
    if _HAS_O3D:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        return np.asarray(pcd.points)
    keys = np.floor(points / voxel_size).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return points[idx]


def dbscan_cluster(points: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """DBSCAN cluster labels for an Nx3 point cloud. Returns (-1=noise, 0..K=clusters)."""
    if _HAS_O3D:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        return np.asarray(pcd.cluster_dbscan(eps=eps, min_points=min_samples, print_progress=False))
    from sklearn.cluster import DBSCAN
    return DBSCAN(eps=eps, min_samples=min_samples).fit(points).labels_


def build_human_sdf_kdtree(
    human_meshes_verts: np.ndarray,   # (T, 655, 3) — body verts across frames
    human_faces: np.ndarray,          # (F, 3)
    grid_dim: int = 128,
    surface_samples: int = 50000,
    pad: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a 3D unsigned-distance field of the merged human mesh via KDTree on sampled points.

    Returns (centroid (3,), extents (3,), sdf (D, D, D)) — same layout as utils.generate_sdf so
    place_obj_opt.compute_signed_distances can consume it (it normalizes via centroid + extents.max()).
    Note: distance is **unsigned**; for the penetration term in place_obj_opt this means we treat
    "near the human" as "penetrating" (since we cannot tell inside vs outside without watertightness).
    Acceptable for object placement against a roughly-cylindrical human surface.
    """
    flat_verts = human_meshes_verts.reshape(-1, 3)
    bmin = flat_verts.min(axis=0) - pad
    bmax = flat_verts.max(axis=0) + pad
    centroid = ((bmin + bmax) / 2).astype(np.float32)
    extents = (bmax - bmin).astype(np.float32)

    # Build a single trimesh from all frames concatenated for surface sampling.
    T, N, _ = human_meshes_verts.shape
    all_vs = []
    all_fs = []
    for t in range(T):
        all_vs.append(human_meshes_verts[t])
        all_fs.append(human_faces + t * N)
    merged = trimesh.Trimesh(
        vertices=np.concatenate(all_vs, axis=0),
        faces=np.concatenate(all_fs, axis=0),
        process=False,
    )
    n_samp = max(surface_samples, len(merged.vertices))
    try:
        samples, _ = trimesh.sample.sample_surface(merged, n_samp)
    except Exception:
        samples = merged.vertices
    tree = cKDTree(np.asarray(samples))

    xs = np.linspace(bmin[0], bmax[0], grid_dim, dtype=np.float32)
    ys = np.linspace(bmin[1], bmax[1], grid_dim, dtype=np.float32)
    zs = np.linspace(bmin[2], bmax[2], grid_dim, dtype=np.float32)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing='ij')
    pts = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)
    dists, _ = tree.query(pts, k=1)
    sdf = dists.reshape(grid_dim, grid_dim, grid_dim).astype(np.float32)
    return centroid, extents, sdf


def build_humoto_classes_eps(
    objects_dir: Path,
    name_to_id: dict[str, int],
    cache_path: Path,
    eps_min: float = 0.2,
    eps_max: float = 1.0,
) -> dict[int, float]:
    """Per-class DBSCAN eps from mesh bbox diagonal: eps = clamp(0.5 * diag(mesh.bounds), 0.2, 1.0)."""
    if cache_path.exists():
        return {int(k): float(v) for k, v in json.load(open(cache_path)).items()}
    classes_eps: dict[int, float] = {}
    for name, cid in name_to_id.items():
        mp = objects_dir / name / f'{name}.obj'
        if not mp.exists():
            continue
        m = trimesh.load_mesh(str(mp), process=False)
        diag = float(np.linalg.norm(m.bounds[1] - m.bounds[0]))
        classes_eps[cid] = float(min(max(0.5 * diag, eps_min), eps_max))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump({str(k): v for k, v in classes_eps.items()}, f, indent=2)
    return classes_eps


def build_humoto_color_palette(num_classes: int, cache_path: Path) -> np.ndarray:
    """Per-class RGB in [0,1] of shape (num_classes, 3). Cycles tab20 for visual distinction."""
    if cache_path.exists():
        return np.asarray(json.load(open(cache_path)), dtype=np.float32)
    import matplotlib
    cmap = matplotlib.colormaps['tab20']
    rgb = np.array([cmap(i % 20)[:3] for i in range(num_classes)], dtype=np.float32)
    rgb[0] = (0.5, 0.5, 0.5)  # background = grey
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump(rgb.tolist(), f, indent=2)
    return rgb


def load_taxonomy(path: Path) -> tuple[dict[str, int], dict[int, str]]:
    """Load humoto class taxonomy JSON. Returns (name_to_id, id_to_name)."""
    raw = json.load(open(path))
    name_to_id = {k: int(v) for k, v in raw.items() if k != 'background'}
    id_to_name = {0: 'background'}
    for n, i in name_to_id.items():
        id_to_name[i] = n
    return name_to_id, id_to_name


def main() -> None:
    """Fit best humoto objects to predicted contact labels for one sequence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence_name', type=str, required=True, help='e.g. 0000585')
    parser.add_argument('--vertices_path', type=str, required=True,
                        help='Path to {seq}_verts.npy from humoto_gen_pred_vertices.py')
    parser.add_argument('--contact_labels_path', type=str, required=True,
                        help='Path to {seq}.npy from predict_contact.py')
    parser.add_argument('--output_dir', type=str, default='fitting_results/humoto_pred')
    parser.add_argument('--input_probability', action='store_true',
                        help='Set if contact_labels_path is softmax (T, 655, K) — argmax internally.')
    parser.add_argument('--objects_dir', type=str,
                        default='/home/haoyuyh3/Documents/maxhsu/imu-humans/humoto/humoto_objects_0805')
    parser.add_argument('--taxonomy_json', type=str,
                        default=str(SCRIPT_DIR / 'contactFormer' / 'humoto_class_taxonomy.json'))
    parser.add_argument('--mesh_ds_dir', type=str, default=str(SCRIPT_DIR / 'mesh_ds'))
    parser.add_argument('--jump_step', type=int, default=8,
                        help='Frame stride used during predict_contact.py.')
    parser.add_argument('--sdf_grid_dim', type=int, default=128)
    args = parser.parse_args()

    sequence_name = args.sequence_name
    output_dir = Path(args.output_dir)
    objects_dir = Path(args.objects_dir)
    taxonomy_path = Path(args.taxonomy_json)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Sequence: {sequence_name}, device: {device}")

    name_to_id, id_to_name = load_taxonomy(taxonomy_path)
    num_classes = len(id_to_name)  # 73 for humoto

    # Per-class DBSCAN eps and color palette (built once, cached on disk).
    eps_cache = SCRIPT_DIR / 'contactFormer' / 'humoto_classes_eps.json'
    color_cache = SCRIPT_DIR / 'contactFormer' / 'humoto_color_palette.json'
    classes_eps = build_humoto_classes_eps(objects_dir, name_to_id, eps_cache)
    color_palette = build_humoto_color_palette(num_classes, color_cache)

    # Fitting hyperparams (reuse SUMMON's defaults, with class-agnostic fallback weights).
    pcd_down_voxel_size = config.voxel_size
    voting_eps = config.voting_eps
    cluster_min_points = config.cluster_min_points
    p = config.params['default']
    grid_search_contact_weight = p['grid_search_contact_weight']
    grid_search_pen_thresh = p['grid_search_pen_thresh']
    lr = p['lr']
    opt_steps = p['opt_steps']
    opt_contact_weight = p['opt_contact_weight']
    opt_pen_thresh = p['opt_pen_thresh']
    # Per-class penalty weights: fall back to a default for unmapped humoto classes.
    from collections import defaultdict
    grid_search_classes_pen_weight = defaultdict(lambda: 10, p['grid_search_classes_pen_weight'])
    opt_classes_pen_weight = defaultdict(lambda: 1, p['opt_classes_pen_weight'])

    # Load body verts (T, 655, 3) and contact labels.
    vertices = np.load(args.vertices_path)
    contact_labels = np.load(args.contact_labels_path)
    if args.input_probability:
        contact_labels = np.argmax(contact_labels, axis=-1)
    contact_labels = contact_labels.squeeze()  # (T_skipped, 655)
    T_skipped = contact_labels.shape[0]

    # Re-align body verts to the predicted-frame stride. predict_contact uses jump_step.
    vertices_down = []
    for f in range(T_skipped):
        idx = f * args.jump_step
        if idx >= vertices.shape[0]:
            break
        vertices_down.append(vertices[idx])
    vertices = np.asarray(vertices_down)               # (T_skipped, 655, 3)
    contact_labels = contact_labels[:vertices.shape[0]]
    print(f"Vertices: {vertices.shape}, contact labels: {contact_labels.shape}, "
          f"unique label IDs: {sorted(np.unique(contact_labels).tolist())}")

    # --- Human SDF (KDTree on sampled merged-mesh surface) ----------------
    seq_out = output_dir / sequence_name
    human_dir = seq_out / 'human'
    human_dir.mkdir(parents=True, exist_ok=True)
    sdf_path = human_dir / 'sdf.npy'
    sdf_json = human_dir / 'sdf.json'
    if sdf_path.exists() and sdf_json.exists():
        print("Loading cached human SDF...")
        info = json.load(open(sdf_json))
        centroid = np.asarray(info['centroid'], dtype=np.float32)
        extents = np.asarray(info['extents'], dtype=np.float32)
        sdf = np.load(sdf_path)
    else:
        print("Building human SDF (KDTree on sampled surface)...")
        faces = trimesh.load(str(Path(args.mesh_ds_dir) / 'mesh_2.obj'), process=False).faces
        t0 = time.time()
        centroid, extents, sdf = build_human_sdf_kdtree(
            vertices, faces, grid_dim=args.sdf_grid_dim,
        )
        print(f"  built {sdf.shape} SDF in {time.time()-t0:.1f}s")
        json.dump({'centroid': centroid.tolist(), 'extents': extents.tolist(),
                   'grid_dim': args.sdf_grid_dim}, open(sdf_json, 'w'))
        np.save(sdf_path, sdf)
    sdf_t = torch.tensor(sdf, dtype=torch.float32, device=device)
    centroid_t = torch.tensor(centroid, dtype=torch.float32, device=device)
    extents_t = torch.tensor(extents, dtype=torch.float32, device=device)

    # --- Floor height: min-z of body verts (no floor class) ---------------
    floor_height = float(vertices[..., 2].min())
    print(f"Floor height (min-z of body verts): {floor_height:.3f}")

    # --- Local majority voting: collect all contact pts per class, mix, DBSCAN ---
    print("Local majority voting...")
    cluster_contact_points: list[np.ndarray] = []
    cluster_contact_labels: list[int] = []
    for cid in classes_eps:
        cls_pts = []
        for f in range(vertices.shape[0]):
            cls_pts.extend(vertices[f][contact_labels[f] == cid])
        if not cls_pts:
            continue
        cls_pts = np.asarray(cls_pts)
        cls_pts = voxel_downsample(cls_pts, pcd_down_voxel_size)
        cluster_contact_points.append(cls_pts)
        cluster_contact_labels.append(np.full(cls_pts.shape[0], cid, dtype=np.int64))
    if not cluster_contact_points:
        print(f"No contact vertices for any humoto class in {sequence_name}; skipping fitting.")
        return
    all_pts = np.concatenate(cluster_contact_points, axis=0)
    all_lbls = np.concatenate(cluster_contact_labels, axis=0)
    cluster_labels = dbscan_cluster(all_pts, voting_eps, cluster_min_points)
    voted_pts: list[np.ndarray] = []
    voted_lbls: list[int] = []
    for lab in range(cluster_labels.max() + 1):
        m = cluster_labels == lab
        if int(m.sum()) < cluster_min_points:
            continue
        majority = int(np.bincount(all_lbls[m]).argmax())
        print(f"  voted cluster {lab}: {int(m.sum())} pts → class {majority} ({id_to_name.get(majority)})")
        voted_pts.append(all_pts[m])
        voted_lbls.append(np.full(int(m.sum()), majority, dtype=np.int64))
    if not voted_pts:
        print(f"No clusters survived voting for {sequence_name}; skipping fitting.")
        return
    voted_pts = np.concatenate(voted_pts, axis=0)
    voted_lbls = np.concatenate(voted_lbls, axis=0)

    # --- Per-class DBSCAN: each cluster becomes one object instance ---
    print("Per-class clustering...")
    clusters_classes: list[int] = []
    clusters_points: list[np.ndarray] = []
    clusters_inst: list[int] = []
    for cid in classes_eps:
        cls_pts = voted_pts[voted_lbls == cid]
        if len(cls_pts) == 0:
            continue
        cls_pts = voxel_downsample(cls_pts, pcd_down_voxel_size)
        if len(cls_pts) < cluster_min_points:
            continue
        c_eps = float(classes_eps[cid])
        labs = dbscan_cluster(cls_pts, c_eps, cluster_min_points)
        for lab in range(labs.max() + 1):
            mask = labs == lab
            if int(mask.sum()) < cluster_min_points:
                continue
            clusters_classes.append(cid)
            clusters_points.append(cls_pts[mask])
            clusters_inst.append(lab)
            print(f"  class {cid} ({id_to_name.get(cid)}): inst {lab}, "
                  f"{int(mask.sum())} pts, eps={c_eps:.2f}")

    # --- Per-cluster pose fitting -------------------------------------------
    for i, cid in enumerate(clusters_classes):
        cls_pts = clusters_points[i]
        inst_idx = clusters_inst[i]
        cls_name = id_to_name.get(cid, f'cls{cid}')
        cluster_base = seq_out / 'fit_best_obj' / cls_name / str(inst_idx)
        cluster_base.mkdir(parents=True, exist_ok=True)
        # Save the cluster pcd as obj (open3d-free).
        np.save(cluster_base / 'cluster_pts.npy', cls_pts)

        cluster_pts_tensor = torch.tensor(cls_pts, dtype=torch.float32, device=device)
        contact_min_x = cls_pts[:, 0].min(); contact_max_x = cls_pts[:, 0].max()
        contact_min_y = cls_pts[:, 1].min(); contact_max_y = cls_pts[:, 1].max()
        contact_center_x = (contact_max_x + contact_min_x) / 2
        contact_center_y = (contact_max_y + contact_min_y) / 2

        # Single canonical mesh per humoto class.
        obj_path = objects_dir / cls_name / f'{cls_name}.obj'
        if not obj_path.exists():
            print(f"  [skip] no mesh for class '{cls_name}': {obj_path}")
            continue
        obj_mesh = trimesh.load_mesh(str(obj_path), process=False)
        obj_verts = np.asarray(obj_mesh.vertices)
        obj_faces = np.asarray(obj_mesh.faces)

        # Align the (Y-up local) mesh to Z-up floor and lift to floor_height.
        floor_aligned_verts = align_obj_to_floor(obj_verts, obj_faces,
                                                 str(cluster_base / 'floor_aligned.obj'))
        transformed_verts = np.copy(floor_aligned_verts)
        transformed_verts[:, 2] += floor_height

        # Translate xy so object centroid matches cluster centroid.
        omin = transformed_verts.min(axis=0); omax = transformed_verts.max(axis=0)
        ocx = (omin[0] + omax[0]) / 2; ocy = (omin[1] + omax[1]) / 2
        x_t = contact_center_x - ocx; y_t = contact_center_y - ocy
        transformed_verts[:, 0] += x_t; transformed_verts[:, 1] += y_t
        ocx += x_t; ocy += y_t
        omin[0] += x_t; omax[0] += x_t; omin[1] += y_t; omax[1] += y_t
        write_verts_faces_obj(transformed_verts, obj_faces, str(cluster_base / 'transformed.obj'))

        # Sample the centered object surface with poisson-disk-like density.
        centered_verts = np.copy(transformed_verts)
        centered_verts[:, 0] -= ocx; centered_verts[:, 1] -= ocy
        centered_mesh = trimesh.Trimesh(vertices=centered_verts, faces=obj_faces, process=False)
        size = (omax - omin)
        n_pts_target = max(int(size[0] * size[1] * size[2] * (config.pts_per_unit ** 3)), 256)
        try:
            obj_pts_centered, _ = trimesh.sample.sample_surface(centered_mesh, n_pts_target)
        except Exception:
            obj_pts_centered = centered_verts
        obj_pts_centered = voxel_downsample(np.asarray(obj_pts_centered), pcd_down_voxel_size)
        print(f"  [{cls_name}/{inst_idx}] sampled {len(obj_pts_centered)} object surface points")

        # Grid search.
        t0 = time.time()
        gbest_loss, gbest_rot, gbest_x, gbest_y, gbest_pts = grid_search(
            cid, obj_pts_centered, ocx, ocy, omin[0], omin[1], omax[0], omax[1],
            cluster_pts_tensor, contact_min_x, contact_min_y, contact_max_x, contact_max_y,
            sdf_t, centroid_t, extents_t,
            grid_search_contact_weight, grid_search_pen_thresh,
            grid_search_classes_pen_weight, device,
        )
        print(f"  [{cls_name}/{inst_idx}] grid: loss={gbest_loss:.3f} rot={gbest_rot}° "
              f"in {time.time()-t0:.1f}s")
        # Save grid result.
        rmat = R.from_euler('XYZ', [0, 0, gbest_rot], degrees=True)
        cand_centered = rmat.apply(centered_verts)
        cand = np.copy(cand_centered)
        cand[:, 0] += ocx + gbest_x; cand[:, 1] += ocy + gbest_y
        write_verts_faces_obj(cand, obj_faces, str(cluster_base / 'grid_search_best.obj'))
        json.dump({'loss': float(gbest_loss), 'rot_deg': float(gbest_rot),
                   'transl_x': float(gbest_x), 'transl_y': float(gbest_y)},
                  open(cluster_base / 'grid_search_best.json', 'w'))

        # Adam refinement.
        gx_center = ocx + gbest_x; gy_center = ocy + gbest_y
        t0 = time.time()
        bloss, brot, btx, bty, bpts = optimization(
            cid, obj_pts_centered, gx_center, gy_center, gbest_rot,
            cluster_pts_tensor, contact_min_x, contact_min_y, contact_max_x, contact_max_y,
            sdf_t, centroid_t, extents_t,
            opt_contact_weight, opt_pen_thresh, opt_classes_pen_weight,
            lr, opt_steps, device,
        )
        print(f"  [{cls_name}/{inst_idx}] opt:  loss={bloss:.3f} rot={brot/math.pi*180:.2f}° "
              f"in {time.time()-t0:.1f}s")
        rmat = R.from_euler('XYZ', [0, 0, brot], degrees=False)
        opt_v = rmat.apply(cand_centered)
        opt_v[:, 0] += gx_center + btx; opt_v[:, 1] += gy_center + bty
        write_verts_faces_obj(opt_v, obj_faces, str(cluster_base / 'opt_best.obj'))
        json.dump({'loss': float(bloss), 'rot_deg': float(brot / math.pi * 180),
                   'transl_x': float(btx), 'transl_y': float(bty),
                   'class_id': int(cid), 'class_name': cls_name, 'instance_idx': int(inst_idx)},
                  open(cluster_base / 'opt_best.json', 'w'))

    print(f"Done. Results in {seq_out}")


if __name__ == '__main__':
    main()
