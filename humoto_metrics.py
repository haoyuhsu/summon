"""3D-object-prediction metrics for humoto test set.

Compares predicted fitted objects (from `humoto_fit_best_obj.py`) against the GT
object layout from humoto pickles, after canonicalizing both motions to a shared
"first-frame pelvis at origin, +Z forward, +Y up" frame.

Per-sequence outputs (verbatim port of summon/viz_prediction_from_nn_with_metric_2.py):
- ID Precision / ID Recall  (set match on object names; 'ground' excluded)
- 3D bounding-box IoU       (axis-aligned, per matched class)
- Normalized L1             (centroid distance / scene diagonal)
- Precision@0.45 / Recall@0.45 (threshold on normalized L1)

Empty-prediction sequences contribute zeros to the mean (penalising failure modes).
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
import trimesh
from human_body_prior.body_model.body_model import BodyModel
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

# Stub modules data_utils.py imports but doesn't need (only needed when humoto_utils touches them).
for _m in ('open3d', 'pandas', 'eulerangles', 'smplx'):
    sys.modules.setdefault(_m, types.ModuleType(_m))

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, '/home/haoyuyh3/Documents/maxhsu/imu-humans/humoto')

from humoto_utils import (  # noqa: E402
    HUMOTO_DATA_DIR, HUMOTO_OBJECTS_DIR, R_YUP_TO_ZUP, SMPLX_MODEL_PATH,
    pose7_to_rotmat_transl,
)


PRED_MOTION_DIR_DEFAULT = '/home/haoyuyh3/Documents/maxhsu/imu-humans/_tmp_pred_mobileposer/result_humoto'
PRED_FITTING_DIR_DEFAULT = str(SCRIPT_DIR / 'fitting_results' / 'humoto_pred')
TEST_INDICES_DEFAULT = '/home/haoyuyh3/Documents/maxhsu/imu-humans/humoto/humoto_data/test_indices.npy'


def canonical_transform(global_orient_aa: np.ndarray, pelvis_world: np.ndarray) -> np.ndarray:
    """Return 4x4 affine that places pelvis at origin and rotates body-forward (+Z in SMPL-X canonical) onto +Z.
    The rotation is around the +Y axis only, so Y stays the up axis.
    """
    R0 = R.from_rotvec(np.asarray(global_orient_aa, dtype=np.float64)).as_matrix()
    fwd = R0 @ np.array([0.0, 0.0, 1.0])
    fwd_xz = np.array([fwd[0], 0.0, fwd[2]])
    n = float(np.linalg.norm(fwd_xz))
    if n < 1e-8:
        # Degenerate: body looking straight up/down. Identity rotation.
        Ry = np.eye(3)
    else:
        fwd_xz /= n
        angle = -np.arctan2(fwd_xz[0], fwd_xz[2])  # rotate around Y so fwd → +Z
        Ry = R.from_rotvec(np.array([0.0, angle, 0.0])).as_matrix()
    T = np.eye(4)
    T[:3, :3] = Ry
    T[:3, 3] = -Ry @ np.asarray(pelvis_world, dtype=np.float64)
    return T.astype(np.float32)


def apply_T(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply 4x4 affine to (..., 3) points."""
    return points @ T[:3, :3].T + T[:3, 3]


def run_smplx_first_frame_pelvis(
    body_model: BodyModel,
    pose_body: np.ndarray,    # (T, 63)
    root_orient: np.ndarray,  # (T, 3)
    trans: np.ndarray,        # (T, 3)
    device: torch.device,
) -> np.ndarray:
    """Run SMPL-X for frame 0 only and return the world-space pelvis position."""
    with torch.no_grad():
        out = body_model(
            pose_body=torch.tensor(pose_body[:1]).to(device),
            root_orient=torch.tensor(root_orient[:1]).to(device),
            trans=torch.tensor(trans[:1]).to(device),
            betas=torch.zeros(1, 10, dtype=torch.float32, device=device),
        )
    return out.Jtr[0, 0].detach().cpu().numpy()


def run_smplx_all(
    body_model: BodyModel,
    pose_body: np.ndarray,
    root_orient: np.ndarray,
    trans: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Run SMPL-X for the full sequence, return body verts (T, 10475, 3)."""
    T = pose_body.shape[0]
    with torch.no_grad():
        out = body_model(
            pose_body=torch.tensor(pose_body).to(device),
            root_orient=torch.tensor(root_orient).to(device),
            trans=torch.tensor(trans).to(device),
            betas=torch.zeros(T, 10, dtype=torch.float32, device=device),
        )
    return out.v.detach().cpu().numpy()


def calculate_bbox_ious(gt_objects_data: list, pred_objects_data: list) -> tuple[dict, float, float]:
    """3D AABB IoU per matched class + ID precision/recall. Verbatim from reference."""
    pred_dict = {obj['name']: obj for obj in pred_objects_data}
    gt_names = set(o['name'] for o in gt_objects_data if o['name'] != 'ground')
    pred_names = set(o['name'] for o in pred_objects_data if o['name'] != 'ground')
    tp = len(gt_names & pred_names)
    fp = len(pred_names - gt_names)
    fn = len(gt_names - pred_names)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    bbox_ious: dict = {}
    for gt in gt_objects_data:
        n = gt['name']
        if n == 'ground':
            continue
        gv = gt['verts'][0]
        gmin, gmax = gv.min(axis=0), gv.max(axis=0)
        gvol = float(np.prod(gmax - gmin))
        if n not in pred_dict:
            bbox_ious[n] = 0.0
            continue
        pv = pred_dict[n]['verts'][0]
        pmin, pmax = pv.min(axis=0), pv.max(axis=0)
        pvol = float(np.prod(pmax - pmin))
        imin = np.maximum(gmin, pmin)
        imax = np.minimum(gmax, pmax)
        ivol = float(np.prod(np.maximum(0.0, imax - imin)))
        union = gvol + pvol - ivol
        bbox_ious[n] = (ivol / union) if union > 0 else 0.0
    return bbox_ious, precision, recall


def calculate_normalized_l1(
    gt_objects_data: list,
    pred_objects_data: list,
    gt_human_verts_canon: np.ndarray | None,
    distance_threshold: float = 0.45,
) -> tuple[dict, float, float, float, float]:
    """Normalized L1 + label-based + threshold-based P/R. Verbatim from reference."""
    pred_dict = {obj['name']: obj for obj in pred_objects_data}
    all_v: list = []
    for o in gt_objects_data:
        if o['name'] != 'ground':
            all_v.append(o['verts'][0])
    if gt_human_verts_canon is not None and gt_human_verts_canon.size > 0:
        all_v.append(gt_human_verts_canon.reshape(-1, 3))
    if all_v:
        all_v = np.concatenate(all_v, axis=0)
        scene_size = float(np.linalg.norm(all_v.max(axis=0) - all_v.min(axis=0)))
    else:
        scene_size = 1.0

    gt_names = set(o['name'] for o in gt_objects_data if o['name'] != 'ground')
    pred_names = set(o['name'] for o in pred_objects_data if o['name'] != 'ground')
    tp = len(gt_names & pred_names)
    fp = len(pred_names - gt_names)
    fn = len(gt_names - pred_names)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    tp_t = fp_t = fn_t = 0
    l1_errors: dict = {}
    for gt in gt_objects_data:
        n = gt['name']
        if n == 'ground':
            continue
        gc = gt['verts'][0].mean(axis=0)
        if n not in pred_dict:
            l1_errors[n] = 1.0
            fn_t += 1
            continue
        pc = pred_dict[n]['verts'][0].mean(axis=0)
        l1 = float(np.sum(np.abs(gc - pc)))
        nl1 = min(l1 / scene_size, 1.0) if scene_size > 0 else 0.0
        l1_errors[n] = nl1
        if nl1 < distance_threshold:
            tp_t += 1
        else:
            fn_t += 1
            fp_t += 1
    for p in pred_objects_data:
        if p['name'] == 'ground':
            continue
        if p['name'] not in gt_names:
            fp_t += 1

    p_t = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0.0
    r_t = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0.0
    return l1_errors, precision, recall, p_t, r_t


def dedupe_pred_by_class(pred_objs: list, gt_objs: list) -> list:
    """If multiple pred instances share the same class, keep the one closest to the GT
    instance of the same class (L1 of centroids). Classes without a GT match keep the
    first instance — they'll be FPs regardless."""
    gt_centroids = {o['name']: o['verts'][0].mean(axis=0)
                    for o in gt_objs if o['name'] != 'ground'}
    by_cls: dict = {}
    for p in pred_objs:
        by_cls.setdefault(p['name'], []).append(p)
    out: list = []
    for cls, instances in by_cls.items():
        if cls in gt_centroids and len(instances) > 1:
            target = gt_centroids[cls]
            best = min(instances,
                       key=lambda x: float(np.sum(np.abs(x['verts'][0].mean(axis=0) - target))))
        else:
            best = instances[0]
        out.append(best)
    return out


def load_gt(seq_id: str, gt_data_dir: Path, gt_objects_dir: Path,
            body_model: BodyModel, device: torch.device) -> tuple[np.ndarray, list, np.ndarray]:
    """Returns (T_gt, gt_objs_canon, gt_human_verts_canon). Y-up native; canonicalized to +Z forward."""
    with open(gt_data_dir / f'{seq_id}.pkl', 'rb') as f:
        gt = pickle.load(f)
    motion = gt['motion_smpl'].astype(np.float32)
    pose = motion[:, 3:66]
    orient = motion[:, 0:3]
    transl = motion[:, 72:75]

    pelvis0 = run_smplx_first_frame_pelvis(body_model, pose, orient, transl, device)
    T_gt = canonical_transform(orient[0], pelvis0)

    # GT human verts (full sequence) for the scene-size term.
    verts = run_smplx_all(body_model, pose, orient, transl, device)
    verts_canon = apply_T(T_gt, verts.reshape(-1, 3)).reshape(verts.shape)

    gt_objs: list = []
    for name, pose7 in gt.get('objects', {}).items():
        if name == 'ground':
            continue
        mesh_p = gt_objects_dir / name / f'{name}.obj'
        if not mesh_p.exists():
            continue
        mesh = trimesh.load(str(mesh_p), process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(mesh.dump())
        pose7 = np.asarray(pose7, dtype=np.float32)
        if pose7.shape != (7,):
            continue
        R_obj, t_obj = pose7_to_rotmat_transl(pose7)
        v_world = (np.asarray(mesh.vertices) @ R_obj.T) + t_obj
        v_canon = apply_T(T_gt, v_world).astype(np.float32)
        gt_objs.append({'name': name, 'verts': v_canon[None],
                        'faces': np.asarray(mesh.faces, dtype=np.int32)})
    return T_gt, gt_objs, verts_canon


def load_pred(seq_n: int, seq_id: str, pred_motion_dir: Path, pred_fitting_dir: Path,
              body_model: BodyModel, device: torch.device,
              R_z2y: np.ndarray) -> tuple[np.ndarray, list]:
    """Returns (T_pred, pred_objs_canon). Pred motion is Y-up; fitted objects are Z-up → reversed."""
    pred_p = pred_motion_dir / f'sample_seq_{seq_n:04d}.npy'
    pred = np.load(pred_p, allow_pickle=True).item()['pred']
    pose = pred['pose'].astype(np.float32)
    orient = pred['orient'].astype(np.float32)
    transl = pred['transl'].astype(np.float32)
    pelvis0 = run_smplx_first_frame_pelvis(body_model, pose, orient, transl, device)
    T_pred = canonical_transform(orient[0], pelvis0)

    pred_objs: list = []
    seq_fit = pred_fitting_dir / seq_id / 'fit_best_obj'
    if seq_fit.is_dir():
        for cls_dir in sorted(seq_fit.iterdir()):
            if not cls_dir.is_dir():
                continue
            for inst_dir in sorted(cls_dir.iterdir(), key=lambda p: p.name):
                obj_p = inst_dir / 'opt_best.obj'
                if not obj_p.exists():
                    continue
                mesh = trimesh.load(str(obj_p), process=False)
                if isinstance(mesh, trimesh.Scene):
                    mesh = trimesh.util.concatenate(mesh.dump())
                v_yup = np.asarray(mesh.vertices) @ R_z2y.T  # Z-up → Y-up
                v_canon = apply_T(T_pred, v_yup).astype(np.float32)
                pred_objs.append({'name': cls_dir.name, 'inst': inst_dir.name,
                                  'verts': v_canon[None],
                                  'faces': np.asarray(mesh.faces, dtype=np.int32)})
    return T_pred, pred_objs


def compute_seq_metrics(gt_objs: list, pred_objs: list,
                        gt_human_verts_canon: np.ndarray | None,
                        distance_threshold: float) -> dict:
    """Run the two reference helpers + dedupe, return a flat dict of per-sequence metrics."""
    pred_objs_d = dedupe_pred_by_class(pred_objs, gt_objs)
    ious, id_p, id_r = calculate_bbox_ious(gt_objs, pred_objs_d)
    l1s, _, _, p_t, r_t = calculate_normalized_l1(
        gt_objs, pred_objs_d, gt_human_verts_canon, distance_threshold,
    )
    mean_iou = float(np.mean(list(ious.values()))) if ious else 0.0
    mean_l1 = float(np.mean(list(l1s.values()))) if l1s else 1.0
    return {
        'id_precision': float(id_p),
        'id_recall': float(id_r),
        'mean_iou': mean_iou,
        'mean_l1': mean_l1,
        'precision_at_thresh': float(p_t),
        'recall_at_thresh': float(r_t),
        'per_obj_iou': {k: float(v) for k, v in ious.items()},
        'per_obj_l1': {k: float(v) for k, v in l1s.items()},
        'gt_classes': sorted(o['name'] for o in gt_objs if o['name'] != 'ground'),
        'pred_classes': sorted(set(o['name'] for o in pred_objs_d if o['name'] != 'ground')),
        'n_pred_instances_raw': len(pred_objs),
    }


def main() -> None:
    """Compute 3D metrics over the 30 humoto test sequences and save aggregate to JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gt_data_dir', type=str, default=str(HUMOTO_DATA_DIR))
    parser.add_argument('--gt_objects_dir', type=str, default=str(HUMOTO_OBJECTS_DIR))
    parser.add_argument('--test_indices', type=str, default=TEST_INDICES_DEFAULT)
    parser.add_argument('--pred_motion_dir', type=str, default=PRED_MOTION_DIR_DEFAULT)
    parser.add_argument('--pred_fitting_dir', type=str, default=PRED_FITTING_DIR_DEFAULT)
    parser.add_argument('--smplx_model_path', type=str, default=SMPLX_MODEL_PATH)
    parser.add_argument('--distance_threshold', type=float, default=0.45)
    parser.add_argument('--out_json', type=str,
                        default=str(SCRIPT_DIR / 'fitting_results' / 'humoto_pred' / 'metrics.json'))
    args = parser.parse_args()

    gt_data_dir = Path(args.gt_data_dir)
    gt_objects_dir = Path(args.gt_objects_dir)
    pred_motion_dir = Path(args.pred_motion_dir)
    pred_fitting_dir = Path(args.pred_fitting_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"GT data:    {gt_data_dir}")
    print(f"GT objects: {gt_objects_dir}")
    print(f"Pred motion:  {pred_motion_dir}")
    print(f"Pred fitting: {pred_fitting_dir}")
    print(f"Distance threshold (τ): {args.distance_threshold}")

    body_model = BodyModel(
        bm_fname=args.smplx_model_path, num_betas=10, model_type='smplx',
    ).to(device)
    R_z2y = R_YUP_TO_ZUP.T.astype(np.float32)

    test_indices = np.load(args.test_indices)
    print(f"Test indices: {len(test_indices)} sequences")

    rows: list = []
    for n, hidx in enumerate(tqdm(test_indices, desc='Sequences')):
        seq_id = f'{int(hidx):07d}'
        try:
            _, gt_objs, gt_verts_canon = load_gt(seq_id, gt_data_dir, gt_objects_dir, body_model, device)
        except FileNotFoundError as e:
            print(f"  [skip] {seq_id}: GT data missing ({e})")
            continue
        try:
            _, pred_objs = load_pred(n, seq_id, pred_motion_dir, pred_fitting_dir,
                                     body_model, device, R_z2y)
        except FileNotFoundError as e:
            print(f"  [warn] {seq_id}: pred motion missing ({e}); treating as empty pred.")
            pred_objs = []
        m = compute_seq_metrics(gt_objs, pred_objs, gt_verts_canon, args.distance_threshold)
        m['seq_n'] = int(n)
        m['humoto_id'] = seq_id
        rows.append(m)

    # Aggregate.
    keys = ['id_precision', 'id_recall', 'mean_iou', 'mean_l1',
            'precision_at_thresh', 'recall_at_thresh']
    means = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    stds = {k: float(np.std([r[k] for r in rows])) for k in keys}

    print()
    print("=" * 80)
    print(f"Per-sequence metrics across {len(rows)} sequences")
    print("=" * 80)
    print(f"{'seq_n':>5} {'humoto_id':>9} {'IDP':>5} {'IDR':>5} {'IoU':>5} {'L1':>5} {'P@τ':>5} {'R@τ':>5}  preds")
    for r in rows:
        print(f"{r['seq_n']:>5d} {r['humoto_id']:>9} "
              f"{r['id_precision']:>5.2f} {r['id_recall']:>5.2f} "
              f"{r['mean_iou']:>5.2f} {r['mean_l1']:>5.2f} "
              f"{r['precision_at_thresh']:>5.2f} {r['recall_at_thresh']:>5.2f}  "
              f"{r['n_pred_instances_raw']}")
    print()
    print("Aggregate (mean ± std across sequences):")
    print(f"  ID Precision      : {means['id_precision']:.4f} ± {stds['id_precision']:.4f}")
    print(f"  ID Recall         : {means['id_recall']:.4f} ± {stds['id_recall']:.4f}")
    print(f"  3D bbox IoU       : {means['mean_iou']:.4f} ± {stds['mean_iou']:.4f}")
    print(f"  Normalized L1     : {means['mean_l1']:.4f} ± {stds['mean_l1']:.4f}")
    print(f"  Precision@{args.distance_threshold:.2f}    : {means['precision_at_thresh']:.4f} ± {stds['precision_at_thresh']:.4f}")
    print(f"  Recall@{args.distance_threshold:.2f}       : {means['recall_at_thresh']:.4f} ± {stds['recall_at_thresh']:.4f}")

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump({'aggregate': {'mean': means, 'std': stds, 'n_sequences': len(rows)},
                   'per_sequence': rows,
                   'config': {'distance_threshold': args.distance_threshold,
                              'pred_fitting_dir': str(pred_fitting_dir)}},
                  f, indent=2)
    print(f"\nSaved metrics to {out_json}")


if __name__ == '__main__':
    main()
