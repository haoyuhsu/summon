"""Single-sequence rerun visualizer for GT vs predicted humoto scenes.

Loads one of the 30 humoto test sequences (selected by `--seq_n N` in 0..29) and
renders both GT and pred motions + objects in a shared canonical Y-up frame
(first-frame pelvis at origin, body-forward → +Z). Useful for sanity-checking
the metric script's coordinate alignment and for inspecting failure modes.

GT  : orange body, green objects.
Pred: light-blue body, pink objects.
World axes (red/green/blue) at origin = X / Y(up) / Z(forward).
"""
from __future__ import annotations

import argparse
import pickle
import sys
import types
from pathlib import Path

import numpy as np
import rerun as rr
import torch
import trimesh
from human_body_prior.body_model.body_model import BodyModel
from tqdm import tqdm

# Same module-stub trick used elsewhere — these are imported in summon code paths
# we don't actually need.
for _m in ('open3d', 'pandas', 'eulerangles', 'smplx'):
    sys.modules.setdefault(_m, types.ModuleType(_m))

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, '/home/haoyuyh3/Documents/maxhsu/imu-humans/humoto')

from humoto_utils import (  # noqa: E402
    HUMOTO_DATA_DIR, HUMOTO_OBJECTS_DIR, R_YUP_TO_ZUP, SMPLX_MODEL_PATH,
    pose7_to_rotmat_transl,
)
from humoto_metrics import (  # noqa: E402
    apply_T,
    calculate_bbox_ious,
    calculate_normalized_l1,
    canonical_transform,
    dedupe_pred_by_class,
)


PRED_MOTION_DIR_DEFAULT = '/home/haoyuyh3/Documents/maxhsu/imu-humans/_tmp_pred_mobileposer/result_humoto'
PRED_FITTING_DIR_DEFAULT = str(SCRIPT_DIR / 'fitting_results' / 'humoto_pred')
TEST_INDICES_DEFAULT = '/home/haoyuyh3/Documents/maxhsu/imu-humans/humoto/humoto_data/test_indices.npy'

GT_HUMAN_COLOR = [255, 180, 0]
PRED_HUMAN_COLOR = [150, 150, 255]
GT_OBJ_COLOR = [100, 255, 100]
PRED_OBJ_COLOR = [255, 150, 150]


def log_world_axes(scale: float = 0.5) -> None:
    """Log RGB-coloured X/Y/Z axes at the origin."""
    o = np.zeros(3)
    rr.log('world/axes/x',
           rr.LineStrips3D([np.stack([o, [scale, 0, 0]])], colors=[255, 0, 0],
                           radii=0.005, labels=['X']),
           static=True)
    rr.log('world/axes/y',
           rr.LineStrips3D([np.stack([o, [0, scale, 0]])], colors=[0, 255, 0],
                           radii=0.005, labels=['Y (up)']),
           static=True)
    rr.log('world/axes/z',
           rr.LineStrips3D([np.stack([o, [0, 0, scale]])], colors=[0, 0, 255],
                           radii=0.005, labels=['Z (forward)']),
           static=True)


def run_smplx_seq(body_model: BodyModel, pose: np.ndarray, orient: np.ndarray,
                  trans: np.ndarray, device: torch.device) -> np.ndarray:
    """Run SMPL-X for a full sequence and return body verts (T, V, 3)."""
    T = pose.shape[0]
    with torch.no_grad():
        out = body_model(
            pose_body=torch.tensor(pose).to(device),
            root_orient=torch.tensor(orient).to(device),
            trans=torch.tensor(trans).to(device),
            betas=torch.zeros(T, 10, dtype=torch.float32, device=device),
        )
    return out.v.detach().cpu().numpy()


def main() -> None:
    """Render GT vs pred for one humoto test sequence in rerun (canonical Y-up frame)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seq_n', type=int, default=18,
                        help='Index in test_indices.npy (0..29). Default 18 = humoto 0000241 '
                             '(densest pred — table + utility_cart + working_chair).')
    parser.add_argument('--gt_data_dir', type=str, default=str(HUMOTO_DATA_DIR))
    parser.add_argument('--gt_objects_dir', type=str, default=str(HUMOTO_OBJECTS_DIR))
    parser.add_argument('--test_indices', type=str, default=TEST_INDICES_DEFAULT)
    parser.add_argument('--pred_motion_dir', type=str, default=PRED_MOTION_DIR_DEFAULT)
    parser.add_argument('--pred_fitting_dir', type=str, default=PRED_FITTING_DIR_DEFAULT)
    parser.add_argument('--smplx_model_path', type=str, default=SMPLX_MODEL_PATH)
    parser.add_argument('--fps', type=float, default=30.0)
    parser.add_argument('--max_frames', type=int, default=300,
                        help='Cap sequence length to keep rerun responsive.')
    parser.add_argument('--distance_threshold', type=float, default=0.45)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_indices = np.load(args.test_indices)
    n = args.seq_n
    assert 0 <= n < len(test_indices), f"--seq_n must be in [0, {len(test_indices)})"
    seq_id = f'{int(test_indices[n]):07d}'
    print(f"Sequence {n} → humoto id {seq_id}")

    body_model = BodyModel(
        bm_fname=args.smplx_model_path, num_betas=10, model_type='smplx',
    ).to(device)
    R_z2y = R_YUP_TO_ZUP.T.astype(np.float32)

    # ----------------- GT side -----------------
    with open(Path(args.gt_data_dir) / f'{seq_id}.pkl', 'rb') as f:
        gt = pickle.load(f)
    gt_motion = gt['motion_smpl'].astype(np.float32)
    gt_pose = gt_motion[:, 3:66]
    gt_orient = gt_motion[:, 0:3]
    gt_transl = gt_motion[:, 72:75]
    gt_text = gt.get('text', [''])
    print(f"GT  T = {gt_pose.shape[0]}, text = {gt_text}")

    print("Running GT SMPL-X forward...")
    gt_verts = run_smplx_seq(body_model, gt_pose, gt_orient, gt_transl, device)
    gt_pelvis0 = gt_verts.mean(axis=1)[0]  # rough; fine for canon. Or use Jtr.
    # Use the joints-based pelvis explicitly for accuracy.
    with torch.no_grad():
        gt_pelvis0 = body_model(
            pose_body=torch.tensor(gt_pose[:1]).to(device),
            root_orient=torch.tensor(gt_orient[:1]).to(device),
            trans=torch.tensor(gt_transl[:1]).to(device),
            betas=torch.zeros(1, 10, dtype=torch.float32, device=device),
        ).Jtr[0, 0].detach().cpu().numpy()
    T_gt = canonical_transform(gt_orient[0], gt_pelvis0)
    gt_verts_canon = apply_T(T_gt, gt_verts.reshape(-1, 3)).reshape(gt_verts.shape).astype(np.float32)
    smplx_faces = np.asarray(body_model.f.detach().cpu().numpy(), dtype=np.int32)

    gt_objs: list = []
    for name, pose7 in gt.get('objects', {}).items():
        if name == 'ground':
            continue
        mp = Path(args.gt_objects_dir) / name / f'{name}.obj'
        if not mp.exists():
            continue
        mesh = trimesh.load(str(mp), process=False)
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
    print(f"GT  objects: {[o['name'] for o in gt_objs]}")

    # ----------------- Pred side -----------------
    pred_p = Path(args.pred_motion_dir) / f'sample_seq_{n:04d}.npy'
    pred = np.load(pred_p, allow_pickle=True).item()['pred']
    pred_pose = pred['pose'].astype(np.float32)
    pred_orient = pred['orient'].astype(np.float32)
    pred_transl = pred['transl'].astype(np.float32)
    print(f"Pred T = {pred_pose.shape[0]}")

    print("Running Pred SMPL-X forward...")
    pred_verts = run_smplx_seq(body_model, pred_pose, pred_orient, pred_transl, device)
    with torch.no_grad():
        pred_pelvis0 = body_model(
            pose_body=torch.tensor(pred_pose[:1]).to(device),
            root_orient=torch.tensor(pred_orient[:1]).to(device),
            trans=torch.tensor(pred_transl[:1]).to(device),
            betas=torch.zeros(1, 10, dtype=torch.float32, device=device),
        ).Jtr[0, 0].detach().cpu().numpy()
    T_pred = canonical_transform(pred_orient[0], pred_pelvis0)
    pred_verts_canon = apply_T(T_pred, pred_verts.reshape(-1, 3)).reshape(pred_verts.shape).astype(np.float32)

    pred_objs: list = []
    seq_fit = Path(args.pred_fitting_dir) / seq_id / 'fit_best_obj'
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
                v_yup = np.asarray(mesh.vertices) @ R_z2y.T
                v_canon = apply_T(T_pred, v_yup).astype(np.float32)
                pred_objs.append({'name': cls_dir.name, 'inst': inst_dir.name,
                                  'verts': v_canon[None],
                                  'faces': np.asarray(mesh.faces, dtype=np.int32)})
    print(f"Pred objects (raw): {[(o['name'], o['inst']) for o in pred_objs]}")

    # ----------------- Metrics readout for this sequence -----------------
    pred_objs_d = dedupe_pred_by_class(pred_objs, gt_objs)
    ious, id_p, id_r = calculate_bbox_ious(gt_objs, pred_objs_d)
    l1s, _, _, p_t, r_t = calculate_normalized_l1(
        gt_objs, pred_objs_d, gt_verts_canon, args.distance_threshold,
    )
    print()
    print(f"=== Metrics for {seq_id} ===")
    print(f"  ID Precision = {id_p:.4f},  ID Recall = {id_r:.4f}")
    print(f"  3D bbox IoU per class: " + ", ".join(f"{k}={v:.3f}" for k, v in ious.items()))
    print(f"  Normalized L1 per class: " + ", ".join(f"{k}={v:.3f}" for k, v in l1s.items()))
    print(f"  Precision@{args.distance_threshold} = {p_t:.4f}, Recall@{args.distance_threshold} = {r_t:.4f}")

    # ----------------- Rerun logging -----------------
    rr.init('humoto_metrics_viz', spawn=True)
    rr.log('world', rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)
    log_world_axes()

    # Static: GT and pred objects (frame 0).
    for o in gt_objs:
        m = trimesh.Trimesh(vertices=o['verts'][0], faces=o['faces'], process=False)
        rr.log(
            f"world/gt/objects/{o['name']}",
            rr.Mesh3D(vertex_positions=m.vertices, vertex_normals=m.vertex_normals,
                      triangle_indices=m.faces, vertex_colors=GT_OBJ_COLOR),
            static=True,
        )
    for o in pred_objs:
        m = trimesh.Trimesh(vertices=o['verts'][0], faces=o['faces'], process=False)
        rr.log(
            f"world/pred/objects/{o['name']}__{o['inst']}",
            rr.Mesh3D(vertex_positions=m.vertices, vertex_normals=m.vertex_normals,
                      triangle_indices=m.faces, vertex_colors=PRED_OBJ_COLOR),
            static=True,
        )

    # Per-frame: GT and pred body meshes.
    n_frames = min(args.max_frames, gt_verts_canon.shape[0], pred_verts_canon.shape[0])
    print(f"Logging {n_frames} frames...")
    dt = 1.0 / args.fps
    for i in tqdm(range(n_frames), dynamic_ncols=True):
        rr.set_time_sequence('frame', i)
        rr.set_time_seconds('sensor_time', i * dt)

        gm = trimesh.Trimesh(vertices=gt_verts_canon[i], faces=smplx_faces, process=False)
        rr.log('world/gt/smplx',
               rr.Mesh3D(vertex_positions=gm.vertices, vertex_normals=gm.vertex_normals,
                         triangle_indices=gm.faces, vertex_colors=GT_HUMAN_COLOR),
               static=False)

        pm = trimesh.Trimesh(vertices=pred_verts_canon[i], faces=smplx_faces, process=False)
        rr.log('world/pred/smplx',
               rr.Mesh3D(vertex_positions=pm.vertices, vertex_normals=pm.vertex_normals,
                         triangle_indices=pm.faces, vertex_colors=PRED_HUMAN_COLOR),
               static=False)

    print("\nIn rerun:")
    print("  world/gt/...   (orange body, green objects)")
    print("  world/pred/... (blue body, pink objects)")
    print("Both bodies should start at origin facing +Z. If chairs from GT and pred")
    print("land at similar positions, the metric agrees with intuition.")


if __name__ == '__main__':
    main()
