"""Build ContactFormer-input vertices from predicted SMPL-X motion.

Reads sample_seq_NNNN.npy files in /_tmp_pred_mobileposer/result_humoto/ (the test
set predicted by mobileposer; 30 sequences). Each .npy is a pickled dict with
{'gt': {pose, transl, orient}, 'pred': {pose, transl, orient}}. The predicted
motion is in **Y-up** SMPL-X (verified by extending the body at frame 0 and
finding y-axis spans ~1.7 m), so we apply the same Y→Z conversion as training.

Output naming maps `sample_seq_NNNN.npy` → humoto sample id `f"{test_indices[NNNN]:07d}"`,
which matches the layout of training data so cross-referencing is direct.

Per .npy, produces 2 files in {output_dir}/:
    vertices/<id>_verts.npy         (T, 655, 3) world-space body verts (Z-up)
    vertices_can/<id>_verts_can.npy (T, 655, 3) pelvis-centered, yaw-normalized
"""
from __future__ import annotations

import argparse
import os
import sys
import types
from pathlib import Path

import numpy as np
import torch
from human_body_prior.body_model.body_model import BodyModel
from tqdm import tqdm

# `data_utils` imports `open3d`/`pandas`/`eulerangles`/`smplx` at module scope but
# none of the functions we need use them. Stub to allow import in any env.
for _mod in ('open3d', 'pandas', 'eulerangles', 'smplx'):
    sys.modules.setdefault(_mod, types.ModuleType(_mod))

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, '/home/haoyuyh3/Documents/maxhsu/imu-humans/humoto')

import data_utils as du  # noqa: E402
from humoto_utils import SMPLX_MODEL_PATH, yup_to_zup_smplx  # noqa: E402


PRED_DIR_DEFAULT = '/home/haoyuyh3/Documents/maxhsu/imu-humans/_tmp_pred_mobileposer/result_humoto'
TEST_INDICES_DEFAULT = '/home/haoyuyh3/Documents/maxhsu/imu-humans/humoto/humoto_data/test_indices.npy'
OUTPUT_DIR_DEFAULT = '/home/haoyuyh3/Documents/maxhsu/imu-humans/summon/data_custom/humoto_pred'
MESH_DS_DIR_DEFAULT = '/home/haoyuyh3/Documents/maxhsu/imu-humans/summon/mesh_ds'


def process_one_pred(
    npy_path: Path,
    seq_id: str,
    body_model: BodyModel,
    rest_pelvis: np.ndarray,
    ds_fn1, ds_fn2,
    associated_joints: torch.Tensor,
    output_dir: Path,
    device: torch.device,
    use_branch: str = 'pred',
) -> tuple[int, tuple[float, float]]:
    """Load predicted motion → Y→Z → SMPL-X → downsample → save. Returns (T, (z_min, z_max))."""
    data = np.load(npy_path, allow_pickle=True).item()
    branch = data[use_branch]
    pose = branch['pose'].astype(np.float32)        # (T, 63)
    orient_y = branch['orient'].astype(np.float32)  # (T, 3)
    transl_y = branch['transl'].astype(np.float32)  # (T, 3)
    T = pose.shape[0]

    # Y → Z conversion (same helper used at training time).
    orient_z, transl_z = yup_to_zup_smplx(orient_y, transl_y, rest_pelvis)

    # Single batched SMPL-X forward.
    with torch.no_grad():
        out = body_model(
            pose_body=torch.tensor(pose, device=device),
            root_orient=torch.tensor(orient_z, device=device),
            trans=torch.tensor(transl_z, device=device),
            betas=torch.zeros((T, 10), dtype=torch.float32, device=device),
        )
    verts_world = out.v                          # (T, 10475, 3)
    pelvis_world = out.Jtr[:, 0]                 # (T, 3)

    # Canonical = world minus per-frame pelvis (no second forward pass).
    verts_can_full = verts_world - pelvis_world.unsqueeze(1)

    # Downsample 10475 → 2619 → 655 via the same spiral matrices used at training.
    verts_world_655 = ds_fn2(ds_fn1(verts_world))
    verts_can_655 = ds_fn2(ds_fn1(verts_can_full))

    # First-frame yaw normalization (assumes Z-up world).
    verts_can_norm = du.normalize_orientation(verts_can_655, associated_joints, device)

    np.save(output_dir / 'vertices' / f'{seq_id}_verts.npy',
            verts_world_655.detach().cpu().numpy())
    np.save(output_dir / 'vertices_can' / f'{seq_id}_verts_can.npy',
            verts_can_norm.detach().cpu().numpy())

    z = verts_world_655[..., 2]
    return T, (float(z.min().item()), float(z.max().item()))


def main() -> None:
    """Generate ContactFormer-input vertices for the 30 mobileposer-predicted test sequences."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pred_dir', type=str, default=PRED_DIR_DEFAULT)
    parser.add_argument('--test_indices', type=str, default=TEST_INDICES_DEFAULT)
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR_DEFAULT)
    parser.add_argument('--smplx_model_path', type=str, default=SMPLX_MODEL_PATH)
    parser.add_argument('--mesh_ds_us_dir', type=str, default=MESH_DS_DIR_DEFAULT)
    parser.add_argument('--use_branch', type=str, default='pred', choices=['pred', 'gt'],
                        help="Which branch of the .npy to read; 'pred' for inference, "
                             "'gt' for sanity-checking against ground-truth motion.")
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    output_dir = Path(args.output_dir)
    mesh_ds_dir = Path(args.mesh_ds_us_dir)
    for sub in ('vertices', 'vertices_can'):
        (output_dir / sub).mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Map sample_seq_NNNN → humoto id via test_indices.
    test_indices = np.load(args.test_indices)
    print(f"Loaded {len(test_indices)} test indices from {args.test_indices}")

    pred_files = sorted(p for p in os.listdir(pred_dir) if p.startswith('sample_seq_') and p.endswith('.npy'))
    assert len(pred_files) == len(test_indices), (
        f"# pred files ({len(pred_files)}) != # test indices ({len(test_indices)})"
    )

    print("Loading SMPL-X model...")
    body_model = BodyModel(
        bm_fname=args.smplx_model_path, num_betas=10, model_type='smplx',
    ).to(device)
    rest_pelvis = body_model().Jtr[0, 0].detach().cpu().numpy().astype(np.float32)

    _, _, D1 = du.get_graph_params(str(mesh_ds_dir), device, layer=1)
    ds1 = du.ds_us(D1).to(device)
    _, _, D2 = du.get_graph_params(str(mesh_ds_dir), device, layer=2)
    ds2 = du.ds_us(D2).to(device)

    ds_weights = torch.tensor(np.load(SCRIPT_DIR / 'support_files' / 'downsampled_weights.npy'))
    associated_joints = torch.argmax(ds_weights, dim=1)

    print(f"Processing {len(pred_files)} predicted motion files (branch='{args.use_branch}') → {output_dir}")
    summary = []
    for fname in tqdm(pred_files, desc='Pred', dynamic_ncols=True):
        n = int(fname.replace('sample_seq_', '').replace('.npy', ''))
        seq_id = f'{int(test_indices[n]):07d}'
        T, z_range = process_one_pred(
            pred_dir / fname, seq_id, body_model, rest_pelvis, ds1, ds2,
            associated_joints, output_dir, device, use_branch=args.use_branch,
        )
        summary.append((fname, seq_id, T, z_range))

    print()
    print("Per-sequence body z-range (sanity check; expect feet near 0, head near 1.7):")
    for fname, seq_id, T, (z_min, z_max) in summary:
        print(f"  {fname} → {seq_id}  T={T:>4d}  z=[{z_min:+.3f}, {z_max:+.3f}]")


if __name__ == '__main__':
    main()
