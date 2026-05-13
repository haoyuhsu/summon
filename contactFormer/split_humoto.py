"""Create humoto_train/ and humoto_valid/ from data_custom/humoto/ via symlinks.

Uses the canonical split shipped with humoto:
    humoto/humoto_data/train_indices.npy  (705 indices)
    humoto/humoto_data/test_indices.npy   (30 indices; doubles as validation set)

Each index `i` maps to filename `{i:07d}` (matching humoto_gen_dataset.py output).

Usage:
    python split_humoto.py                                 # use canonical split
    python split_humoto.py --src_dir <path>                # if your data is elsewhere
    python split_humoto.py --valid_indices <other.npy>     # override validation indices
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


SUBDIRS = ('vertices', 'vertices_can', 'semantics', 'contacts')


def main() -> None:
    """Split humoto-generated dataset into train/valid via symlinks using canonical indices."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--src_dir', type=str,
                        default='/home/haoyuyh3/Documents/maxhsu/imu-humans/summon/data_custom/humoto')
    parser.add_argument('--train_dir', type=str,
                        default='/home/haoyuyh3/Documents/maxhsu/imu-humans/summon/data_custom/humoto_train')
    parser.add_argument('--valid_dir', type=str,
                        default='/home/haoyuyh3/Documents/maxhsu/imu-humans/summon/data_custom/humoto_valid')
    parser.add_argument('--train_indices', type=str,
                        default='/home/haoyuyh3/Documents/maxhsu/imu-humans/humoto/humoto_data/train_indices.npy')
    parser.add_argument('--valid_indices', type=str,
                        default='/home/haoyuyh3/Documents/maxhsu/imu-humans/humoto/humoto_data/test_indices.npy',
                        help='Indices used for validation (test set doubles as validation).')
    parser.add_argument('--overwrite', action='store_true',
                        help='Remove existing symlinks in train/valid dirs first.')
    args = parser.parse_args()

    src = Path(args.src_dir)
    train = Path(args.train_dir)
    valid = Path(args.valid_dir)

    sem_dir = src / 'semantics'
    assert sem_dir.is_dir(), f"missing {sem_dir} — run humoto_gen_dataset.py first"
    available_seqs = {f.split('_cfs.npy')[0] for f in os.listdir(sem_dir) if f.endswith('_cfs.npy')}

    train_idx = np.load(args.train_indices)
    valid_idx = np.load(args.valid_indices)
    assert len(set(train_idx.tolist()) & set(valid_idx.tolist())) == 0, "train/valid overlap!"

    train_ids = [f'{int(i):07d}' for i in train_idx]
    valid_ids = [f'{int(i):07d}' for i in valid_idx]

    missing_train = [s for s in train_ids if s not in available_seqs]
    missing_valid = [s for s in valid_ids if s not in available_seqs]
    if missing_train or missing_valid:
        print(f"  [warn] {len(missing_train)} train + {len(missing_valid)} valid sequences not found "
              f"in {sem_dir}; they will be skipped.")
    train_ids = [s for s in train_ids if s in available_seqs]
    valid_ids = [s for s in valid_ids if s in available_seqs]

    print(f"src:   {src}")
    print(f"train: {train} ({len(train_ids)} sequences from {args.train_indices})")
    print(f"valid: {valid} ({len(valid_ids)} sequences from {args.valid_indices})")

    suffix_for = {
        'vertices': '_verts.npy',
        'vertices_can': '_verts_can.npy',
        'semantics': '_cfs.npy',
        'contacts': '_cf.npy',
    }

    for out_root, ids in [(train, train_ids), (valid, list(valid_ids))]:
        for sub in SUBDIRS:
            (out_root / sub).mkdir(parents=True, exist_ok=True)
            if args.overwrite:
                for p in (out_root / sub).iterdir():
                    if p.is_symlink() or p.is_file():
                        p.unlink()
        for seq in ids:
            for sub in SUBDIRS:
                fname = f'{seq}{suffix_for[sub]}'
                src_path = src / sub / fname
                dst_path = out_root / sub / fname
                if not src_path.exists():
                    continue
                if dst_path.exists() or dst_path.is_symlink():
                    dst_path.unlink()
                os.symlink(src_path.resolve(), dst_path)

    print("Done.")


if __name__ == '__main__':
    main()
