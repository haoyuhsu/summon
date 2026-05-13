"""Run humoto_fit_best_obj.py for every predicted humoto test sequence.

Resumable: skips sequences whose output dir already exists. Sequence enumeration uses
the .npy files in contact_labels_dir (so seq names are humoto IDs like 0000585).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent


def main() -> None:
    """Iterate predicted contact .npy files and dispatch humoto_fit_best_obj.py per sequence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vertices_dir', type=str,
                        default=str(SCRIPT_DIR / 'data_custom' / 'humoto_pred' / 'vertices'))
    parser.add_argument('--contact_labels_dir', type=str,
                        default=str(SCRIPT_DIR / 'contact_predictions' / 'humoto_pred'))
    parser.add_argument('--output_dir', type=str,
                        default=str(SCRIPT_DIR / 'fitting_results' / 'humoto_pred'))
    parser.add_argument('--input_probability', action='store_true', default=True,
                        help='Default True: predict_contact.py was run with --save_probability.')
    parser.add_argument('--no_input_probability', dest='input_probability', action='store_false')
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=-1)
    parser.add_argument('--python', type=str, default=sys.executable,
                        help="Python interpreter to use for spawned humoto_fit_best_obj.py runs.")
    args = parser.parse_args()

    contact_dir = Path(args.contact_labels_dir)
    vertices_dir = Path(args.vertices_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seq_names = sorted(p.stem for p in contact_dir.glob('*.npy'))
    if args.end_idx < 0:
        args.end_idx = len(seq_names)
    seq_names = seq_names[args.start_idx:args.end_idx]
    print(f"Found {len(seq_names)} sequences in {contact_dir}")

    fit_script = SCRIPT_DIR / 'humoto_fit_best_obj.py'
    progress_path = output_dir / 'progress.txt'

    for seq in tqdm(seq_names, desc='Fitting', dynamic_ncols=True):
        seq_out = output_dir / seq
        if seq_out.exists() and any(seq_out.glob('fit_best_obj/*/*/opt_best.obj')):
            print(f"Skipping {seq}: results already exist.")
            continue
        verts_p = vertices_dir / f'{seq}_verts.npy'
        contact_p = contact_dir / f'{seq}.npy'
        if not verts_p.exists() or not contact_p.exists():
            print(f"  [skip] missing inputs for {seq}: {verts_p.exists()=}, {contact_p.exists()=}")
            continue

        cmd = (f'{args.python} {fit_script} '
               f'--sequence_name {seq} '
               f'--vertices_path {verts_p} '
               f'--contact_labels_path {contact_p} '
               f'--output_dir {output_dir}')
        if args.input_probability:
            cmd += ' --input_probability'

        t0 = time.time()
        ret = os.system(cmd)
        if ret != 0:
            print(f"  [warn] {seq} exited with code {ret}")
        with open(progress_path, 'a') as f:
            f.write(f'{seq}\t{ret}\t{time.time()-t0:.1f}s\n')


if __name__ == '__main__':
    main()
