import os
from tqdm import tqdm
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Fit best objects to contact labels and vertices")
    parser.add_argument("--vertices_path", type=str, required=True,
                        help="Path to the numpy file containing the vertices sequence")
    parser.add_argument("--contact_labels_path", type=str, required=True,
                        help="Path to the numpy file containing the contact labels sequence")
    parser.add_argument("--output_dir", type=str, default="fitting_results",
                        help="Directory to save the fitting results")
    parser.add_argument('--start_idx', type=int, default=0,
                        help='Start index of the sequence to process')
    parser.add_argument('--end_idx', type=int, default=-1,
                        help='End index of the sequence to process')
    args = parser.parse_args()

    sequence_fnames = sorted(os.listdir(args.contact_labels_path))

    if args.end_idx == -1:
        args.end_idx = len(sequence_fnames)

    for sequence_fname in tqdm(sequence_fnames[args.start_idx:args.end_idx]):

        sequence_name = os.path.splitext(sequence_fname)[0]

        if os.path.exists(f'{args.output_dir}/{sequence_name}'):
            print(f'Skipping {sequence_name}, already exists.')
            continue

        # Run the fitting script
        os.system(f'python fit_best_obj.py \
            --sequence_name {sequence_name} \
            --vertices_path {args.vertices_path}/{sequence_name}_verts.npy \
            --contact_labels_path {args.contact_labels_path}/{sequence_name}.npy \
            --output_dir {args.output_dir}'
        )

        with open(f'{args.output_dir}/progress.txt', 'a') as f:
            f.write(f'{sequence_name}\n')