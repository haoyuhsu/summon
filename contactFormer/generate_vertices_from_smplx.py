import os
import numpy as np
import torch
import argparse
import data_utils as du
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32


def generate_vertices_from_smplx_params(
    smplx_params_dict,
    output_dir,
    sequence_name,
    smplx_model_dir,
    ds_us_dir,
    gender='neutral',
    num_pca_comps=6,
    downsample=True,
    normalize_ori=True
):
    """
    Generate verts.npy and verts_can.npy from SMPL-X parameters.
    
    Args:
        smplx_params_dict: Dictionary containing SMPL-X parameters
            - 'global_orient': (N, 3) rotation vector
            - 'transl': (N, 3) translation
            - 'body_pose': (N, 63) or (N, 21, 3) body pose
            - 'betas': (N, 10) or (10,) shape parameters
            - 'left_hand_pose': (N, 6) or (N, 12) [optional]
            - 'right_hand_pose': (N, 6) or (N, 12) [optional]
        output_dir: Directory to save the output files
        sequence_name: Name of the sequence (e.g., 'MPH11_00150_01')
        smplx_model_dir: Path to SMPL-X model directory
        ds_us_dir: Path to mesh downsampling/upsampling matrices
        gender: 'neutral', 'male', or 'female'
        num_pca_comps: Number of PCA components for hand pose
        downsample: Whether to downsample the mesh
        normalize_ori: Whether to normalize orientation
    """
    
    # Create output directories
    verts_dir = os.path.join(output_dir, "vertices")
    verts_can_dir = os.path.join(output_dir, "vertices_can")
    os.makedirs(verts_dir, exist_ok=True)
    os.makedirs(verts_can_dir, exist_ok=True)
    
    # Load SMPL-X body model
    num_frames = smplx_params_dict['global_orient'].shape[0]
    body_model = du.load_body_model(
        model_folder=smplx_model_dir,
        num_pca_comps=num_pca_comps,
        batch_size=1,
        gender=gender
    ).to(device)
    
    # Load downsampling functions if needed
    if downsample:
        _, _, D_1 = du.get_graph_params(ds_us_dir, device, layer=1)
        down_sample_fn = du.ds_us(D_1).to(device)
        
        _, _, D_2 = du.get_graph_params(ds_us_dir, device, layer=2)
        down_sample_fn2 = du.ds_us(D_2).to(device)
    
    # Load orientation normalization data
    if normalize_ori:
        ds_weights = torch.tensor(np.load("./support_files/downsampled_weights.npy"))
        associated_joints = torch.argmax(ds_weights, dim=1)
    
    # Convert parameters to tensors
    global_orient = torch.tensor(smplx_params_dict['global_orient'], dtype=dtype).to(device)
    transl = torch.tensor(smplx_params_dict['transl'], dtype=dtype).to(device)
    body_pose = torch.tensor(smplx_params_dict['body_pose'], dtype=dtype).to(device)
    
    # Handle betas (can be same for all frames)
    betas = torch.tensor(smplx_params_dict['betas'], dtype=dtype).to(device)
    if betas.dim() == 1:
        betas = betas.unsqueeze(0).repeat(num_frames, 1)
    
    # Handle hand poses (optional)
    left_hand_pose = None
    right_hand_pose = None
    if 'left_hand_pose' in smplx_params_dict:
        left_hand_pose = torch.tensor(smplx_params_dict['left_hand_pose'], dtype=dtype).to(device)
    else:
        left_hand_pose = torch.zeros(num_frames, num_pca_comps, dtype=dtype).to(device)
    
    if 'right_hand_pose' in smplx_params_dict:
        right_hand_pose = torch.tensor(smplx_params_dict['right_hand_pose'], dtype=dtype).to(device)
    else:
        right_hand_pose = torch.zeros(num_frames, num_pca_comps, dtype=dtype).to(device)
    
    # Process each frame
    vertices_list = []
    vertices_can_list = []
    
    print(f"Processing {num_frames} frames for sequence {sequence_name}...")
    for i in tqdm(range(num_frames)):
        # Prepare parameters for current frame
        torch_param = {
            'betas': betas[i:i+1, :10],
            'global_orient': global_orient[i:i+1],
            'body_pose': body_pose[i:i+1],
            'transl': transl[i:i+1],
            'left_hand_pose': left_hand_pose[i:i+1, :num_pca_comps],
            'right_hand_pose': right_hand_pose[i:i+1, :num_pca_comps]
        }
        
        # Generate body mesh with translation (world space)
        body_model.reset_params(**torch_param)
        body_output = body_model(return_verts=True)
        vertices = body_output.vertices.squeeze()  # (10475, 3) or (6890, 3)

        # get the minimum z value across all vertices
        # print("min y value:", torch.min(vertices[:, 1]))
        # print("min z value:", torch.min(vertices[:, 2]))
        # breakpoint()

        # Generate canonical mesh (centered at pelvis, no translation)
        torch_param_can = torch_param.copy()
        torch_param_can['transl'] = torch.zeros_like(transl[i:i+1])
        body_model.reset_params(**torch_param_can)
        body_output_can = body_model(return_verts=True)
        vertices_can = body_output_can.vertices.squeeze()
        pelvis = body_output_can.joints[:, 0, :].reshape(1, 3)
        vertices_can = vertices_can - pelvis  # Center at pelvis
        
        # Store vertices (unsqueeze to add batch dimension)
        vertices_list.append(vertices.unsqueeze(0))
        vertices_can_list.append(vertices_can.unsqueeze(0))
    
    # Stack all frames
    vertices_all = torch.cat(vertices_list, dim=0)  # (N, V, 3)
    vertices_can_all = torch.cat(vertices_can_list, dim=0)  # (N, V, 3)
    
    # Downsample if requested
    if downsample:
        print("Downsampling vertices...")
        vertices_all = down_sample_fn.forward(vertices_all)
        vertices_all = down_sample_fn2.forward(vertices_all)
        
        vertices_can_all = down_sample_fn.forward(vertices_can_all)
        vertices_can_all = down_sample_fn2.forward(vertices_can_all)
    
    # Normalize orientation if requested
    if normalize_ori:
        print("Normalizing orientation...")
        vertices_can_all = du.normalize_orientation(vertices_can_all, associated_joints, device)
    
    # Save to files
    verts_path = os.path.join(verts_dir, f"{sequence_name}_verts.npy")
    verts_can_path = os.path.join(verts_can_dir, f"{sequence_name}_verts_can.npy")
    
    np.save(verts_path, vertices_all.detach().cpu().numpy())
    np.save(verts_can_path, vertices_can_all.detach().cpu().numpy())
    
    print(f"Saved vertices to: {verts_path}")
    print(f"  Shape: {vertices_all.shape}")
    print(f"Saved canonical vertices to: {verts_can_path}")
    print(f"  Shape: {vertices_can_all.shape}")
    
    return verts_path, verts_can_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate vertices from SMPL parameters")
    # parser.add_argument("--smpl_params_path", type=str, required=True,
    #                     help="Path to .npz or .pkl file containing SMPL parameters")
    # parser.add_argument("--sequence_name", type=str, required=True,
    #                     help="Name of the sequence (e.g., MPH11_00150_01)")
    parser.add_argument("--root_dir", type=str, default="/home/haoyuyh3/Documents/maxhsu/imu-humans/_tmp_test_data/result_humoto",
                        help="Root directory containing SMPL parameters files")
    parser.add_argument("--output_dir", type=str, default="../data/custom_sequences",
                        help="Output directory for generated vertices")
    parser.add_argument("--smplx_model_dir", type=str, default="../../POSA/POSA_dir/smplx_models",
                        help="Path to SMPL-X model directory")
    parser.add_argument("--mesh_ds_us_dir", type=str, default="../../POSA/POSA_dir/mesh_ds",
                        help="Path to mesh downsampling matrices")
    parser.add_argument("--gender", type=str, default='neutral',
                        choices=['neutral', 'male', 'female'],
                        help="Gender of the body model")
    parser.add_argument("--downsample", action='store_true', default=True,
                        help="Downsample vertices (10475 -> 655)")
    parser.add_argument("--normalize_ori", action='store_true', default=True,
                        help="Normalize orientation to face forward")
    
    args = parser.parse_args()
    
    # # Load SMPL parameters from file
    # if args.smpl_params_path.endswith('.npz'):
    #     smpl_data = np.load(args.smpl_params_path)
    #     smpl_params = {key: smpl_data[key] for key in smpl_data.files}
    # elif args.smpl_params_path.endswith('.pkl'):
    #     import pickle
    #     with open(args.smpl_params_path, 'rb') as f:
    #         smpl_params = pickle.load(f)
    # else:
    #     raise ValueError("SMPL parameters file must be .npz or .pkl")


    from scipy.spatial.transform import Rotation as R
    import sys
    sys.path.append('/home/haoyuyh3/Documents/maxhsu/imu-humans/imu-human-mllm/dataset_process/motionmillion_and_lingo/smpl272_utils')
    from motion_conversion import smpl85_to_smpl272, load_smplx_model
    # Load SMPL-X model (use human_body_prior)
    smplx_model = load_smplx_model('/home/haoyuyh3/Documents/maxhsu/imu-humans/related_works/motion/MotionMillion-Codes/body_models/human_model_files/smplx/SMPLX_NEUTRAL.npz')
    # Get default pelvis position
    default_smplx_output = smplx_model()
    rest_pelvis = default_smplx_output.Jtr[0, 0].detach().cpu().numpy()


    # import sys
    # import rerun as rr
    # sys.path.append('/home/haoyuyh3/Documents/maxhsu/imu-humans/imu-human-mllm/imu_synthesis/')
    # from viewer import Viewer, generate_meshes_body_model
    # from visualize_imu import log_smpl_85
    # viewer = Viewer()
    # fps = 30.0
    # dt = 1.0 / fps
    # log_smpl_85(viewer, smpl85_data, seq_idx=seq_name, label='smpl85')


    root_dir = args.root_dir

    for filename in sorted(os.listdir(root_dir)):

        res = np.load(os.path.join(root_dir, filename), allow_pickle=True).item()

        gt_motion, pred_motion = res['gt'], res['pred']

        global_orient = pred_motion['orient']    # (N, 3)
        transl = pred_motion['transl']           # (N, 3)
        body_pose = pred_motion['pose']          # (N, 63)


        # # transl aligned to the first frame at origin
        # meshes = generate_meshes_body_model(orient=global_orient, pose=body_pose, transl=transl-rest_pelvis, smplx_model=smplx_model)
        # # Log SMPL-X meshes
        # for frame_idx in tqdm(range(global_orient.shape[0]), desc="Logging SMPL-X meshes", dynamic_ncols=True):
        #     rr.set_time_sequence("frames", frame_idx)
        #     rr.set_time_seconds("sensor_time", frame_idx * dt)
        #     viewer.log_smplx_mesh(meshes[frame_idx], 0, label='Before')


        # Transform the SMPL-X from y-up to z-up (DEBUG)
        R_yup_to_zup = np.array([
            [1,  0,  0],
            [0,  0, -1],
            [0,  1,  0]
        ])
        transl = (R_yup_to_zup @ transl.T).T  # (N, 3)
        transl -= rest_pelvis
        global_orient = R.from_matrix(
            np.einsum('ij,njk->nik', R_yup_to_zup, R.from_rotvec(global_orient.reshape(-1, 3)).as_matrix())
        ).as_rotvec().reshape(-1, 3)


        # meshes = generate_meshes_body_model(orient=global_orient, pose=body_pose, transl=transl, smplx_model=smplx_model)
        # # Log SMPL-X meshes
        # for frame_idx in tqdm(range(global_orient.shape[0]), desc="Logging SMPL-X meshes", dynamic_ncols=True):
        #     rr.set_time_sequence("frames", frame_idx)
        #     rr.set_time_seconds("sensor_time", frame_idx * dt)
        #     viewer.log_smplx_mesh(meshes[frame_idx], 0, label='After')
        # breakpoint()


        smplx_params = {
            'global_orient': global_orient,  # (N, 3)
            'transl': transl,                # (N, 3)
            'body_pose': body_pose,          # (N, 63)
            'betas': np.zeros((global_orient.shape[0], 10)),                  # (10,) or (N, 10)
            # 'left_hand_pose': res['left_hand_pose'],    # (N, 6) optional
            # 'right_hand_pose': res['right_hand_pose'],  # (N, 6) optional
        }
        sequence_name = filename.split('.')[0]
    
        # Generate vertices
        generate_vertices_from_smplx_params(
            smplx_params_dict=smplx_params,
            output_dir=args.output_dir,
            sequence_name=sequence_name,
            smplx_model_dir=args.smplx_model_dir,
            ds_us_dir=args.mesh_ds_us_dir,
            gender=args.gender,
            downsample=args.downsample,
            normalize_ori=args.normalize_ori
        )