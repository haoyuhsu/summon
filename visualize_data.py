import os
import numpy as np
import glob
import json
from pathlib import Path
import torch
import trimesh
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import sys
sys.path.append('/home/haoyuyh3/Documents/maxhsu/imu-humans/imu-human-mllm/dataset_process/motionmillion_and_lingo/smpl272_utils')
from motion_conversion import smpl85_to_smpl272, load_smplx_model
# Load SMPL-X model (use human_body_prior)
smplx_model = load_smplx_model('/home/haoyuyh3/Documents/maxhsu/imu-humans/related_works/motion/MotionMillion-Codes/body_models/human_model_files/smplx/SMPLX_NEUTRAL.npz')
# Get default pelvis position
default_smplx_output = smplx_model()
rest_pelvis = default_smplx_output.Jtr[0, 0].detach().cpu().numpy()

import sys
import rerun as rr
sys.path.append('/home/haoyuyh3/Documents/maxhsu/imu-humans/imu-human-mllm/imu_synthesis/')
from viewer import Viewer, generate_meshes_body_model
from visualize_imu import log_smpl_85
viewer = Viewer()
fps = 30.0
dt = 1.0 / fps



root_dir = '/home/haoyuyh3/Documents/maxhsu/imu-humans/summon/fitting_results/humoto'
root_vert_dir = '/home/haoyuyh3/Documents/maxhsu/imu-humans/summon/data_custom/humoto/vertices'
motion_dir = '/home/haoyuyh3/Documents/maxhsu/imu-humans/_tmp_test_data/result_humoto'

seq_fnames = sorted(os.listdir(root_dir))
for seq_fname in seq_fnames:

    input_dir = Path(root_dir) / seq_fname

    fit_best_obj_dir = input_dir / 'fit_best_obj'
    if not fit_best_obj_dir.exists():
        continue

    print(f"Processing sequence: {seq_fname}")

    res_dir = input_dir / 'fit_best_obj'
    obj_mesh_list = []
    for obj_class_dir in res_dir.iterdir():
        for obj_dir in obj_class_dir.iterdir():
            with open(str(obj_dir / 'best_obj_id.json'), "r") as f:
                best_obj_json = json.load(f)
            best_obj_id = best_obj_json['best_obj_id']
            best_obj_path = obj_dir / best_obj_id / 'opt_best.obj'
            # obj_mesh = o3d.io.read_triangle_mesh(str(best_obj_path))
            # obj_mesh.compute_vertex_normals()
            obj_mesh = trimesh.load_mesh(str(best_obj_path))

            # Convert from z-up to y-up
            R_zup_to_yup = np.array([
                [1,  0,  0],
                [0,  0,  1],
                [0, -1,  0]
            ])
            obj_mesh.vertices = (R_zup_to_yup @ obj_mesh.vertices.T).T

            # log the mesh
            # rr.set_time_sequence("frames", 0)
            # rr.set_time_seconds("sensor_time", 0.0)
            # rr.log(
            #     f"world/seq_{seq_fname}/obj_{obj_class_dir.name}",
            #     rr.Mesh3D(  
            #         vertex_positions=obj_mesh.vertices,
            #         vertex_normals=obj_mesh.vertex_normals,
            #         triangle_indices=obj_mesh.faces,
            #         vertex_colors=[100, 100, 100],
            #         ),
            #     static=True,
            # )

            obj_mesh_list.append(obj_mesh)

    # # load vertices
    # vert_path_list = sorted(glob.glob(os.path.join(root_vert_dir, seq_fname.split('.')[0] + '_verts.npy')))
    # verts_all = np.load(vert_path_list[0])  # (N, 10475, 3)

    
    # res = np.load(os.path.join(motion_dir, seq_fname.split('.')[0] + '.npy'), allow_pickle=True).item()

    # gt_motion, pred_motion = res['gt'], res['pred']

    # global_orient = pred_motion['orient'].astype(np.float32)    # (N, 3)
    # transl = pred_motion['transl'].astype(np.float32)           # (N, 3)
    # body_pose = pred_motion['pose'].astype(np.float32)          # (N, 63)

    # # Transform the SMPL-X from y-up to z-up (DEBUG)
    # # R_yup_to_zup = np.array([
    # #     [1,  0,  0],
    # #     [0,  0, -1],
    # #     [0,  1,  0]
    # # ])
    # # transl = (R_yup_to_zup @ transl.T).T  # (N, 3)
    # # transl -= rest_pelvis
    # # global_orient = R.from_matrix(
    # #     np.einsum('ij,njk->nik', R_yup_to_zup, R.from_rotvec(global_orient.reshape(-1, 3)).as_matrix())
    # # ).as_rotvec().reshape(-1, 3)


    # default_smplx_output = smplx_model()
    # rest_pelvis = default_smplx_output.Jtr[0, 0].detach().cpu().numpy()
    # transl = transl - rest_pelvis


    # import torch
    # seq_len = global_orient.shape[0]
    # with torch.no_grad():
    #     smplx_output = smplx_model(
    #         pose_body=torch.from_numpy(body_pose).cuda().float(), 
    #         root_orient=torch.from_numpy(global_orient).cuda().float(),
    #         trans=torch.from_numpy(transl).cuda().float(),
    #         betas=torch.zeros((global_orient.shape[0], 10), dtype=torch.float32).cuda()
    #     )
    #     vertices = smplx_output.v.detach().cpu().numpy()
    # faces = smplx_model.f.detach().cpu().numpy()
    # # faces = simplified_smplx_model.faces.detach().cpu().numpy()
    # original_meshes = [trimesh.Trimesh(vertices=vertices[i], faces=faces, process=False) for i in range(seq_len)]


    # # Generate SMPL-X meshes
    # for frame_idx in tqdm(range(seq_len), desc="Logging SMPL-X meshes", dynamic_ncols=True):
    #     rr.set_time_sequence("frames", frame_idx)
    #     rr.set_time_seconds("sensor_time", frame_idx * dt)
    #     rr.log(
    #         f"world/seq_{seq_fname}/obj_traj",
    #         rr.Mesh3D(  
    #             vertex_positions=original_meshes[frame_idx].vertices,
    #             vertex_normals=original_meshes[frame_idx].vertex_normals,
    #             triangle_indices=original_meshes[frame_idx].faces,
    #             vertex_colors=[255, 180, 0],
    #             ),
    #         static=False,
    #     )

    # rr.flush()