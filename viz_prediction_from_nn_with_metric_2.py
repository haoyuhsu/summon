import sys
sys.path.append('../imu-human-mllm')

import os
import pickle
import numpy as np
import trimesh
import torch
import argparse
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4
from tqdm import tqdm
import random
import copy
from typing import Dict, List, Optional
from collections import deque
from scipy.spatial.transform import Rotation as R
import json

import rerun as rr
import rerun.blueprint as rrb

from smplx import SMPLX
import torch

from rot2 import convert_rotation
# from smooth import *
# from smooth2 import reshape_and_smooth_timeseries
from third_party.Showo.models.metrics import compute_mpjpe

import matplotlib.pyplot as plt
from torch.nn import functional as F
def smooth_time_series(x: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    """
    Smooth a [ntime, nvar] tensor along the time dimension using
    a moving-average filter.

    Args:
        x: [ntime, nvar]
        kernel_size: odd int, size of temporal smoothing window

    Returns:
        smoothed: [ntime, nvar]
    """
    assert x.dim() == 2, "x must be [ntime, nvar]"
    ntime, nvar = x.shape

    # [ntime, nvar] -> [1, nvar, ntime] for conv1d (batch, channels, length)
    x_ = x.T.unsqueeze(0)

    # Moving-average kernel
    kernel = torch.ones(1, 1, kernel_size, device=x.device, dtype=x.dtype)
    kernel = kernel / kernel_size
    # One kernel per variable (groups conv)
    kernel = kernel.expand(nvar, 1, kernel_size)

    padding = kernel_size // 2  # keep same length

    smoothed = F.conv1d(x_, kernel, padding=padding, groups=nvar)  # [1, nvar, ntime]
    smoothed = smoothed.squeeze(0).T  # -> [ntime, nvar]

    return smoothed

@dataclass(frozen=True)
class ViewerConfig:
    output_rrd: Path = None
    sample_fps: float = 1
    rotate_rgb: bool = True
    downsample_rgb: bool = True
    jpeg_quality: int = 90
    traj_tail_length: int = 100
    distance_threshold: float = 0.45  # Threshold for normalized L1 distance to consider as True Positive

    point_radii: float = 0.008
    line_radii: float = 0.008
    skel_radii: float = 0.01
 
class DataViewer(ViewerConfig):

    palette: Dict[str, list] = {
        "scene": [200, 200, 200],
        "smplx": [255, 180, 0],
        "bev": [150, 150, 255],
        "traj": [
            [81, 71, 252], 
            [80, 244, 204],
            [255, 34, 17],
        ],
    }

    def __init__(self, smplx_model_path, all_objects_folder,**kwargs) -> None:
        super().__init__(**kwargs)

        rr.init(
            "data viewer 4", 
            spawn=(self.output_rrd is None),
            recording_id=uuid4()
        )
        if self.output_rrd is not None:
            rr.save(self.output_rrd)

        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)

        # Load SMPL-X model
        self.smplx_model = SMPLX(model_path=smplx_model_path, use_pca=False)
        
        self._epaths_3d: set[str] = set()
        self._traj_deques: dict[str, deque] = {}
        self._log_coordinate_axes()  # world coordinate axes

        self.seq_len_limit = 300     # avoid logging too long sequences
        self.log_description = False  # log sample descriptions
        self.all_objects_folder = all_objects_folder



    def _calculate_normalized_l1_error(self, gt_objects_data, pred_objects_data, gt_human_meshes=None, distance_threshold=0.1):
        """
        Calculate normalized L1 error for object positions.
        Normalized by the scene size (diagonal of scene bounding box).
        If label is wrong (object not predicted or wrongly predicted), set L1 to 1.0.
        
        Args:
            gt_objects_data: List of dicts with GT object data, each containing:
                'name': str, 'verts': [n_frame, n_point, 3], 'faces': [n_face, 3]
            pred_objects_data: List of dicts with predicted object data, same structure
            gt_human_meshes: Optional list of trimesh objects representing GT human meshes
            distance_threshold: Threshold for normalized L1 distance. Objects with correct label
                                and distance < threshold are considered True Positives.
        
        Returns:
            dict: Mapping from GT object name to normalized L1 error (1.0 if label is wrong)
            float: Precision (for label matching)
            float: Recall (for label matching)
            float: Precision@threshold (based on distance threshold)
            float: Recall@threshold (based on distance threshold)
        """
        # Create a mapping from object name to predicted object data
        pred_objects_dict = {obj['name']: obj for obj in pred_objects_data}
        
        print("Calculating normalized L1 errors...")
        print(f"GT objects: {[obj['name'] for obj in gt_objects_data]}")
        print(f"Predicted objects: {[obj['name'] for obj in pred_objects_data]}")
        print(f"Distance threshold: {distance_threshold}")
        
        # Calculate scene size for normalization (including both objects and human mesh)
        all_verts = []
        # Add object vertices
        for obj in gt_objects_data:
            if obj['name'] != 'ground':
                all_verts.append(obj['verts'][0])  # [n_point, 3]
        
        # Add human mesh vertices
        if gt_human_meshes is not None:
            for mesh in gt_human_meshes:
                if hasattr(mesh, 'vertices'):
                    all_verts.append(mesh.vertices)  # [n_point, 3]
        
        if len(all_verts) > 0:
            all_verts = np.concatenate(all_verts, axis=0)  # [total_points, 3]
            scene_min = np.min(all_verts, axis=0)
            scene_max = np.max(all_verts, axis=0)
            scene_size = np.linalg.norm(scene_max - scene_min)  # Diagonal of scene bbox
        else:
            scene_size = 1.0  # Default if no objects
        
        print(f"Scene size (diagonal): {scene_size:.4f}")
        
        # Calculate recall and precision for object label IDs (original metrics)
        gt_object_names = set(obj['name'] for obj in gt_objects_data if obj['name'] != 'ground')
        pred_object_names = set(obj['name'] for obj in pred_objects_data if obj['name'] != 'ground')
        
        # Calculate metrics (label-based only)
        true_positives_label = len(gt_object_names & pred_object_names)
        false_positives_label = len(pred_object_names - gt_object_names)
        false_negatives_label = len(gt_object_names - pred_object_names)
        
        # Calculate precision and recall (label-based)
        precision = true_positives_label / (true_positives_label + false_positives_label) if (true_positives_label + false_positives_label) > 0 else 0.0
        recall = true_positives_label / (true_positives_label + false_negatives_label) if (true_positives_label + false_negatives_label) > 0 else 0.0
        
        # Calculate threshold-based metrics
        # True Positive: correct label AND distance < threshold
        # False Positive: wrong label OR (correct label BUT distance >= threshold)
        # False Negative: GT object not predicted OR (predicted but distance >= threshold)
        true_positives_thresh = 0
        false_positives_thresh = 0
        false_negatives_thresh = 0
        
        l1_errors = {}
        gt_processed = set()  # Track which GT objects have been processed
        
        # Process GT objects to calculate threshold-based metrics
        for gt_obj in gt_objects_data:
            obj_name = gt_obj['name']
            
            # Skip ground plane as it's not a meaningful object
            if obj_name == 'ground':
                continue

            gt_processed.add(obj_name)

            # Get GT object center (average of all vertices)
            gt_verts = gt_obj['verts'][0]  # [n_point, 3]
            gt_center = np.mean(gt_verts, axis=0)  # [3]
            
            # Check if this object was predicted
            if obj_name not in pred_objects_dict:
                # Label is wrong (missing prediction) - set to 1.0
                l1_errors[obj_name] = 1.0
                false_negatives_thresh += 1  # GT object not predicted
                continue
            
            # Get predicted object center
            pred_obj = pred_objects_dict[obj_name]
            pred_verts = pred_obj['verts'][0]  # [n_point, 3]
            pred_center = np.mean(pred_verts, axis=0)  # [3]
            
            # Calculate L1 distance (sum of absolute differences)
            l1_distance = np.sum(np.abs(gt_center - pred_center))
            
            # Normalize by scene size
            normalized_l1 = l1_distance / scene_size if scene_size > 0 else 0.0
            
            # Clamp to [0, 1] range (in case error exceeds scene size)
            normalized_l1 = min(normalized_l1, 1.0)
            
            l1_errors[obj_name] = normalized_l1
            
            # Check threshold for this GT object
            if normalized_l1 < distance_threshold:
                true_positives_thresh += 1  # Correct label AND distance < threshold
            else:
                false_negatives_thresh += 1  # Correct label BUT distance >= threshold
                false_positives_thresh += 1  # Also counts as FP for precision (prediction not good enough)
        
        # Process predicted objects that don't exist in GT (false positives)
        for pred_obj in pred_objects_data:
            obj_name = pred_obj['name']
            
            # Skip ground plane
            if obj_name == 'ground':
                continue
            
            # If this predicted object doesn't exist in GT, it's a false positive
            if obj_name not in gt_object_names:
                false_positives_thresh += 1
        
        # Calculate precision@threshold and recall@threshold
        precision_at_thresh = true_positives_thresh / (true_positives_thresh + false_positives_thresh) if (true_positives_thresh + false_positives_thresh) > 0 else 0.0
        recall_at_thresh = true_positives_thresh / (true_positives_thresh + false_negatives_thresh) if (true_positives_thresh + false_negatives_thresh) > 0 else 0.0
        
        return l1_errors, precision, recall, precision_at_thresh, recall_at_thresh

    @torch.no_grad()
    def __call__(self, seq_id) -> None:

        # Get the canonical pelvis position (without any transformations)
        default_smplx_output = self.smplx_model()
        rest_pelvis = default_smplx_output.joints[0, 0].detach().cpu().numpy()
        # mixamo_model_folder = '/home/tianhang/code/smplx/mixamo'
        # mean_coord = np.load(os.path.join(mixamo_model_folder, 'mixamo_template_mean_coord.npy'))
        # scale_factors = np.load(os.path.join(mixamo_model_folder, 'mixamo_template_scale_factors.npy'))
        # translation = np.load(os.path.join(mixamo_model_folder, 'mixamo_template_translation.npy'))

        # TODO: some magic numbers here, not elegant but works for now
        # mean_coord = np.array([1.6916836e-04, 1.0473369e+00, 4.3568998e-03], dtype=np.float32)
        scale_factors = np.array([1.0379714, 1.0817566, 1.0400887], dtype=np.float32)
        # translation = np.array([-3.8248476e-05, -1.7167634e-01,  1.0316576e-03], dtype=np.float32)
        # object_offset = np.array([-2.0601762e-04, -1.2060384e+00, -3.3650058e-03], dtype=np.float32)

        """
        dict_keys(['imu_traj', 'motion_smpl', 'text', 'objects'])

        imu_traj: [n_frame, n_imu, 6]
        motion_smpl: [n_frame, 75]  # global_orient(3), body_pose(63), global_transl(3)
        text: list of str
        objects: dict of object_name -> object_data
            {object_name: data: [7] (quaternion(4), transl(3))}
        """
        #ely
        # 0, 0,

        viz_baseline = False

        old = False
        dataset_name = 'humoto'
        # predict_path = '/home/tianhang/Documents/seq/id_{}_step.npy'.format(seq_id)
        # predict_path = '/home/tianhang/Documents/seq/humoto/id_{}_step_0.npy'.format(seq_id)
        # predict_path = f'/home/tianhang/Documents/seq/{dataset_name}/viz_test_generate_number_merged/id_{seq_id}_step_0.npy'
        # predict_path = f'/home/tianhang/Documents/seq/humoto/motion_only/viz_test_generate_number_shifted_2/id_{seq_id}_step_0.npy'
        predict_path = '/home/tianhang/Desktop/viz_test_generate_number_shifted_0/id_{}_step_0.npy'.format(int(seq_id))
        predict_data = np.load(predict_path, allow_pickle=True).item()
        # predict_data['pred'] = predict_data['gt'] # FIXME: remove this
        original_mpjpe = compute_mpjpe(predict_data)[0]
        mpjpe = original_mpjpe

        # predict_path_shifted = '/home/tianhang/Documents/seq/id_{}_step_shifted.npy'.format(seq_id)
        # if os.path.exists(predict_path_shifted):
        #     predict_data_shifted = np.load(predict_path_shifted, allow_pickle=True).item()
        #     cut_len = min(predict_data['pred']['pose'].shape[0]-2, predict_data_shifted['pred']['pose'].shape[0])
        #     predict_data['pred']['pose'][2:2+cut_len] = (predict_data['pred']['pose'][2:2+cut_len] + predict_data_shifted['pred']['pose'][:cut_len]) / 2.0
        #     mpjpe = compute_mpjpe(predict_data)[0]

        pred_text = predict_data['pred']['description']
        gt_text = predict_data['gt']['description']
        print(f'GT description: {gt_text}')
        print(f'Pred description: {pred_text}')

        print(f'original MPJPE: {original_mpjpe:.2f} mm, merged MPJPE: {mpjpe:.2f} mm, difference: {mpjpe - original_mpjpe:.2f} mm')

        for obj_name, obj_data in predict_data['pred']['objects'].items():
            if isinstance(obj_data['rot'], torch.Tensor):
                predict_data['pred']['objects'][obj_name]['rot'] = predict_data['pred']['objects'][obj_name]['rot'].float().detach().cpu().numpy()
            if isinstance(obj_data['transl'], torch.Tensor):
                predict_data['pred']['objects'][obj_name]['transl'] = predict_data['pred']['objects'][obj_name]['transl'].float().detach().cpu().numpy()

        gt_sample = predict_data['gt']
        gt_smpl_transl = gt_sample['transl'].reshape(-1, 3)
        gt_smpl_orient = gt_sample['orient'].reshape(-1, 3)
        gt_smpl_pose = gt_sample['pose'].reshape(-1, 21, 3)
        n_time = gt_smpl_orient.shape[0]

        pred_sample = predict_data['pred']
        if old:
            pred_sample['transl'] = pred_sample['transl'] - rest_pelvis
        pred_smpl_transl = pred_sample['transl'].reshape(-1, 3)
        pred_smpl_orient = pred_sample['orient'].reshape(-1, 3)
        pred_smpl_pose = pred_sample['pose'].reshape(-1, 21, 3)


        pred_smpl_pose = convert_rotation(torch.from_numpy(pred_smpl_pose).float(), 'aa', '6d').reshape(-1, 21*6)
        # pred_smpl_pose = reshape_and_smooth_timeseries(pred_smpl_pose.float().numpy(), window_size=2)
        pred_smpl_pose = smooth_time_series(pred_smpl_pose.float(), kernel_size=5).reshape(-1, 21*6).numpy()
        pred_smpl_pose = convert_rotation(torch.from_numpy(pred_smpl_pose).float().reshape(-1, 6), '6d', 'aa').reshape(-1, 21, 3).cpu().numpy()    
        pred_sample['pose'] = pred_smpl_pose

        if 'ground' not in pred_sample['objects']:
            pred_sample['objects']['ground'] = gt_sample['objects']['ground']  # Use GT ground if not predicted, to avoid NaN errors and for better visualization
        ground_height = pred_sample['objects']['ground']['transl'][1] # y of the ground plane
        pred_ground_vec = np.array([0, ground_height,  0]) # Just for visualization!
        viz_ground_vec = pred_ground_vec

        if 'ground' in gt_sample['objects']:
            ground_height = gt_sample['objects']['ground']['transl'][1] # y of the ground plane
            if isinstance(ground_height, torch.Tensor):
                ground_height = ground_height.detach().cpu().numpy()
            gt_ground_vec = np.array([0, ground_height,  0]) # Just for visualization!
            viz_ground_vec = gt_ground_vec
        
        if viz_baseline:
            baseline_path = f'/home/tianhang/Documents/seq/{dataset_name}/6/sample_seq_{str(seq_id).zfill(4)}.npy'
            baseline_data = np.load(baseline_path, allow_pickle=True).item()
            baseline_sample = baseline_data.get('pred', baseline_data)
            baseline_smpl_transl = baseline_sample['transl'].reshape(-1, 3)
            baseline_smpl_orient = baseline_sample['orient'].reshape(-1, 3)
            baseline_smpl_pose = baseline_sample['pose'].reshape(-1, 21, 3)
        
        sample_gt = {
            'transl': gt_smpl_transl - gt_ground_vec,
            'orient': gt_smpl_orient,
            'pose': gt_smpl_pose
        }
        sample_pred = {
            'transl': pred_smpl_transl - pred_ground_vec,
            'orient': pred_smpl_orient,
            'pose': pred_smpl_pose
        }
        if viz_baseline:
            # Align baseline with ground (use GT ground for consistency)
            sample_baseline = {
                'transl': baseline_smpl_transl - gt_ground_vec,
                'orient': baseline_smpl_orient,
                'pose': baseline_smpl_pose
            }
        seq_len = n_time

        gt_text = gt_sample['description'] # list of strings
        pred_text = pred_sample['description'] # single string

        # fig = plt.figure(figsize=(10, 10))
        # ax = fig.add_subplot(111)
        # key = 'pose'
        # time = np.arange(sample_gt[key].shape[0]) / self.sample_fps
        # ax.plot(time, sample_gt[key][:, 0], label='X', color='r')
        # ax.plot(time, sample_gt[key][:, 1], label='Y', color='g')
        # ax.plot(time, sample_gt[key][:, 2], label='Z', color='b')
        # ax.plot(time, sample_pred[key][:, 0], label='X_pred', color='r', linestyle='dashed')
        # ax.plot(time, sample_pred[key][:, 1], label='Y_pred', color='g', linestyle='dashed')
        # ax.plot(time, sample_pred[key][:, 2], label='Z_pred', color='b', linestyle='dashed')
        # ax.set_xlabel('Time (s)')
        # if key == 'transl':
        #     ax.set_ylabel('Position (m)')
        #     ax.set_ylim([-0.15, 0.85])
        #     ax.set_title('Trajectory')
        # elif key == 'orient':
        #     ax.set_ylabel('Orientation (axis-angle)')
        #     ax.set_ylim([-1.5, 1.5])
        #     ax.set_title('Pelvis Orientation (axis-angle)')
        # elif key == 'pose':
        #     ax.set_ylabel('Pose (axis-angle)')
        #     ax.set_ylim([-1.5, 1.5])
        #     ax.set_title('Pelvis Pose (axis-angle)')
        # ax.legend()
        # plt.show()


        # mixamo_vertices = fitted_smplx['mixamo_vertices'] + transl[:, None] # [n_frame, n_point, 3]
        # mixamo_faces = fitted_smplx['mixamo_faces'] # [n_face, 3]

        # load object
        gt_objects = gt_sample['objects']
        objects_data = []
        for obj_name, obj_data in gt_objects.items():
            obj_name = obj_name.split('.')[0]
            # obj_name: std
            # obj_data: [7] # quaternion(4) + transl(3)
            if obj_name != 'ground': 
                object_mesh_path = os.path.join(self.all_objects_folder, obj_name, f'{obj_name}.obj')
                if not os.path.exists(object_mesh_path):
                    raise FileNotFoundError(f'Object mesh not found: {object_mesh_path}')
                mesh = trimesh.load(object_mesh_path, process=False)
                if isinstance(mesh, trimesh.Scene):
                    # Convert all geometry in the scene into a single mesh
                    mesh = trimesh.util.concatenate(mesh.dump())
                else:
                    mesh = mesh  
                vertices = mesh.vertices  # [n_point, 3]
                faces = mesh.faces  # [n_face, 3]

            if isinstance(obj_data['rot'], torch.Tensor):
                obj_data['rot'] = obj_data['rot'].detach().cpu().numpy()
            rot_mat = convert_rotation(torch.from_numpy(obj_data['rot']).float(), '6d', 'mat').numpy()  # [3, 3]
            transl_vec = obj_data['transl']  # [3]
            
            # apply rotation and translation
            if obj_name == 'ground':
                size = 10
                y = 0
                vertices = np.array([
                    [-size, y, -size],
                    [ size, y, -size],
                    [ size, y,  size],
                    [-size, y,  size],
                ])  # [4, 3]
                faces = np.array([
                    [0, 1, 2],
                    [0, 2, 3],
                ])  # [2, 3]
                obj_verts = ((vertices + transl_vec) - gt_ground_vec)[None, :, :]  # [1, n_point, 3]
            else:
                # obj_verts = (((vertices @ rot_mat.T + transl_vec))  * scale_factors - gt_ground_vec)[None, :, :]  # [1, n_point, 3]
                obj_verts = ((vertices @ rot_mat.T + transl_vec) - gt_ground_vec)[None, :, :]  # [1, n_point, 3]
            
            obj_verts = np.repeat(obj_verts, n_time, axis=0)  # [n_frame, n_point, 3] static object for now
            if obj_name != 'ground':
                obj_verts = obj_verts  # for visualization, align with human origin
            objects_data.append({
                'name': obj_name,
                'verts': obj_verts, # for visualization, align with human origin
                'faces': faces,  # [1, n_face, 3]
                'color': [100, 255, 100],
            })
            _=1

        pred_objects = pred_sample['objects']
        pred_objects_data = []
        for obj_name, obj_data in pred_objects.items():
            obj_name = obj_name.split('.')[0]
            # obj_name: std
            # obj_data: [7] # quaternion(4) + transl(3)
            if obj_name != 'ground': 
                object_mesh_path = os.path.join(self.all_objects_folder, obj_name, f'{obj_name}.obj')
                if not os.path.exists(object_mesh_path):
                    raise FileNotFoundError(f'Object mesh not found: {object_mesh_path}')
                mesh = trimesh.load(object_mesh_path, process=False)
                if isinstance(mesh, trimesh.Scene):
                    # Convert all geometry in the scene into a single mesh
                    mesh = trimesh.util.concatenate(mesh.dump())
                else:
                    mesh = mesh  
                vertices = mesh.vertices  # [n_point, 3]
                faces = mesh.faces  # [n_face, 3]

            if isinstance(obj_data['rot'], torch.Tensor):
                obj_data['rot'] = obj_data['rot'].detach().cpu().numpy()
            if isinstance(obj_data['transl'], torch.Tensor):
                obj_data['transl'] = obj_data['transl'].detach().cpu().numpy()
            rot_mat = convert_rotation(torch.from_numpy(obj_data['rot']).float(), '6d', 'mat').numpy()  # [3, 3]
            transl_vec = obj_data['transl']  # [3]
            
            # apply rotation and translation
            if obj_name == 'ground':
                size = 10
                y = 0
                vertices = np.array([
                    [-size, y, -size],
                    [ size, y, -size],
                    [ size, y,  size],
                    [-size, y,  size],
                ])  # [4, 3]
                faces = np.array([
                    [0, 1, 2],
                    [0, 2, 3],
                ])  # [2, 3]
                obj_verts = ((vertices + transl_vec) - pred_ground_vec)[None, :, :]  # [1, n_point, 3]
            else:
                obj_verts = (((vertices @ rot_mat.T + transl_vec) - pred_ground_vec)  )[None, :, :]  # [1, n_point, 3]
            if old:
                obj_verts = obj_verts - rest_pelvis
            obj_verts = np.repeat(obj_verts, n_time, axis=0)  # [n_frame, n_point, 3] static object for now
            if obj_name != 'ground':
                obj_verts = obj_verts  # for visualization, align with human origin
            pred_objects_data.append({
                'name': obj_name,
                'verts': obj_verts, # for visualization, align with human origin
                'faces': faces,  # [1, n_face, 3]
                'color': [255, 150, 150],  # Light red/pink to distinguish from GT objects
            })
        


        # object_path = f'/media/tianhang/Getea/humoto_0805_render_canonical/{seq_name}/gt_objects.pkl'
        # with open(object_path, 'rb') as f:
        #     objects = pickle.load(f)
        # # structure objects into a list of dicts for easy per-frame logging
        # objects_data = []
        # for obj_name, obj_data in objects.items():
        #     # obj_data: [verts[n_frame, n_point, 3], faces[n_face, 3], color[3]]
        #     obj_verts = obj_data[0].detach().cpu().numpy()
        #     # apply inverse scale and translation to the vertices
        #     # obj_verts = (obj_verts - mean_coord + translation / scale_factors) * scale_factors
        #     obj_verts = (obj_verts ) * scale_factors
        #     # obj_verts = (obj_verts_canonical @ pred_rot + pred_transl - mean_coord + translation / scale_factors) * scale_factors
        #     obj_faces = obj_data[1].detach().cpu().numpy()
        #     obj_color = [100, 255, 100]
        #     objects_data.append({
        #         'name': obj_name,
        #         'verts': obj_verts,
        #         'faces': obj_faces,
        #         'color': obj_color,
        #     })


        # sample_gt = sample_pred.copy()
        # seqlen = sample_pred['transl'].shape[0]  # Use the length of the transl sequence as seqlen
        # sample_pred['transl'] = sample_pred['transl'][:seqlen] 
        # # sample_pred['transl'] = sample_gt['transl']
        # sample_pred['orient'] = sample_pred['orient'][:seqlen]
        # sample_pred['pose'] = sample_pred['pose'][:seqlen]

        # Load SMPL-X meshes for all samples
        smplx_meshes_list = []
        smplx_meshes_list_quant = []
        smplx_meshes_list_baseline = []
        smplx_trajs_list = []
        smplx_trajs_list_quant = []
        # smplx_orient_list = []
        
        meshes = self._load_smplx_meshes(sample_gt)
        smplx_meshes_list.append(meshes)
        
        # Trajectories from simulating the SMPL-X model
        # trajs_sim = self._compute_pelvis_trajectories(sample)

        # Trajectories from applying transformations to the pelvis original position
        meshes_quant = self._load_smplx_meshes(sample_pred)
        smplx_meshes_list_quant.append(meshes_quant)
        
        if viz_baseline:
            meshes_baseline = self._load_smplx_meshes(sample_baseline)
            smplx_meshes_list_baseline.append(meshes_baseline)
        
        smplx_trajs_list.append(sample_gt['transl'])
        smplx_trajs_list_quant.append(sample_pred['transl'])


        # TODO: debug to log pelvis orientations
        # self._log_orientations(smplx_orient_list)
        
        # Visualize trajectories
        # texts = [sample["description"] for sample in samples] if self.log_description else None
        self._log_trajectories(smplx_trajs_list, labels=None, is_pred=False)
        self._log_trajectories(smplx_trajs_list_quant, labels=None, is_pred=True)
        # print(f'gt description: {sample_gt["description"]}\n')
        # print(f'pred description: {sample_pred["description"]}\n')
            
        # Determine sequence length (max among all samples)
        # seqlen = max([len(meshes) for meshes in smplx_meshes_list])
        
        # Log sequences frame by frame
        viz_mesh = True
        dt = 1.0 / self.sample_fps
        for frame_idx in tqdm(range(seq_len)):
            break
            rr.set_time_sequence("frames", frame_idx)
            rr.set_time_seconds("sensor_time", frame_idx * dt)
            

            if viz_mesh:
                # Log SMPL-X meshes for this frame
                meshes_at_frame = []
                for meshes in smplx_meshes_list:
                    if frame_idx < len(meshes):
                        meshes_at_frame.append(meshes[frame_idx])
                    else:
                        meshes_at_frame.append(meshes[-1])  # Use last frame if out of range
                self._log_smplx_meshes(meshes_at_frame)

                meshes_at_frame_quant = []
                for meshes in smplx_meshes_list_quant:
                    if frame_idx < len(meshes):
                        meshes_at_frame_quant.append(meshes[frame_idx])
                    else:
                        meshes_at_frame_quant.append(meshes[-1])
                self._log_smplx_meshes(meshes_at_frame_quant, is_pred=True)


            # Log variable number of objects for this frame (aligned with human origin)
            if len(objects_data) > 0 and frame_idx==0:
                for obj in objects_data:
                    obj_len = obj['verts'].shape[0]
                    frame_id = frame_idx if frame_idx < obj_len else obj_len - 1
                    # verts_f = obj['verts'][frame_id] - init_global_transl[None, :]
                    verts_f = obj['verts'][frame_id]
                    faces = obj['faces']
                    color =np.array(obj['color'])
                    
                    # Create trimesh object to compute vertex normals
                    obj_mesh = trimesh.Trimesh(
                        vertices=verts_f,
                        faces=faces,
                        process=False
                    )
                    
                    ep = f"world/obj/{obj['name']}"
                    rr.log(
                        ep,
                        rr.Mesh3D(
                            vertex_positions=obj_mesh.vertices,
                            vertex_normals=obj_mesh.vertex_normals,
                            triangle_indices=obj_mesh.faces,
                            vertex_colors=color,
                        ),
                        static=False,
                    )
                    self._epaths_3d.add(ep)

            # Log predicted objects for this frame (aligned with human origin)
            if len(pred_objects_data) > 0 and frame_idx==0: 
                for obj in pred_objects_data:
                    obj_len = obj['verts'].shape[0]
                    frame_id = frame_idx if frame_idx < obj_len else obj_len - 1
                    # verts_f = obj['verts'][frame_id] - init_global_transl[None, :]
                    verts_f = obj['verts'][frame_id]
                    faces = obj['faces']
                    color = np.array(obj['color'])
                    
                    # Create trimesh object to compute vertex normals
                    obj_mesh = trimesh.Trimesh(
                        vertices=verts_f,
                        faces=faces,
                        process=False
                    )
                    
                    ep = f"world/obj_pred/{obj['name']}"
                    rr.log(
                        ep,
                        rr.Mesh3D(
                            vertex_positions=obj_mesh.vertices,
                            vertex_normals=obj_mesh.vertex_normals,
                            triangle_indices=obj_mesh.faces,
                            vertex_colors=color,
                        ),
                        static=False,
                    )
                    self._epaths_3d.add(ep)

            # Log text labels near the human
            if viz_mesh:
                self._log_text_labels(
                    meshes_at_frame, 
                    meshes_at_frame_quant,
                    gt_text, 
                    pred_text, 
                    frame_idx
                )

            # Log IMU data for this frame
            # if log_imus:
            #     self._log_imu_transforms(sample_input, frame_idx, sample_idx=0)


        # save all info into 1 file
        """
        sample_pred: {'transl': [n_frame, 3], 'orient': [n_frame, 3], 'pose': [n_frame, 21, 3]}
        sample_gt: {'transl': [n_frame, 3], 'orient': [n_frame, 3], 'pose': [n_frame, 21, 3]}
        objects_data: list of dicts, each dict contains 'name', 'verts', 'faces', 'color'
            e.g. {'name': 'bowl', 'verts': [n_frame, n_point, 3], 'faces': [n_face, 3], 'color': [3]}
        smplx_meshes_vertices: [n_frame, n_point, 3]
        smplx_meshes_faces: [n_face, 3]
        """

        def extract_object_data(objects_data):
            """Extract numpy arrays from object data, removing trimesh dependencies"""
            extracted = []
            for obj_data in objects_data:
                extracted_obj = {
                    'name': obj_data['name'],
                    'verts': np.array(obj_data['verts']).astype(np.float32),  # Ensure it's numpy array
                    'faces': np.array(obj_data['faces']).astype(np.int32)     # Ensure it's numpy array
                }
                extracted.append(extracted_obj)
            return extracted

        objects_data_gt_clean = extract_object_data(objects_data)
        objects_data_pred_clean = extract_object_data(pred_objects_data)

        # Calculate 3D bounding box IoU for each GT object
        bbox_ious, precision, recall = self._calculate_bbox_ious(objects_data, pred_objects_data)
        print("\n=== 3D Bounding Box IoU Results ===")
        for obj_name, iou in bbox_ious.items():
            print(f"{obj_name}: IoU = {iou:.4f}")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}")
        mean_iou = np.mean(list(bbox_ious.values())) if bbox_ious else 0.0
        print(f"Mean IoU: {mean_iou:.4f}\n")

        # Calculate normalized L1 error for predicted objects
        gt_human_meshes = smplx_meshes_list[0] if smplx_meshes_list else None
        l1_errors, l1_precision, l1_recall, l1_precision_at_thresh, l1_recall_at_thresh = self._calculate_normalized_l1_error(
            objects_data, pred_objects_data, gt_human_meshes, distance_threshold=self.distance_threshold
        )
        print("\n=== Normalized L1 Error Results ===")
        for obj_name, l1_error in l1_errors.items():
            print(f"{obj_name}: Normalized L1 = {l1_error:.4f}")
        print(f"Precision (label-based): {l1_precision:.4f}, Recall (label-based): {l1_recall:.4f}")
        print(f"Precision@{self.distance_threshold}: {l1_precision_at_thresh:.4f}, Recall@{self.distance_threshold}: {l1_recall_at_thresh:.4f}")
        mean_l1 = np.mean(list(l1_errors.values())) if l1_errors else 0.0
        print(f"Mean Normalized L1: {mean_l1:.4f}\n")

        return mean_iou, precision, recall, mean_l1, l1_precision, l1_recall, l1_precision_at_thresh, l1_recall_at_thresh

        seq_name = f'humoto_{seq_id}'
        save_path = f'/home/tianhang/Desktop/CVPR IMU human/imu_human_viz_for_blender/{seq_name}/viz_info.pkl'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        pred_smplx_meshes_vertices = np.stack([np.array(mesh.vertices).astype(np.float32) for mesh in smplx_meshes_list_quant[0]], axis=0) # [n_frame, n_point, 3]
        gt_smplx_meshes_vertices = np.stack([np.array(mesh.vertices).astype(np.float32) for mesh in smplx_meshes_list[0]], axis=0) # [n_frame, n_point, 3]
        human_faces = np.array(self.smplx_model.faces).astype(np.int32) # [n_face, 3]
        if viz_baseline:
            baseline_smplx_meshes_vertices = np.stack([np.array(mesh.vertices).astype(np.float32) for mesh in smplx_meshes_list_baseline[0]], axis=0) # [n_frame, n_point, 3]
        else:
            baseline_smplx_meshes_vertices = pred_smplx_meshes_vertices
        viz_info = {
            'seq_name': seq_name,
            'sample_pred': sample_pred,
            'sample_gt': sample_gt,
            'objects_data_gt': objects_data_gt_clean,
            'objects_data_pred': objects_data_pred_clean,
            'pred_smplx_meshes_vertices': pred_smplx_meshes_vertices, # [n_frame, n_point, 3]
            'gt_smplx_meshes_vertices': gt_smplx_meshes_vertices, # [n_frame, n_point, 3]
            'baseline_smplx_meshes_vertices': baseline_smplx_meshes_vertices, # [n_frame, n_point, 3]
            'human_faces': human_faces, # [n_face, 3]
        }
        pickle.dump(viz_info, open(save_path, 'wb'))
        print(f'Visualization info saved to {save_path}')
        exit(0)

    def _calculate_bbox_ious(self, gt_objects_data, pred_objects_data):
        """
        Calculate 3D bounding box IoU for each ground-truth object.
        
        Args:
            gt_objects_data: List of dicts with GT object data, each containing:
                'name': str, 'verts': [n_frame, n_point, 3], 'faces': [n_face, 3]
            pred_objects_data: List of dicts with predicted object data, same structure
        
        Returns:
            dict: Mapping from GT object name to IoU value (0.0 if object not predicted)
        """
        # Create a mapping from object name to predicted object data
        pred_objects_dict = {obj['name']: obj for obj in pred_objects_data}

        print("Calculating 3D bounding box IoUs...")
        print(f"GT objects: {[obj['name'] for obj in gt_objects_data]}")
        print(f"Predicted objects: {[obj['name'] for obj in pred_objects_data]}")

        # Calculate recall and precision for object label IDs
        # Extract object names (excluding 'ground' as it's not a meaningful object)
        gt_object_names = set(obj['name'] for obj in gt_objects_data if obj['name'] != 'ground')
        pred_object_names = set(obj['name'] for obj in pred_objects_data if obj['name'] != 'ground')
        
        # Calculate metrics
        true_positives = len(gt_object_names & pred_object_names)  # Objects in both GT and predictions
        false_positives = len(pred_object_names - gt_object_names)  # Objects predicted but not in GT
        false_negatives = len(gt_object_names - pred_object_names)  # Objects in GT but not predicted
        
        # Calculate precision and recall
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        
        print(f"\n=== Object Label ID Metrics ===")
        print(f"True Positives (TP): {true_positives}")
        print(f"False Positives (FP): {false_positives}")
        print(f"False Negatives (FN): {false_negatives}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        if precision + recall > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
            print(f"F1 Score: {f1_score:.4f}")
        print("=" * 30 + "\n")
        
        bbox_ious = {}
        
        for gt_obj in gt_objects_data:
            obj_name = gt_obj['name']
            
            # Skip ground plane as it's not a meaningful object for IoU
            if obj_name == 'ground':
                continue
            
            # Get vertices for GT object (use first frame for static objects)
            gt_verts = gt_obj['verts'][0]  # [n_point, 3]
            
            # Compute GT bounding box (axis-aligned)
            gt_min = np.min(gt_verts, axis=0)  # [3]
            gt_max = np.max(gt_verts, axis=0)  # [3]
            gt_size = gt_max - gt_min  # [3]
            gt_volume = np.prod(gt_size)  # scalar
            
            # Check if this object was predicted
            if obj_name not in pred_objects_dict:
                bbox_ious[obj_name] = 0.0
                continue
            
            # Get predicted object vertices
            pred_obj = pred_objects_dict[obj_name]
            pred_verts = pred_obj['verts'][0]  # [n_point, 3]
            
            # Compute predicted bounding box (axis-aligned)
            pred_min = np.min(pred_verts, axis=0)  # [3]
            pred_max = np.max(pred_verts, axis=0)  # [3]
            pred_size = pred_max - pred_min  # [3]
            pred_volume = np.prod(pred_size)  # scalar
            
            # Calculate intersection bounding box
            inter_min = np.maximum(gt_min, pred_min)  # [3]
            inter_max = np.minimum(gt_max, pred_max)  # [3]
            inter_size = np.maximum(0, inter_max - inter_min)  # [3] (clamp to 0 if no overlap)
            inter_volume = np.prod(inter_size)  # scalar
            
            # Calculate union volume
            union_volume = gt_volume + pred_volume - inter_volume
            
            # Calculate IoU
            if union_volume > 0:
                iou = inter_volume / union_volume
            else:
                iou = 0.0
            
            bbox_ious[obj_name] = iou
        
        return bbox_ious, precision, recall

    def _log_imu_transforms(self, sample, frame_idx, sample_idx) -> None:
        """
        Log IMU transformations for the current frame index
        """
        devices = ["airpod", "watch", "iphone"]
        device_colors = {
            "airpod": [0, 191, 255],  # Deep Sky Blue
            "watch": [255, 69, 0],    # Orangered
            "iphone": [50, 205, 50]   # Lime Green
        }

        axis_scale = 0.1  # Scale for the coordinate axes

        for device in devices:
            trans_key = f'imu_trans_{device}'
            if trans_key not in sample:
                continue

            device_color = device_colors[device]
            imu_trans = sample[trans_key]  # Shape: [seq_len, 6]

            # log full trajectory path only once (when first frame is processed)
            if frame_idx == 0:
                positions = imu_trans[:, 3:6]
                rr.log(
                    f"world/imu_traj_{sample_idx}_{device}",
                    rr.LineStrips3D(
                        positions,
                        colors=device_color,
                        radii=self.line_radii * 0.8,
                        labels=[device]
                    ),
                    static=True,
                )
            
            if frame_idx < len(imu_trans):
                frame_data = imu_trans[frame_idx]
            else:
                frame_data = imu_trans[-1]

            rot_vec = frame_data[0:3]
            trans_vec = frame_data[3:6]
            rot_matrix = R.from_rotvec(rot_vec).as_matrix()

            # calculate the local coordinate axes
            origin = trans_vec
            x_axis = origin + rot_matrix[:, 0] * axis_scale
            y_axis = origin + rot_matrix[:, 1] * axis_scale
            z_axis = origin + rot_matrix[:, 2] * axis_scale

            ep_base = f"world/imu_local_rot_{sample_idx}_{device}"

            # Log X-axis (red)
            rr.log(
                f"{ep_base}/x_axis",
                rr.LineStrips3D(
                    [origin, x_axis],
                    colors=[255, 0, 0],
                    radii=self.line_radii
                ),
                static=False,
            )
            
            # Log Y-axis (green)
            rr.log(
                f"{ep_base}/y_axis",
                rr.LineStrips3D(
                    [origin, y_axis],
                    colors=[0, 255, 0],
                    radii=self.line_radii
                ),
                static=False,
            )
            
            # Log Z-axis (blue)
            rr.log(
                f"{ep_base}/z_axis",
                rr.LineStrips3D(
                    [origin, z_axis],
                    colors=[0, 0, 255],
                    radii=self.line_radii
                ),
                static=False,
            )


    def _log_coordinate_axes(self, scale=0.5) -> None:
        """Add coordinate axes to visualize the world coordinate system"""
        # Define the axes
        origin = np.array([0, 0, 0])
        x_axis = np.array([scale, 0, 0]) 
        y_axis = np.array([0, scale, 0])
        z_axis = np.array([0, 0, scale])
        
        # Log X-axis (red)
        rr.log("world/axes/x", 
            rr.LineStrips3D(
                [origin, x_axis],
                colors=[255, 0, 0],
                radii=self.line_radii*1.5,
                labels=["X"]
            ),
            static=True)
        
        # Log Y-axis (green)
        rr.log("world/axes/y", 
            rr.LineStrips3D(
                [origin, y_axis],
                colors=[0, 255, 0],
                radii=self.line_radii*1.5,
                labels=["Y"]
            ),
            static=True)
        
        # Log Z-axis (blue) - the first frame's forward direction
        rr.log("world/axes/z", 
            rr.LineStrips3D(
                [origin, z_axis],
                colors=[0, 0, 255], 
                radii=self.line_radii*1.5,
                labels=["Z (forward)"]
            ),
            static=True)


    def _remove_wall(self, mesh) -> trimesh.Trimesh:
        """Remove the wall from the mesh"""
        vs = mesh.vertices
        v_min = np.min(vs, axis=0)
        v_max = np.max(vs, axis=0)
        # print(v_min, v_max)
        valid = (vs[:, 1] < v_max[1] - 0.1)
        fs = mesh.faces
        valid_fs = valid[fs.reshape(-1)].reshape(fs.shape[0], 3).min(axis=1)
        mesh.faces = fs[valid_fs]
        return mesh

    def _load_smplx_meshes(self, sample):
        """Convert SMPL-X parameters to meshes"""
        orients = sample['orient']
        poses = sample['pose']
        transls = sample['transl']
        
        meshes = []
        for i, (orient, pose, transl) in enumerate(zip(orients, poses, transls)):
            
            if i >= self.seq_len_limit:
                break

            smplx_output = self.smplx_model(
                global_orient=torch.tensor(orient[None]).float(),
                body_pose=torch.tensor(pose[None]).float(),
                transl=torch.tensor(transl[None]).float(),
            )
            mesh = trimesh.Trimesh(
                vertices=smplx_output.vertices.detach().cpu().numpy()[0], 
                faces=self.smplx_model.faces, 
                process=False
            )
            # set to orange
            meshes.append(mesh)

        return meshes
    

    def _compute_pelvis_trajectories(self, sample):
        """Extract pelvis trajectories from SMPL-X parameters"""
        orients = sample['orient']
        poses = sample['pose']
        transls = sample['transl']
        
        trajs = []
        for i, (orient, pose, transl) in enumerate(zip(orients, poses, transls)):

            if i >= self.seq_len_limit:
                break

            smplx_output = self.smplx_model(
                global_orient=torch.tensor(orient[None]).float(),
                body_pose=torch.tensor(pose[None]).float(),
                transl=torch.tensor(transl[None]).float(),
            )

            # The 0th joint is the pelvis
            trajs.append(smplx_output.joints[0, 0].detach().cpu().numpy())

        return np.array(trajs)
    

    def _log_trajectories(self, trajs_list, labels=None, is_pred=False) -> None:
        """Visualize trajectories of the pelvis"""
        for i, traj in enumerate(trajs_list):
            ep = f"world/traj_pelvis_{i}"
            if is_pred:
                ep = ep + "_pred"
            label = labels[i] if labels is not None else 'traj'

            if is_pred:
                if label is None:
                    label = "pred"
                else:
                    label += "_pred"
            
            # Use color from palette cycling through available colors
            color_idx = i % len(self.palette.get("traj"))
            color = self.palette.get("traj")[color_idx]
            
            rr.log(
                ep,
                rr.LineStrips3D(
                    traj,
                    colors=color,
                    radii=self.line_radii if not is_pred else self.line_radii * 0.3,
                    labels=[label] if label is not None else None,
                ),
                static=True,
            )
            self._epaths_3d.add(ep)
    

    def _log_smplx_meshes(self, meshes, is_pred=False) -> None:
        """Visualize SMPL-X meshes for the current frame"""
        for i, mesh in enumerate(meshes):
            ep = f"world/smplx_{i}"
            if is_pred:
                ep = ep + "_pred_human"
            cc = self.palette.get("smplx")
            if is_pred:
                # use light blue
                cc = [150, 150, 255]
            rr.log(
                ep,
                rr.Mesh3D(
                    vertex_positions=mesh.vertices,
                    vertex_normals=mesh.vertex_normals,
                    triangle_indices=mesh.faces,
                    vertex_colors=cc,
                ),
                static=False,
            )
            self._epaths_3d.add(ep)

    def _log_text_labels(self, gt_meshes, pred_meshes, gt_text, pred_text, frame_idx) -> None:
        """Log text labels near the human meshes using rerun's text logging"""
        # Get the position of the GT human (center of mesh, top of head)
        if len(gt_meshes) > 0 and gt_meshes[0] is not None:
            gt_mesh = gt_meshes[0]
            # Get the top of the mesh (highest Y coordinate)
            gt_vertices = gt_mesh.vertices
            gt_top_y = np.max(gt_vertices[:, 1])
            gt_center = np.mean(gt_vertices, axis=0)
            gt_text_pos = np.array([gt_center[0], gt_top_y + 0.1, gt_center[2]])  # Position above the head
            
            # Log GT text if not empty
            if gt_text and len(gt_text) > 0:
                # Handle both list and string cases
                if isinstance(gt_text, list):
                    text_to_show = gt_text[0] if len(gt_text) > 0 and gt_text[0] else None
                else:
                    text_to_show = gt_text if gt_text else None
                
                if text_to_show and text_to_show.strip():
                    # Use Points3D with labels to display text in 3D space
                    rr.log(
                        "world/text_labels/gt_text",
                        rr.Points3D(
                            positions=[gt_text_pos],
                            labels=[f"GT: {text_to_show}"],
                            radii=0.02,
                            colors=[255, 200, 0],  # Orange color for GT
                        ),
                        static=False,
                    )
        
        # Get the position of the predicted human
        if len(pred_meshes) > 0 and pred_meshes[0] is not None:
            pred_mesh = pred_meshes[0]
            # Get the top of the mesh (highest Y coordinate)
            pred_vertices = pred_mesh.vertices
            pred_top_y = np.max(pred_vertices[:, 1])
            pred_center = np.mean(pred_vertices, axis=0)
            pred_text_pos = np.array([pred_center[0], pred_top_y + 0.3, pred_center[2]])  # Position above the head
            
            # Log predicted text if not empty
            if pred_text and len(pred_text) > 0:
                # Handle both list and string cases
                if isinstance(pred_text, list):
                    # sort to get the longest text
                    pred_text.sort(key=len, reverse=True)
                    text_to_show = pred_text[0] if len(pred_text) > 0 and pred_text[0] else None
                else:
                    text_to_show = pred_text if pred_text else None
                
                if text_to_show and text_to_show.strip():
                    # Use Points3D with labels to display text in 3D space
                    rr.log(
                        "world/text_labels/pred_text",
                        rr.Points3D(
                            positions=[pred_text_pos],
                            labels=[f"Pred: {text_to_show}"],
                            radii=0.02,
                            colors=[150, 150, 255],  # Light blue color for prediction
                        ),
                        static=False,
                    )

    def _convert_occ_grid_to_pcd(self, occ_grid, scene_grid_params):
        """
        Convert an occupancy grid to 3D point cloud.
        
        Args:
            occupancy_grid: Boolean tensor/array of shape [nx, ny, nz]
            scene_grid_params: Parameters defining the grid dimensions [x_min, y_min, z_min, x_max, y_max, z_max, nx, ny, nz]
        
        Returns:
            points: Numpy array of shape [N, 3] containing the center coordinates of occupied voxels
        """
        # Extract grid parameters
        x_min, y_min, z_min = scene_grid_params[0:3]
        x_max, y_max, z_max = scene_grid_params[3:6]
        nx, ny, nz = scene_grid_params[6:9]

        # Calculate voxel size
        voxel_size_x = (x_max - x_min) / nx
        voxel_size_y = (y_max - y_min) / ny
        voxel_size_z = (z_max - z_min) / nz

        if isinstance(occ_grid, torch.Tensor):
            occ_grid = occ_grid.cpu().numpy()
        
        occupied_indices = np.nonzero(occ_grid)

        indices = np.stack(occupied_indices, axis=1)
        scales = np.array([voxel_size_x, voxel_size_y, voxel_size_z])
        mins = np.array([x_min, y_min, z_min])
        offsets = np.array([0.5, 0.5, 0.5])
        
        points = (indices + offsets) * scales + mins
    
        return points


if __name__ == "__main__":
    all_objects_folder = '/home/tianhang/code/humoto/data/humoto_objects_0805' # TODO: put your object folder here
    smplx_model_path = '/home/tianhang/code/imu-human-mllm/models_smplx_v1_1/models/smplx' # TODO: put your SMPL-X model path here
    viewer = DataViewer(smplx_model_path=smplx_model_path, all_objects_folder=all_objects_folder)

    mean_ious = []
    precisions = []
    recalls = []
    mean_l1s = []
    l1_precisions = []
    l1_recalls = []
    l1_precisions_at_thresh = []
    l1_recalls_at_thresh = []
    for seq_id in range(0, 30):
        mean_iou, precision, recall, mean_l1, l1_precision, l1_recall, l1_precision_at_thresh, l1_recall_at_thresh = viewer(seq_id)
        print(f"Seq {seq_id}: Mean IoU = {mean_iou:.4f}, Precision = {precision:.4f}, Recall = {recall:.4f}")
        print(f"Seq {seq_id}: Mean L1 = {mean_l1:.4f}, L1 Precision = {l1_precision:.4f}, L1 Recall = {l1_recall:.4f}")
        print(f"Seq {seq_id}: L1 Precision@{viewer.distance_threshold} = {l1_precision_at_thresh:.4f}, L1 Recall@{viewer.distance_threshold} = {l1_recall_at_thresh:.4f}")
        mean_ious.append(mean_iou)
        precisions.append(precision)
        recalls.append(recall)
        mean_l1s.append(mean_l1)
        l1_precisions.append(l1_precision)
        l1_recalls.append(l1_recall)
        l1_precisions_at_thresh.append(l1_precision_at_thresh)
        l1_recalls_at_thresh.append(l1_recall_at_thresh)
    print(f"Mean IoU: {np.mean(mean_ious):.4f}")
    print(f"Mean Precision: {np.mean(precisions):.4f}")
    print(f"Mean Recall: {np.mean(recalls):.4f}")
    print(f"Mean Normalized L1: {np.mean(mean_l1s):.4f}")
    print(f"Mean L1 Precision: {np.mean(l1_precisions):.4f}")
    print(f"Mean L1 Recall: {np.mean(l1_recalls):.4f}")
    print(f"Mean L1 Precision@{viewer.distance_threshold}: {np.mean(l1_precisions_at_thresh):.4f}")
    print(f"Mean L1 Recall@{viewer.distance_threshold}: {np.mean(l1_recalls_at_thresh):.4f}")