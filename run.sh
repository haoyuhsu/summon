

# cd contactFormer

# # Generate vertices from SMPL parameters (use imu-humans)
# python generate_vertices_from_smpl.py \
#     --root_dir /home/haoyuyh3/Documents/maxhsu/imu-humans/_tmp_test_data/result_humoto \
#     --output_dir ../data_custom/humoto \
#     --smplx_model_dir /home/haoyuyh3/Documents/maxhsu/imu-humans/body_models/human_model_files \
#     --mesh_ds_us_dir ../mesh_ds \
#     --gender neutral \
#     --downsample \
#     --normalize_ori

# python generate_vertices_from_smpl.py \
#     --root_dir /home/haoyuyh3/Documents/maxhsu/imu-humans/_tmp_test_data/result_lingo \
#     --output_dir ../data_custom/lingo \
#     --smplx_model_dir /home/haoyuyh3/Documents/maxhsu/imu-humans/body_models/human_model_files \
#     --mesh_ds_us_dir ../mesh_ds \
#     --gender neutral \
#     --downsample \
#     --normalize_ori


# #### Remember to change the conda env to contactformer ####

# # Predict contact from generated vertices
# python predict_contact.py ../data_custom/humoto/vertices_can \
#     --load_model ../training/contactformer/model_ckpt/best_model_recon_acc.pt \
#     --output_dir ../contact_predictions/humoto/

# python predict_contact.py ../data_custom/lingo/vertices_can \
#     --load_model ../training/contactformer/model_ckpt/best_model_recon_acc.pt \
#     --output_dir ../contact_predictions/lingo/


# cd ../


# Fit 3D objects
# python fit_best_obj.py \
#     --sequence_name MPH11_00150_01 \
#     --vertices_path ./data/custom_sequences/vertices/MPH11_00150_01_verts.npy \
#     --contact_labels_path ./contact_predictions/MPH11_00150_01.npy \
#     --output_dir fitting_results
# python run_fit_best_obj.py \
#     --vertices_path ./data_custom_mobileposer/humoto/vertices \
#     --contact_labels_path ./contact_predictions/humoto \
#     --output_dir fitting_results/humoto/
# python run_fit_best_obj.py \
#     --vertices_path ./data_custom_mobileposer/lingo/vertices \
#     --contact_labels_path ./contact_predictions/lingo \
#     --output_dir fitting_results/lingo/
python run_fit_best_obj.py \
    --vertices_path ./data_custom_mobileposer/parahome/vertices \
    --contact_labels_path ./contact_predictions/parahome \
    --output_dir fitting_results/parahome/
python run_fit_best_obj.py \
    --vertices_path ./data_custom_mobileposer/virtualhome/vertices \
    --contact_labels_path ./contact_predictions/virtualhome \
    --output_dir fitting_results/virtualhome/



# Visualize fitting results (optional)
# python vis_fitting_results.py \
#     --fitting_results_path fitting_results/lingo/sample_seq_0883 \
#     --vertices_path ./data_custom/lingo/vertices/sample_seq_0883_verts.npy



# # Complete the scene with the fitted 3D object models (spare_length is for room region extension, num_iter is for number of objects to add)
# python scene_completion.py \
#     --fitting_results_path fitting_results/MPH11_00150_01/ \
#     --path_to_model atiss_ckpt \
#     --obj_dataset_path 3D_Future/models \
#     --spare_length 1 \
#     --num_iter 2
