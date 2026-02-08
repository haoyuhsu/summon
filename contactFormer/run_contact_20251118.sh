
#### Remember to change the conda env to imu-humans ####

# Generate vertices from SMPL parameters (use imu-humans)
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

# python generate_vertices_from_smpl.py \
#     --root_dir /home/haoyuyh3/Documents/maxhsu/imu-humans/_tmp_test_data/result_parahome \
#     --output_dir ../data_custom/parahome \
#     --smplx_model_dir /home/haoyuyh3/Documents/maxhsu/imu-humans/body_models/human_model_files \
#     --mesh_ds_us_dir ../mesh_ds \
#     --gender neutral \
#     --downsample \
#     --normalize_ori

# python generate_vertices_from_smpl.py \
#     --root_dir /home/haoyuyh3/Documents/maxhsu/imu-humans/_tmp_test_data/result_virtualhome \
#     --output_dir ../data_custom/virtualhome \
#     --smplx_model_dir /home/haoyuyh3/Documents/maxhsu/imu-humans/body_models/human_model_files \
#     --mesh_ds_us_dir ../mesh_ds \
#     --gender neutral \
#     --downsample \
#     --normalize_ori


#### Remember to change the conda env back to summon ####

# # Predict contact from generated vertices
# python predict_contact.py ../data_custom/humoto/vertices_can \
#     --load_model ../training/contactformer/model_ckpt/best_model_recon_acc.pt \
#     --output_dir ../contact_predictions/humoto/

# python predict_contact.py ../data_custom/lingo/vertices_can \
#     --load_model ../training/contactformer/model_ckpt/best_model_recon_acc.pt \
#     --output_dir ../contact_predictions/lingo/

# python predict_contact.py ../data_custom/parahome/vertices_can \
#     --load_model ../training/contactformer/model_ckpt/best_model_recon_acc.pt \
#     --output_dir ../contact_predictions/parahome/

# python predict_contact.py ../data_custom/virtualhome/vertices_can \
#     --load_model ../training/contactformer/model_ckpt/best_model_recon_acc.pt \
#     --output_dir ../contact_predictions/virtualhome/