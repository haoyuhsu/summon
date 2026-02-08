

python predict_contact.py /projects/benk/hhsu2/imu-humans/related_works/summon/outputs/vertices_ours/humoto/vertices_can \
    --load_model ../training/contactformer/model_ckpt/best_model_recon_acc.pt \
    --output_dir /projects/benk/hhsu2/imu-humans/related_works/summon/outputs/contact_predictions_ours/humoto


python predict_contact.py /projects/benk/hhsu2/imu-humans/related_works/summon/outputs/vertices_ours/parahome/vertices_can \
    --load_model ../training/contactformer/model_ckpt/best_model_recon_acc.pt \
    --output_dir /projects/benk/hhsu2/imu-humans/related_works/summon/outputs/contact_predictions_ours/parahome


python predict_contact.py /projects/benk/hhsu2/imu-humans/related_works/summon/outputs/vertices_ours/virtualhome/vertices_can \
    --load_model ../training/contactformer/model_ckpt/best_model_recon_acc.pt \
    --output_dir /projects/benk/hhsu2/imu-humans/related_works/summon/outputs/contact_predictions_ours/virtualhome