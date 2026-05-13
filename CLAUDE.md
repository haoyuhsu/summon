# SUMMON — Scene Synthesis from Human Motion (SIGGRAPH Asia 2022)

This repo synthesizes a 3D scene that is consistent with an input human motion sequence.
The pipeline has three stages: (1) per-vertex contact prediction with **ContactFormer**,
(2) per-cluster best-fit object retrieval/optimization from a **3D-FUTURE** library, and
(3) optional non-contact scene completion with **ATISS**.

Body representation: **SMPL-X** (10475 vertices, neutral gender by default), downsampled
to 655 vertices for contact reasoning.

## Q1 — How are contact labels for ContactFormer training obtained?

Ground truth contact labels are **not annotated by hand**; they are computed automatically
from PROX scenes using the scene SDF + scene semantic volume. See [contactFormer/gen_dataset.py](contactFormer/gen_dataset.py#L93-L173).

Per-frame procedure (PROXD dataset):

1. **Build the body mesh.** For each `.pkl` of SMPL-X parameters in PROXD, run SMPL-X
   ([data_utils.pkl_to_canonical](contactFormer/data_utils.py#L165-L213)) and transform to world space using `cam2world.json`.
2. **Downsample** the SMPL-X mesh from 10475 → 655 vertices using two Spiral-Conv
   downsampling matrices `D_1` and `D_2` ([gen_dataset.py:87-91, 122-129](contactFormer/gen_dataset.py#L87-L129)).
3. **Read the scene SDF** at every body vertex via trilinear `grid_sample`
   ([data_utils.read_sdf](contactFormer/data_utils.py#L253-L265)). A vertex is "in contact" iff its signed distance to the
   scene mesh is `< 0.05` m ([gen_dataset.py:131-136](contactFormer/gen_dataset.py#L131-L136)).
4. **Read the scene semantic volume** at the same vertex (mode = `nearest`) to get the
   mpcat40 class id of whichever scene object is closest. Non-contact vertices are zeroed
   out ([gen_dataset.py:138-141](contactFormer/gen_dataset.py#L138-L141)).
5. Label remapping inside [data_utils.load_scene_data](contactFormer/data_utils.py#L216-L251):
   `seating(34)→sofa(10)`, `shower(25)→lighting(28)`.

Outputs per sequence (saved under e.g. [data/proxd_train/](data/proxd_train/)):
- `vertices/<seq>_verts.npy`   → world-space body verts, shape `(T, 655, 3)`
- `vertices_can/<seq>_verts_can.npy` → pelvis-centered ("canonical") body verts
- `contacts/<seq>_cf.npy`       → binary contact mask `(T, 655)`
- `contacts_semantics/<seq>_cfs.npy` → mpcat40 class per vertex `(T, 655)` (0 = no contact)

The training loop ([train_contactformer.py:30-50](contactFormer/train_contactformer.py#L30-L50)) uses only
`vertices_can` (input) and the semantic contact labels (`*_cfs.npy`, one-hot expanded to
the 8-class subset, see [dataset.py:322-323](contactFormer/dataset.py#L322-L323)) as supervision.

The 8-class subset used at training/inference time is in [utils.py:101-110](utils.py#L101-L110):

```
subset_idx → mpcat40_idx
0 void, 1 wall, 2 floor, 3 chair, 4 sofa(10), 5 table(5), 6 bed(11), 7 stool(19)
```

## Q2 — Which simplified body meshes? How are they generated? SMPL or SMPL-X?

**SMPL-X**, neutral gender (10475 vertices, [data_utils.load_body_model](contactFormer/data_utils.py#L104-L122) hard-codes
`model_type='smplx'`). Contact reasoning happens on a downsampled 655-vertex mesh.

The cascade is precomputed and shipped under [mesh_ds/](mesh_ds/) (also duplicated at
[data/mesh_ds/](data/mesh_ds/)):

| level | #verts | template OBJ                | meaning                        |
|-------|--------|-----------------------------|--------------------------------|
| 0     | 10475  | [mesh_0.obj](mesh_ds/mesh_0.obj) | full SMPL-X                    |
| 1     | 2619   | [mesh_1.obj](mesh_ds/mesh_1.obj) | 1× downsampled                 |
| 2     | 655    | [mesh_2.obj](mesh_ds/mesh_2.obj) | 2× downsampled — used in model |
| 3–5   | …      | mesh_3/4/5.obj              | (unused for ContactFormer)     |

For each level `k` there are three sparse matrices:
- `A_k.npz` — adjacency / spiral neighborhood
- `D_k.npz` — downsampling operator from level `k-1` to level `k`
- `U_k.npz` — upsampling operator from level `k` to level `k-1`

These matrices originate from POSA / Coma-style spiral convolutions and are loaded by
[data_utils.get_graph_params](contactFormer/data_utils.py#L51-L62). The downsampling is applied as a sparse
matrix-multiply ([data_utils.ds_us](contactFormer/data_utils.py#L86-L101)). To produce 655-vertex inputs the pipeline
chains `D_1` then `D_2` ([gen_dataset.py:87-91, 122-124](contactFormer/gen_dataset.py#L87-L124)):

```
SMPL-X (10475)  --D_1-->  (2619)  --D_2-->  (655)
```

Faces of the 655-vertex mesh are read from [mesh_ds/mesh_2.obj](mesh_ds/mesh_2.obj) wherever the code needs a
human triangle mesh ([utils.read_sequence_human_mesh](utils.py#L288-L294)).

In addition, [contactFormer/support_files/downsampled_weights.npy](contactFormer/support_files/downsampled_weights.npy) stores per-vertex
SMPL-X skinning weights downsampled to the 655-vertex layout. Its argmax gives the
"associated joint" for each vertex, used in
[normalize_orientation](contactFormer/data_utils.py#L138-L163) to rotate every sequence so the first frame faces a
canonical direction.

These downsampling matrices are **POSA's**, not SUMMON's. They are pre-shipped; the repo
does **not** generate them. To produce them from scratch you would run the POSA / mesh-
sampling pipeline (Coma's `mesh_sampling.py`), which clusters SMPL-X faces with quadric
edge collapse and stores `(A,D,U)` per level.

## Q3 — How are predicted contact labels turned into specific object instances?

After ContactFormer outputs a per-frame, per-vertex class label (8 classes) — saved by
[predict_contact.py](contactFormer/predict_contact.py) — [fit_best_obj.py](fit_best_obj.py) does the
following (this is the per-sequence entry point invoked by [run_fit_best_obj.py](run_fit_best_obj.py)):

1. **Remap subset → mpcat40** using `pred_subset_to_mpcat40` ([fit_best_obj.py:51](fit_best_obj.py#L51)). Vertex
   frames are also subsampled by 8 because contact prediction was run with
   `jump_step=8` ([fit_best_obj.py:48-53](fit_best_obj.py#L48-L53)).
2. **Build a human SDF** by merging all per-frame downsampled human meshes
   (`mesh_2.obj` faces) and running [utils.generate_sdf](utils.py#L242-L275) to produce a 256³ SDF — this is the
   penetration term for object placement.
3. **Estimate floor height** via DBSCAN on the lowest z of vertices labeled `floor=2`
   ([utils.estimate_floor_height](utils.py#L348-L365)).
4. **Local majority voting.** For each contact class in `classes_eps`
   ({3 chair, 5 table, 10 sofa, 11 bed, 19 stool}) collect contact vertices over all
   frames, voxel-downsample to 4 cm. Then DBSCAN over **all** points (mixed classes)
   with `voting_eps=0.1`; each cluster is assigned the **majority** class label
   ([fit_best_obj.py:117-166](fit_best_obj.py#L117-L166)). This filters spurious per-frame label noise.
5. **Per-class clustering.** For each class, voxel-downsample again and DBSCAN with a
   class-specific `eps` from [config.py](config.py) (chair 0.2, table 1.0, sofa 0.8, bed 1.0,
   stool 0.2). Each cluster becomes one **object instance** ([fit_best_obj.py:168-200](fit_best_obj.py#L168-L200)).
6. **Object retrieval & placement.** For every cluster:
   - Iterate every candidate `raw_model.obj` in [3D_Future/models/<class_name>/](3D_Future/models/).
   - Align the candidate to the floor (`X+90°`, set `min_z=0`,
     [utils.align_obj_to_floor](utils.py#L376-L389)).
   - Translate it so its xy-center matches the cluster's xy-center.
   - **Grid search** rotation `0..360°` step 10° × 11×11 xy translations in the cluster
     bbox; loss = contact-distance loss + scene-penetration loss against the human SDF
     ([place_obj_opt.grid_search](place_obj_opt.py#L50-L99), [place_obj_opt.contact_loss](place_obj_opt.py#L10-L15),
     [place_obj_opt.penetration_loss](place_obj_opt.py#L32-L47)).
   - **Adam refinement** of `(rot, transl_x, transl_y)` for 200 steps from the grid
     winner ([place_obj_opt.optimization](place_obj_opt.py#L102-L170)).
   - The candidate with the lowest final loss is recorded as the cluster's instance and
     written to `<output>/<seq>/fit_best_obj/<class>/<idx>/best_obj_id.json`. The fitted
     mesh is at `<best_obj_id>/opt_best.obj`.

So an "object instance" in SUMMON corresponds to **one DBSCAN cluster of contact
vertices of a single class**, and the instance is realized by the 3D-FUTURE model whose
SE(2) pose minimizes contact + penetration loss against that cluster + the cumulative
human SDF.

## End-to-end pipeline (matches [run.sh](run.sh))

```
SMPL-X params  →  generate_vertices_from_smplx*.py     # → vertices/, vertices_can/
                  (downsample to 655, normalize_ori)
vertices_can   →  contactFormer/predict_contact.py     # → per-frame, per-vertex 8-class
                  (loads training/contactformer/...)
contact + verts → fit_best_obj.py / run_fit_best_obj.py # → fitting_results/<seq>/
                  (3D_Future retrieval + Adam fit)
optionally     →  scene_completion.py (ATISS)          # adds non-contact furniture
```

## Useful entry points

- Contact dataset construction: [contactFormer/gen_dataset.py](contactFormer/gen_dataset.py)
- ContactFormer model: [contactFormer/contactFormer.py](contactFormer/contactFormer.py) (`ContactFormer` class
  wraps a frozen POSA cVAE encoder/decoder + a `TransformerDecoder` over time).
- Training loop: [contactFormer/train_contactformer.py](contactFormer/train_contactformer.py)
  (loss = KL + cross-entropy on semantics, [train_contactformer.py:39-49](contactFormer/train_contactformer.py#L39-L49)).
- Inference: [contactFormer/predict_contact.py](contactFormer/predict_contact.py)
  (uses `--save_probability` to keep softmax instead of argmax).
- Object fitting config: [config.py](config.py).
- Contact→object label remap: `pred_subset_to_mpcat40` in [utils.py:101-110](utils.py#L101-L110).
- mpcat40 label table: [mpcat40.tsv](mpcat40.tsv).

## Notes / gotchas

- **Frame skip = 8** is baked into the pipeline. Predictions are produced every 8 frames,
  and `fit_best_obj.py` re-aligns vertex frames with `vertices[frame * 8]`.
- **Only SMPL-X neutral** is used end-to-end; `gender` is plumbed but the run scripts pin
  `neutral`.
- Training labels live in `<data>/semantics/<seq>_cfs.npy`; the binary mask
  `contacts/<seq>_cf.npy` is generated but not used by `train_contactformer.py`.
- Inputs are pelvis-centered + first-frame yaw-normalized
  ([normalize_orientation](contactFormer/data_utils.py#L138-L163)) so the model is invariant to global pose.
- The 655-vertex human SDF is cached on first run (`<output>/<seq>/human/sdf.{npy,json}`).
