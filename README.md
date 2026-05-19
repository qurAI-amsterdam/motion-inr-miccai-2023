> Code accompanying the paper **"Deep Learning for Automatic Strain Quantification in Arrhythmogenic Right Ventricular Cardiomyopathy"** (MICCAI 2023 / STACOM workshop).
---

## Overview

This repository implements an **Implicit Neural Representation (INR)** framework for myocardial motion estimation from cine cardiac MRI. Rather than learning a single feed-forward model across a dataset, a small MLP (SIREN) is optimized per subject, mapping 3D spatial coordinates `(x, y, z)` to a continuous displacement field, producing smooth and differentiable deformation estimates for a given image pair.

The framework supports:
- **ES→ED** registration (single timepoint pair)
- **All-timepoints** mode — sequential registration of every frame to a common reference
- Optional **multi-view** registration combining short-axis (SAX) and 4-chamber (4CH) views
- **INR warm-start initialization** from a previously trained network (e.g. from the preceding timepoint)
- Jacobian-based volume-preservation regularization
- Post-processing to extract radial, circumferential, and longitudinal strain components

---

## Repository Structure

```
motion-inr-miccai-2023/
├── run_registration.py              # Main entry point
├── run_registration_debug.py        # Single-patient debug entry point
├── registration_config.cfg          # All hyperparameters and paths
│
├── models/
│   ├── models_v4.py                 # ImplicitRegistrator: training loop, warping, coord handling
│   └── common.py                    # Shared model utilities
│
├── networks/
│   └── networks.py                  # SIREN and MLP network definitions
│
├── objectives/
│   ├── ncc.py                       # Normalized cross-correlation loss
│   └── regularizers.py              # Jacobian loss, bending energy, compute_jacobian_matrix
│
├── kwatsch/
│   ├── canonical_space.py           # CanonicalImage: coord systems, alignment, view management
│   ├── common.py                    # View keys, resampling, coord transforms
│   ├── simple_aligner.py            # RV-LV alignment utilities
│   ├── helper.py                    # Misc helpers
│   ├── dice_loss.py                 # Dice/overlap computation
│   ├── shape_loss.py                # Shape-based losses
│   └── bob_metrics.py               # Evaluation metrics
│
├── CMRI/
│   ├── general.py                   # Label definitions (MMS2MRILabel), image utilities,
│   │                                #   apex/base checks, centre-of-mass, rotation matrices
│   ├── evaluation/
│   │   ├── dvf.py                   # Strain computation from Jacobian fields
│   │   └── strain_to_aha17.py       # AHA 17-segment model mapping
│   └── contours/
│       └── common.py                # Contour extraction and normal vector computation
│
├── utils/
│   ├── registration_utils.py        # Preprocessing, post_process, plot utilities
│   ├── create_aligned_scan.py       # Standalone scan alignment utilities
│   └── general.py                   # Trilinear interpolation, coordinate de-normalization
│
└── postprocessing_utils/
    ├── strain_contours.py           # get_strain, get_strain_mask, get_strain_lax,
    │                                #   get_strain_contour, AHA part assignment
    └── excel_generation.py          # Results aggregation into Excel
```

---

## Requirements

- Python ≥ 3.8
- PyTorch (CUDA strongly recommended; the script hard-sets `CUDA_VISIBLE_DEVICES` at the top of `run_registration.py` — change as needed)
- SimpleITK
- NumPy, SciPy, pandas, matplotlib, tqdm

Install dependencies (no `requirements.txt` is provided):

```bash
pip install torch torchvision simpleitk numpy scipy pandas matplotlib tqdm openpyxl
```

---

## Data Format

Images and segmentations are expected as **4D NIfTI files** (`.nii.gz`), one file per patient, with the temporal dimension as the last axis. File naming for ED/ES mode encodes timepoints directly in the filename: `<center>_..._<ES-frame>_..._<ED-frame>.nii.gz` — these are parsed automatically by `get_es_and_ed_timepoints()` in `utils/registration_utils.py`.

Segmentations must contain these integer labels (defined in `CMRI/general.py` as `MMS2MRILabel`):
- `1` — LV blood pool (LVBP)
- `2` — LV myocardium (LV)
- `3` — RV blood pool (RVBP)

Expected directory layout:

```
<root>/
  <path_to_data>/
    patient_001.nii.gz
    patient_002.nii.gz
    ...
  <path_to_segmentation>/
    patient_001.nii.gz
    patient_002.nii.gz
    ...
  <path_to_data_lax>/          # Only needed if multi_view = 1
    patient_001.nii.gz
    ...
  <path_to_segmentation_lax>/  # Only needed if multi_view = 1
    patient_001.nii.gz
    ...
```

---

## Preprocessing Pipeline

Preprocessing is orchestrated in `utils/registration_utils.py` and `kwatsch/canonical_space.py`. The steps below run automatically before each registration.

### 1. Image and segmentation loading
`get_images_with_segmentations()` reads the 4D SAX NIfTI (and optionally the 4CH NIfTI) for the requested patient using SimpleITK.

### 2. Timepoint extraction
For ED→ES mode (`all_timepoints = False`), `get_es_and_ed_timepoints()` parses the ES and ED frame indices directly from the patient filename. For all-timepoints mode, frames are iterated sequentially with the last frame used as the fixed reference.

### 3. 3D volume slicing
`get_canonical_image_aligned()` slices the 4D image at the fixed and moving timepoints, yielding two 3D SimpleITK image/mask pairs.

### 4. Intensity normalization
If the image intensities are not already in `[0, 1]`, `sitk.RescaleIntensity` is applied to both fixed and moving images (and optionally the 4CH images).

### 5. ROI cropping (optional, `crop = True`)
`convert_to_binary_and_get_bbox()` binarizes the fixed segmentation and computes a tight bounding box around all non-zero labels using `sitk.LabelShapeStatisticsImageFilter`. A fixed padding of `[15, 10, 5]` voxels is added in x, y, z respectively, and both fixed and moving image/mask pairs are cropped to this box. This speeds up training significantly for large acquisitions.

### 6. Apex–base orientation check
`check_apex_base_orientation()` (in `CMRI/general.py`) inspects the LVBP label distribution along the z-axis to determine whether the stack is ordered apex-to-base or base-to-apex, and sets a flip flag accordingly.

### 7. RV–LV rotational alignment
`get_rv_lv_rot_matrix()` computes the centres of mass of the LVBP and RVBP labels and derives a rotation matrix that brings the RV–LV axis into a canonical (rightward) orientation. An additional y-flip check ensures the RV consistently sits to the right of the LV in image space. The resulting 4×4 matrix is stored in the `CanonicalImage` object.

### 8. CanonicalImage construction and alignment
`CanonicalImage` (in `kwatsch/canonical_space.py`) wraps the SimpleITK image and segmentation and applies the rotation and flip to bring the image into the canonical coordinate frame. For multi-view runs, the 4CH image and segmentation are added via `add_view()`, with their coordinates mapped into the canonical SAX frame.

### 9. Coordinate generation and scaling
`_init_coords()` generates a full grid of world coordinates (in mm) for the SAX volume (and optionally the 4CH plane). All coordinates are min–max scaled to `[-1, 1]` across the combined fixed/moving coordinate range, satisfying the input requirement of the SIREN network.

### 10. Mask preparation for training
Inside `_init_images()` in `models/models_v4.py`, the LV myocardium mask defines the active region for loss computation. The RV myocardium is approximated by dilating the RVBP mask by 2 voxels (`generate_rv_myocardium()`). A dilated union of fixed and moving LV masks is computed as the ROI for the image similarity loss.

---

## Training (INR Optimization)

Each patient/timepoint is registered by optimizing a SIREN network from scratch (or warm-started from a prior INR). The main training loop lives in `ImplicitRegistrator.fit_new()` in `models/models_v4.py`.

### Network architecture
Defined in `networks/networks.py`. The default is a SIREN with sinusoidal activations and architecture specified by `layers` in the config (default `[3, 256, 256, 256, 3]` — 3 spatial inputs, three 256-unit hidden layers, 3 displacement outputs). The SIREN frequency parameter `omega` (default `45`) scales the sinusoidal activations. A second reverse network (`network_rev`) is also instantiated but only used when cycle consistency is enabled, which is off by default (`cycle_alpha = 0`).

### INR warm-start initialization
When `warpinn_init = True` and `all_timepoints = True`, the network weights from the previously converged INR can be loaded to initialize training for the next timepoint. This is handled in `_registration_init()` via `network.load_state_dict(torch.load(network_from_file))`. Providing a warm start can improve convergence speed for temporally adjacent frames.

### Training loop
At each epoch, a random batch of `batch_size` coordinates is sampled from the full coordinate grid. The network predicts a relative displacement at each coordinate; adding this to the input coordinate gives the forward warp estimate. The moving image is trilinearly interpolated at those estimated positions and compared to the fixed image.

### Loss function

**Image similarity** (`objectives/ncc.py`): Normalized cross-correlation (NCC) between the warped moving image samples and the fixed image samples. For multi-view runs, SAX and 4CH losses are computed separately and averaged.

**Jacobian regularization** (`objectives/regularizers.py`): A balanced Jacobian loss is applied within the myocardial mask, encouraging locally volume-preserving deformations. A much lower weight (`alpha_jacobian / 1000`) is applied outside the myocardium to allow free motion in surrounding structures. Controlled by `jacobian_regularization = True` and `alpha_jacobian`.

**Bending energy** (optional, `bending_regularization`): Penalizes second-order spatial derivatives, discouraging locally erratic deformations. Off by default.

### Optimizer and scheduler
Adam optimizer (`lr = 1e-5`). A `ReduceLROnPlateau` scheduler is active when `scheduler = True`, reducing LR by a factor of 10 when the loss plateaus.

### Early stopping
When `early_stopping = True`, training halts if the loss improvement falls below `0.001` for `epochs / 10` consecutive epochs. The actual stopping epoch is recorded in `ImpReg.stopped_at_epoch`.

---

## Post-processing Pipeline

Post-processing runs immediately after training in `post_process_completed()` (`utils/registration_utils.py`).

### 1. Full-field warping
The trained network is evaluated over all SAX coordinates (chunked at 10,000 coordinates to fit GPU memory) to produce the complete DVF. The Jacobian matrix `∇φ` is computed analytically via autograd at every voxel (`compute_jacobian_matrix()` in `objectives/regularizers.py`), both with and without the identity component. Optionally, the Jacobian is re-scaled from normalized coordinate space to physical mm-space via `scale_jacobian()`. The warped image and warped segmentation mask are also produced. For multi-view runs, the 4CH view is additionally warped by mapping 4CH coordinates through the SAX-space network (`warp_4ch_view()`).

### 2. Strain mask preparation
`get_strain_mask()` (`postprocessing_utils/strain_contours.py`) extracts the LV myocardium voxels from the fixed segmentation. When `omit_base_apex = True`, the most basal and apical slices are excluded from the mask by the fraction specified in `omit_base_apex_perc`.

### 3. LV strain from Jacobian
`get_strain()` calls `compute_lv_strain()` in `CMRI/evaluation/dvf.py`, which projects the Jacobian field onto radial (RR), circumferential (CC), and longitudinal (LL) directions to produce voxel-wise strain maps. Radial directions are derived from the gradient of the signed distance to the endocardial surface; circumferential directions are set orthogonal to radial within the short-axis plane. When `conver_to_engineering = True`, Green–Lagrange strain `E` is converted to engineering strain via `ε = (√(1 + 2E) − 1) × 100`. Global scalar values (GLS-RR, GLS-CC, GLS-LL) are computed as the mean over the strain mask. Strain maps are saved to `strain.npz`.

### 4. Regional LV strain
`compute_lv_strain_per_part()` averages strain separately for basal, mid-cavity, and apical slices using the AHA level assignments from `determine_aha_part()`.

### 5. RV and septal strain
`get_strain_contour()` extracts contour points and surface normals for the RVBP and septal labels. Contour points are aligned to the canonical frame, warped through the network, and strain is computed at each contour point using the local surface normals (`compute_strain_with_normals()`). Per-AHA-zone averages (apical, mid, basal) are computed for both RR and CC components.

### 6. 4CH longitudinal strain (multi-view only)
`get_strain_lax()` uses the 4CH-view Jacobian field to compute longitudinal (LL) strain via `compute_ll_strain_4ch()`, yielding a global GLS-LL from the 4CH view.

### 7. Overlap evaluation
Dice similarity coefficients are computed between the warped moving segmentation and the fixed segmentation (`compute_overlap()` in `kwatsch/dice_loss.py`) for all three label classes, both globally and on three representative slices.

### 8. Volume preservation check
`compute_preserve_volume()` computes `|det(∇φ) − 1|` within the LV myocardium on three representative slices, as a measure of how closely the deformation preserves local tissue volume.

### 9. Results export
All scalar metrics (strain values, Dice scores, Jacobian statistics, timepoints, patient ID, stopping epoch) are assembled by `generate_excel_row()` (`postprocessing_utils/excel_generation.py`) and saved to `results.xlsx`.

### 10. Network and array saving
When `save_net = True`, the full `ImplicitRegistrator` object is saved to `model_network.pth`, and the network's input coordinates and output displacements for the full image are saved to `in_and_out_dict.npz`. The saved model can be loaded as a warm-start for the next timepoint.

### 11. Visualizations
Five-slice grid images are saved as `.png` and `.npz` for the fixed, moving, and warped images. A DVF figure (`dvf_img.png`) shows the quiver field and displacement magnitude heatmaps at the mid-cavity slice.

---

## Output Files

For each patient/timepoint, the following files are written under:
`<root>/registration_output_experiments/<experiment_folder_name>/<patient_id>/[tp_<fixed>_<moving>/]`

| File | Description |
|------|-------------|
| `results.xlsx` | All scalar metrics: strain (GLS-RR/CC/LL, per-zone, RV, SEP), Dice, Jacobian stats |
| `strain.npz` | Voxel-wise strain arrays — keys: `strain` (full tensor), `rr`, `cc`, `ll` |
| `strain_4ch.npz` | 4CH longitudinal strain map (multi-view only) |
| `displacement_field.npz` | DVF in voxel space, shape `(z, y, x, 3)` |
| `displacement_field_4ch.npz` | 4CH DVF (multi-view only) |
| `jacobian_field.npz` | Per-voxel Jacobian matrices, shape `(z, y, x, 3, 3)` |
| `model_network.pth` | Full `ImplicitRegistrator` object (when `save_net = True`) |
| `in_and_out_dict.npz` | INR input coordinates and output displacements |
| `result_dict.npz` | Complete result dictionary (images, masks, DVF, Dice, Jacobian stats) |
| `img_fixed.npz` / `.png` | Fixed image array and 5-slice visualization |
| `img_moving.npz` / `.png` | Moving image array and 5-slice visualization |
| `warped_img.npz` / `.png` | Warped image array and 5-slice visualization |
| `dvf_img.png` | DVF quiver and displacement magnitude heatmap |
| `loss_log.txt` | Total loss per epoch |
| `data_loss_log.txt` | Image similarity loss per epoch |
| `kwargs.json` | Copy of all configuration parameters used |

The experiment folder also contains a copy of `registration_config.cfg` and the `run_registration.py` script for full reproducibility.

---

## Configuration Reference

### `[paths]`
| Key | Description |
|-----|-------------|
| `root` | Root directory for data and experiments |
| `path_to_data` | Relative path to SAX image folder |
| `path_to_segmentation` | Relative path to SAX segmentation folder |
| `path_to_data_lax` | LAX/4CH images (only if `multi_view = 1`) |
| `path_to_segmentation_lax` | LAX/4CH segmentations (only if `multi_view = 1`) |
| `experiment_folder_name` | Output folder name under `registration_output_experiments/` |
| `dataset` | Dataset label for logging |

### `[training]`
| Key | Default | Description |
|-----|---------|-------------|
| `lr` | `1e-5` | Adam learning rate |
| `batch_size` | `10000` | Coordinate samples per training iteration |
| `scheduler` | `True` | Enable ReduceLROnPlateau LR scheduler |
| `warpinn_init` | `True` | Warm-start network weights from a previously saved INR |

### `[network]`
| Key | Default | Description |
|-----|---------|-------------|
| `network_type` | `SIREN` | Network type: `SIREN` or `MLP` |
| `layers` | `[3, 256, 256, 256, 3]` | Layer sizes (input, hidden..., output) |
| `omega` | `45` | Frequency parameter for SIREN sinusoidal activations |
| `normalize_coords` | `True` | Scale input coords to `[-1, 1]` |

### `[regularization]`
| Key | Default | Description |
|-----|---------|-------------|
| `jacobian_regularization` | `True` | Enable Jacobian-based volume preservation |
| `alpha_jacobian` | `0.05` | Jacobian regularization weight inside myocardium |
| `background_weight` | `0.0001` | Jacobian weight outside myocardium |
| `bending_regularization` | `False` | Penalize second-order spatial derivatives |
| `early_stopping` | `True` | Stop when loss improvement < 0.001 for `epochs/10` steps |

### `[image hyperparameters]`
| Key | Default | Description |
|-----|---------|-------------|
| `use_mask_loss` | `True` | Restrict image similarity to myocardial ROI |
| `reg_with_mask` | `True` | Restrict Jacobian regularization to myocardial mask |
| `omit_base_apex` | `True` | Exclude most basal/apical slices from strain mask |
| `omit_base_apex_perc` | `0` | Fraction of slices to exclude at each end |

### `[experiment]`
| Key | Default | Description |
|-----|---------|-------------|
| `epochs` | `3500` | Maximum training epochs per subject/timepoint |
| `all_timepoints` | `True` | Register all frames to the last frame as reference |
| `multi_view` | `0` | Enable SAX + 4CH multi-view fusion (0 = off, 1 = on) |
| `crop` | `True` | Crop image to LV bounding box before registration |
| `seed` | `0` | Random seed |
| `conver_to_engineering` | `True` | Convert Green–Lagrange to engineering strain |
| `compute_physical_dvf` | `False` | Use mm-scaled Jacobian for strain (experimental) |

### `[general settings]`
| Key | Description |
|-----|-------------|
| `multiprocessing` | Process patients in parallel (4 workers) |
| `debug` | Process a single patient only |
| `save_net` | Save trained network weights and I/O arrays |
| `wandb_enabled` | Enable Weights & Biases logging |

---

## Running

### 1. Edit paths

```ini
[paths]
root = /path/to/your/data
path_to_data = images/sax
path_to_segmentation = segmentations/sax
experiment_folder_name = my_experiment
dataset = my_cohort
```

### 2. Set GPU

At the top of `run_registration.py`:

```python
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
```

### 3. Run

```bash
python run_registration.py
```

For a quick single-patient test:

```bash
python run_registration_debug.py
# or set debug = True in registration_config.cfg
```

---

## Citation

```bibtex
@inproceedings{alvarez2023deep,
  title={Deep learning for automatic strain quantification in arrhythmogenic right ventricular cardiomyopathy},
  author={Alvarez-Florez, Laura and Sander, J{\"o}rg and Bourfiss, Mimount and Tjong, Fleur VY and Velthuis, Birgitta K and I{\v{s}}gum, Ivana},
  booktitle={International workshop on statistical atlases and computational models of the heart},
  pages={25--34},
  year={2023},
  organization={Springer}
}
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
