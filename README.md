# GRIT: Gradient and Intensity Thresholding CT Lung Nodule Segmentation

Custom 2D V-Net that segments lung nodules on CT. Required input: paths to the CT volumes (`.nii` or `.nii.gz`) and the location of each nodule. Outputs: a binary (0=background, 1=nodule) mask for each nodule. 

Each nodule is processed in these steps:

1. The CT is reoriented to LPS and resampled to 1 mm isotropic. HU values are clipped to [-1000, 400].
2. A 64 × 64 × 64 crop is taken around the nodule coordinate you supplied.
3. Every axial slice of the crop is segmented by the 2D V-Net.
4. The probabilities are thresholded at 0.5.
5. The mask is resampled back onto the original CT grid and written to disk.

---

## 1. Installation

```bash
git clone https://github.com/<your-username>/<this-repo>.git
cd <this-repo>

pip install torch numpy pandas scipy SimpleITK tqdm matplotlib openpyxl
```

| File | Purpose |
|---|---|
| `infer_test_cohort.py` | Entry point (run this) |
| `infer_lib.py` | Model loading, prediction, post-processing, metrics, PNGs |
| `common.py` | I/O, reorientation, resampling, cropping |
| `vnet2d.py` | Model definition |

## 2. Download the weights

Download `best.pt` from Google Drive:

**➡️ [best.pt (Google Drive)](https://drive.google.com/file/d/1clzWr8HuNYqReL5wQ8_jqkJXUV5kvRS5/view?usp=share_link)**

Put it anywhere and pass its path with `--ckpt`.

## 3. Required Inputs Setup

### 3a. Input table (Excel or CSV):

Make one row per nodule. If a CT contains several nodules, list each one on its own row. 

**Required columns** (the names must match exactly):

| Column | Description |
|---|---|
| `pid` | Patient / case ID. Used to name the output files. |
| `image_path` | Absolute path to the CT volume (`.nii.gz` or `.nii`) |
| `coordX` | Nodule x coordinate |
| `coordY` | Nodule y coordinate |
| `coordZ` | Nodule z coordinate |

## 4. Running inference

**If you want to just output masks:**

```bash
CUDA_VISIBLE_DEVICES=0 python infer_test_cohort.py \
    --csv /path/to/nodules.xlsx \
    --ckpt /path/to/weights/best.pt \
    --out_dir /path/to/results/
```

**With ground-truth masks (to compute Dice etc.):**

```bash
CUDA_VISIBLE_DEVICES=0 python infer_test_cohort.py \
    --csv /path/to/nodules.xlsx --mask_col mask_path \
    --ckpt /path/to/weights/best.pt --out_dir /path/to/output/
```

**Quick test on the first 5 CTs:** add `--limit 5`.

### Other options

| Option | Default | Description |
|---|---|---|
| `--coord_system` | `lps` | `lps`, `ras`, `voxel`, `auto` (see §3b) |
| `--coord_cols` | `coordX,coordY,coordZ` | Names of the coordinate columns |
| `--id_cols` | `pid,study_yr,nodule_id` | Columns used to build each nodule's ID (whichever of these exist) |
| `--thr` | `0.5` | Probability threshold |
| `--keep` | `center` | `center` keeps only the component nearest the given coordinate (best for volumetry). `largest` keeps the largest one. `all` keeps every component. |
| `--min_voxels` / `--min_slices` | `8` / `2` | Discard components smaller than this |
| `--tta` | off | Flip test-time augmentation (about 4× slower, sometimes slightly better) |
| `--no_adaptive_fov` | off | Disable the 128³ second pass for nodules touching the crop border |
| `--no_png` | off | Skip the PNG export (faster, uses less disk) |
| `--png_slices` | `union` | `union` = only slices with a prediction. `all` = every slice in the crop. |
| `--batch_size` | `1024` | Axial slices per GPU batch. Lower it if you run out of GPU memory. |
| `--load_workers` / `--save_workers` | `8` / `8` | CPU processes for loading and saving |

Run `python infer_test_cohort.py -h` for the full list.

---

## 5. Output

The output folder looks like this:

```
results/
├── pred_nifti/                     ← final masks (original CT space)
│   ├── <uid>_pred.nii.gz           one binary mask per nodule
│   └── <scan_id>_pred_all.nii.gz   union of all nodules in that CT (CT inputs only)
├── pred_pt/                        ← final masks for .pt inputs
│   └── <uid>_pred.pt               uint8 tensor, same shape & axis order as the input cube
├── crops_64/                       ← 1 mm nodule crops (64³)
│   ├── <uid>_img.nii.gz            CT crop (HU)
│   ├── <uid>_pred.nii.gz           predicted mask
│   ├── <uid>_prob.nii.gz           probability map
│   └── <uid>_gt.nii.gz             ground truth (only with --mask_col)
├── crops_128/                      ← same, for nodules re-run with the expanded 128³ crop
├── png/<uid>/z###.png              ← one QC image per axial slice
├── metrics_per_nodule.csv          ← one row per nodule (see below)
├── coord_check.csv                 ← result of the coordinate-convention check
├── run_config.json                 ← all arguments used for the run
└── failed_nodules.csv              ← only written if some nodules failed
```

**Output names.** The `<uid>` is built from your ID columns: `pid`, `T<study_yr>`, `n<nodule_id>` (e.g. `P001_T1_n2`). Any characters that aren't safe in file names are replaced with `_`. If two rows would get the same ID, a `_r<row>` suffix is added. Add a `nodule_id` column to avoid this.

### `metrics_per_nodule.csv`

This file contains all the columns from your input table, plus these:

| Column | Meaning |
|---|---|
| `uid`, `scan_id` | Nodule ID / CT ID |
| `pred_vol_mm3` | Predicted volume (on the 1 mm grid) |
| `pred_vol_mm3_orig_space` | Predicted volume measured on the original CT grid |
| `equiv_sphere_diam_mm` | Diameter of a sphere with the same volume |
| `max_axial_diam_mm` | Longest in-plane diameter over all axial slices |
| `n_pred_slices` | Number of 1 mm axial slices containing the nodule |
| `empty_prediction` | `True` if nothing was segmented |
| `center_in_pred`, `center_to_pred_mm` | Whether the given coordinate lies within 2 mm of the mask, and the distance |
| `center_in_bounds`, `hu_at_center` | Whether the coordinate falls inside the image, and the HU value there |
| `prob_at_center`, `max_prob` | Model confidence |
| `touches_border`, `expanded_fov`, `touches_border_after_expansion` | Adaptive-FOV flags. If a mask still touches the border after expansion, the nodule may be cut off. |
| `fov_limited_by_cube` | `.pt` inputs only: the mask touches the cube edge and there is no larger context to expand into |
| `dice`, `iou`, `precision`, `recall`, `hd95_mm`, `gt_vol_mm3`, … | Only with `--mask_col` |

At the end of the run, a summary is printed to the terminal. It shows the number of nodules processed, empty predictions, coordinates that fell outside the predicted mask or the CT, the median predicted volume, and the mean Dice if ground truth was given.
