# Lung Nodule Segmentation: Inference

A 2D V-Net that segments lung nodules on CT. You give it CT volumes (`.nii` / `.nii.gz`) and the location of each nodule. It returns a binary mask for each nodule on the original CT grid, along with volume and diameter measurements and QC images.

Each nodule is processed in these steps:

1. The CT is reoriented to LPS and resampled to 1 mm isotropic. HU values are clipped to [-1000, 400].
2. A 64 × 64 × 64 crop is taken around the nodule coordinate you supplied.
3. Every axial slice of the crop is segmented by the 2D V-Net.
4. The probabilities are thresholded at 0.5. Holes are filled, and the 3D connected component closest to your coordinate is kept.
5. **Adaptive field of view:** if the mask touches the crop border (usually a large nodule), the nodule is run again with a 128 × 128 × 128 crop centred on the first-pass mask.
6. The mask is resampled back onto the original CT grid and written to disk.

---

## 1. Installation

You'll need Linux (the worker pools use `fork`) and Python ≥ 3.9. A CUDA GPU is recommended. The code also runs on CPU, but slowly.

```bash
git clone https://github.com/<your-username>/<this-repo>.git
cd <this-repo>

pip install torch numpy pandas scipy SimpleITK tqdm matplotlib openpyxl
```

`openpyxl` is only needed if you give the input table as an Excel file.

Keep these four files in the same folder:

| File | Purpose |
|---|---|
| `infer_test_cohort.py` | Entry point (run this) |
| `infer_lib.py` | Model loading, prediction, post-processing, metrics, PNGs |
| `common.py` | I/O, reorientation, resampling, cropping |
| `vnet2d.py` | Model definition |

## 2. Download the weights

Download `best.pt` from Google Drive:

**➡️ [best.pt (Google Drive)](https://drive.google.com/REPLACE_WITH_YOUR_LINK)**

Put it anywhere and pass its path with `--ckpt`. For example, `weights/best.pt`:

```bash
mkdir -p weights
# optional command-line download:
pip install gdown
gdown --fuzzy "https://drive.google.com/REPLACE_WITH_YOUR_LINK" -O weights/best.pt
```

---

## 3. Input

### 3a. Input table (Excel or CSV): the standard way

Make one row per nodule. If a CT contains several nodules, list each one on its own row with the same `image_path`. Each CT is loaded only once.

**Required columns** (the names must match exactly):

| Column | Description |
|---|---|
| `pid` | Patient / case ID. Used to name the output files. |
| `image_path` | Absolute path to the CT volume (`.nii.gz` or `.nii`) |
| `coordX` | Nodule x coordinate |
| `coordY` | Nodule y coordinate |
| `coordZ` | Nodule z coordinate |

**Optional columns:**

| Column | Description |
|---|---|
| `nodule_id` | Recommended when a CT has more than one nodule. Gives readable output names such as `P001_n2`. |
| `study_yr` | Timepoint / study year, for longitudinal data. Gives output names such as `P001_T1_n2`. |
| `mask_path` | Ground-truth mask (`.nii.gz`). Turns on Dice / IoU / HD95 and adds GT panels to the PNGs. Activate it with `--mask_col mask_path`. |

You can also add any other columns (diagnosis, size category, …). They are copied through to `metrics_per_nodule.csv`.

**Example (`nodules.xlsx`):**

| pid | nodule_id | image_path | coordX | coordY | coordZ |
|---|---|---|---|---|---|
| P001 | 1 | /data/ct/P001.nii.gz | -85.3 | 42.1 | -160.5 |
| P001 | 2 | /data/ct/P001.nii.gz | 102.7 | -10.4 | -120.0 |
| P002 | 1 | /data/ct/P002.nii.gz | 60.2 | 15.9 | -98.3 |

If your coordinate columns have other names, pass them with `--coord_cols x,y,z`.

### 3b. Coordinate system ⚠️

By default the coordinates are **world (physical) coordinates in millimetres, in LPS convention**. This is the convention used by DICOM, ITK and SimpleITK, and the same space as `image.TransformIndexToPhysicalPoint(...)`. The coordinate should be at, or close to, the nodule centre.

Choose another convention with `--coord_system`:

| Value | Meaning |
|---|---|
| `lps` (default) | World mm, LPS (DICOM / ITK / SimpleITK) |
| `ras` | World mm, RAS (e.g. positions read off **3D Slicer**). Same as LPS with x and y negated. |
| `voxel` | 0-based voxel index `(i, j, k)` = (column, row, slice) in the original NIfTI. These are the same indices as a nibabel array `img.get_fdata()[i, j, k]`. |
| `auto` | Pick whichever convention fits the data best (see below) |

**Built-in safety check.** Before the full run, the script samples 30 CTs (`--probe_scans`). It measures how often each convention lands the coordinate on dense, nodule-like tissue (mean HU > −600) and prints a table. If the convention you chose fits clearly worse than another one, the script **stops** rather than segmenting the wrong locations. It then tells you which `--coord_system` to use. The details are saved to `coord_check.csv`. You can skip the check with `--skip_coord_check`, but only do that if you're sure.

### 3c. Pre-cropped `.pt` / `.pth` cubes (alternative)

If your nodules are already cropped into cubes saved as PyTorch tensors, point the script at the folder instead. You don't need a table or coordinates.

* Each file holds **one cube, in raw HU**. It can be a tensor, a numpy array, or a dict containing one of them (keys `image`, `img`, `ct`, `volume`, `vol`, `data`, `x`, `cube`).
* By default the nodule is assumed to be at the **centre of the cube**. The axis order is `(z, y, x)` = (slices, rows, cols), with 1 mm spacing. Change these with `--pt_axes xyz` and `--pt_spacing x,y,z`.
* The output names come from the input file names.

---

## 4. Running inference

**From a table of CTs:**

```bash
CUDA_VISIBLE_DEVICES=0 python infer_test_cohort.py \
    --csv nodules.xlsx \
    --ckpt weights/best.pt \
    --out_dir results/
```

**With ground-truth masks (to compute Dice etc.):**

```bash
CUDA_VISIBLE_DEVICES=0 python infer_test_cohort.py \
    --csv nodules.xlsx --mask_col mask_path \
    --ckpt weights/best.pt --out_dir results/
```

**From a folder of pre-cropped `.pt` cubes:**

```bash
CUDA_VISIBLE_DEVICES=0 python infer_test_cohort.py \
    --pt_dir /path/to/volumes \
    --ckpt weights/best.pt \
    --out_dir /path/to/pred_volumes
```

**Quick test on the first 5 CTs:** add `--limit 5`.

### Useful options

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

**If you just want the masks,** use `pred_nifti/` (or `pred_pt/` for `.pt` inputs).

* `pred_nifti/*.nii.gz` files are on the **original CT grid**: same size, spacing, origin and direction as the input. They overlay directly on the CT in ITK-SNAP, 3D Slicer, etc.
* The files in `crops_*/` are on the 1 mm resampled grid. They still carry real-world geometry, so they also overlay correctly on the original CT.
* In the PNGs, the prediction is outlined in orange and the ground truth in blue. A white `+` marks the coordinate you supplied.

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

### Things to check

* **Many `empty_prediction` or `center_in_pred = False` rows** usually means the coordinates are wrong (wrong convention or units). Check `coord_check.csv` and the PNGs.
* **`touches_border_after_expansion = True`** means the nodule is larger than the 128 mm crop or is attached to a large structure. Inspect these cases visually.

---

## Citation / contact

<!-- Add paper / citation / contact info here -->
