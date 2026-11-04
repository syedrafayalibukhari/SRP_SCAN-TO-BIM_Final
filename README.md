# SRP Scan-to-BIM

Turns a terrestrial laser scan of an existing office floor (`.e57`) into an IFC4 BIM model, using open and license-free tools.

The model contains seven element types and the rooms:

| Element | IFC class |
|---|---|
| Floor | `IfcSlab` (FLOOR) |
| Ceiling | `IfcCovering` (CEILING) |
| Wall | `IfcWall` (STANDARD) |
| Door | `IfcDoor` |
| Window, including glass walls | `IfcWindow`, `IfcCurtainWall` |
| Column | `IfcColumn` |
| Partition (cubicle panel) | `IfcWall` (PARTITIONING) |
| Room | `IfcSpace` |

Glass panels and window bands are exported as `IfcCurtainWall`. A door is cut into its wall or glass panel, or fills the gap between two elements.

## Workflow

Each folder is one step, and each step's output is the next step's input. Every folder has a `HOW_TO_RUN.txt`.

| Folder | Step | Input | Output |
|---|---|---|---|
| `1_preprocessing` | Clean and voxel-downsample the scan | `.e57` scan | `preprocessed.ply`, `scan_positions.json` |
| `2_label_transfer` | Starting labels from the trained model; optional tiles | `preprocessed.ply` | `labelled/` |
| `3_label_correction` | Correct the labels by hand in the browser | files from `labelled/` | `..._corrected.ply` |
| `4_merge_tiles` | Join the corrected files | `..._corrected.ply` | `merged/floor_merged.ply` |
| `5_modeling` | Reconstruct the elements and write IFC | `floor_merged.ply`, `scan_positions.json` | `output/model.ifc` |
| `6_model_check` | Check the IFC against the schema and the labels | `model.ifc`, `floor_merged.ply` | `model_check_report.txt` |
| `7_ifc_viewer` | View and edit the model in the browser | `model.ifc` | - |

`scan2bim/` is the shared library. `tests/` holds the automated tests.

## Requirements

- Windows, macOS or Linux, 64-bit
- Python 3.9 to 3.11
- Google Chrome or Microsoft Edge with WebGL 2, and an internet connection, for steps 3 and 7 (both load their 3D libraries from the web)

The Python code runs on the CPU. Missing libraries are installed on the first run from `requirements.txt`.

## Quick start

```bash
cd 1_preprocessing
python preprocess_pointcloud.py            # copy the .e57 scan into this folder first

cd ../2_label_transfer
python label_transfer.py

# step 3: open 3_label_correction/point_label_corrector.html, correct each file in
#         2_label_transfer/labelled, and copy the saved ..._corrected.ply files
#         into 4_merge_tiles/corrected

cd ../4_merge_tiles
python merge_tiles.py

cd ../5_modeling
python run_modeling.py --project-name "Building A, level 3"

cd ../6_model_check
python check_model.py
```

Each script accepts `--help`.

## Main settings

| Setting | File | Default | Meaning |
|---|---|---|---|
| `VOXEL_SIZE_M` | `1_preprocessing/preprocess_pointcloud.py` | `0.02` | point spacing after downsampling (m) |
| `TILE_SIZE_M` | `2_label_transfer/label_transfer.py` | `8` | tile size for manual labelling (m); `0` = one file |

## Label classes

`0` floor, `1` ceiling, `2` wall, `3` door, `4` window (glass walls included), `5` column, `6` partition, `7` other, `255` unknown.

Furniture, equipment and clutter are labelled `other` and are not modelled. A floor labelled with the earlier twelve classes is converted with `2_label_transfer/convert_legacy_labels.py`.

## Tests

```bash
python -m pip install pytest
python -m pytest tests
```

The tests generate their own small floor, so they need no scan data.

## Data

Scans, point clouds and models are large and are not stored in this repository.
