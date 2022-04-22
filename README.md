# Deep Image Prior and Neural Cellular Automata Nodule Synthesis
Main steps of the study "Patient-Specific 3d Cellular Automata Nodule Growth Synthesis In Lung Cancer Without The Need Of External Data (2021)".

## Main steps
1. LUNA16 Dataset Preprocessing
1. Nodule inpainting (on lungs blocks 96x160x96)
1. Get 64^3 blocks around the nodules
1. Nodule synthesys with Cellular Automata
1. Insert CA generated nodule for nodule detection


#### 1. Data preprocessing:
```bash
# in config/config_preprocessing.yaml select:
# input paths (LUNA16/subsets) & output paths (LUNA16/preprocessed_candidates)
# SUBSET and sample (SKIP_IDX) from where to start 
pip install -r requirements_develop.txt
python preprocessing.py
```
```diff
- INPUTS:
- Path to LUNA16 dataset.
- Path to LUNA16 candidates: candidates_V2.csv
- Path to LUNA16 annotations: annotations.csv
- Path to LUNA16 segmentations: seg-lungs-LUNA16/ 
+ OUTPUTS (preprocessed images for each scan):
+ lungs_segmented
+ processed_images (cluster_id_images)
+ consensus_masks
+ maxvol_masks
```

#### 2. Nodule inpainting with deep image prior
Performs inpaiting on lung nodules on lung blocks (96x160x96) using 2D convolutions
```bash
# in config/config_inpainting.yaml
# input paths (LUNA16/preprocessed_candidates) & output paths (LUNA16/inpainted_nodules)
# select SUBSET and sample (SKIP_IDX) from where to start
pip install -r requirements_inpainting.txt
python inpainting.py
```
```diff
- INPUTS:
- Preprocessed images for each scan (from 1.).
- Path to LUNA16 segmentations: seg-lungs-LUNA16/ 
+ OUTPUTS (for lung block of 96x160x96):
+ arrays/last/{id_series}_{lungL/R}_{ndls_n}
+ arrays/orig/{id_series}_{lungL/R}_{ndls_n}
+ arrays/masks/{id_series}_{lungL/R}_{ndls_n}
+ arrays/masks_nodules/{id_series}_{lungL/R}_{ndls_n}
+ arrays/masks_lungs/{id_series}_{lungL/R}_{ndls_n}
+ box_coords/{id_series}_{lungL/R}_{ndls_n} (coords containing the ndl)
```
<img src="figures_github/lungs_blocks.png" width="150" height="150" />

#### 3. Get only 64^3 blocks around the nodules from the (96x160x96) lung blocks
```bash
# in config/config_cube_around_inpainted_ndl.yaml select:
# input paths (LUNA16/inpainted_nodules) & output paths (LUNA16/inpainted_cubes_for_synthesis)
# select SUBSET and sample (SKIP_IDX) from where to start
pip install -r requirements_develop.txt
python cube_around_inpainted_ndl.py
```
```diff
- INPUTS: (lung blocks of 96x160x96):
- arrays/last
- arrays/orig
- arrays/masks
- arrays/masks_nodules
- arrays/masks_lungs
- box_coords
+ OUTPUTS (cubes of size 64x64x64):
+ original (original image)
+ inpainted_inserted (nodule inpainted inserted into original image)
+ mask
+ nodule_info
```
![image_synthesis](figures_github/cubes_32_size.png?raw=true) 

#### 3.1. (Optional) To check nodule inpainting results: 
```bash
pip install -r requirements_develop.txt
python utils/plot_inpainting_quality_control.py
```
![inpainting_QC](figures_github/inpain_qc_subset1_from_36.png?raw=true) 

#### 4. Nodule synthesis  with cellular automata
```bash
# in config/config_ca.yaml select:
# input paths (LUNA16/inpainted_cubes_for_synthesis) & output paths (LUNA16/synthesis_CA)
# select SUBSET and sample (SKIP_IDX) from where to start
pip install -r requirements_develop.txt
python cellular automata.py
```
```diff
- INPUTS (cubes of size 64x64x64):
- original (original image)
- mask
+ OUTPUTS (cubes of size 64x64x64):
+ 100 samples of nodule synthesis (the first ~60 can be used)
+ coordinates zyx from where the nodule was cropped 
```

#### TO DO nodule growing with image-to-image translation (cycleGAN)
```bash
pip install -r requirements_inpainting.txt
python cycleGAN.py
```

#### False positive Reduction
check https://github.com/octaviomtz/LUNA16_nodule_detection