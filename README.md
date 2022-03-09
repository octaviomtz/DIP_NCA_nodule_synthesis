# Deep Image Prior and Neural Cellular Automata Nodule Synthesis
Main steps of the study "Patient-Specific 3d Cellular Automata Nodule Growth Synthesis In Lung Cancer Without The Need Of External Data (2021)".

## Main steps
1. LUNA16 Dataset Preprocessing
1. Apply deep image prior on nodules (lungs blocks 96x160x96)
1. Prepare dataset for GANs and pylidc characteristics 
1. Get only the nodule
1. Cellular Automata
1. Insert CA generated nodule for nodule detection


#### 1. Data preprocessing:
```bash
pip install -r requirements_develop.txt
python process_lungs_and_segmentations.py
```

```diff
- INPUTS:
- Path to LUNA16 dataset.
- Path to LUNA16 candidates: candidates_V2.csv
- Path to LUNA16 annotations: annotations.csv
- Path to LUNA16 segmentations: seg-lungs-LUNA16/ 
+ OUTPUTS (for each scan):
+ lungs_segmented
+ processed_images (cluster_id_images)
+ consensus_masks
+ maxvol_masks
```

#### 2. Nodule inpainting with deep image prior
Performs inpaiting on lung nodules using 2D convolutions
```bash
pip install -r requirements_inpainting.txt
python inpainting.py
```

#### Nodule synthesis with neural cellular automata
```bash
pip install -r requirements.txt
python neural_cellular_automata.py
```

#### nodule growing with image-to-image translation (cycleGAN)
```bash
pip install -r requirements_inpainting.txt
python cycleGAN.py
```

#### False positive Reduction
check https://github.com/octaviomtz/LUNA16_nodule_detection