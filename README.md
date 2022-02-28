## Deep Image Prior and Neural Cellular Automata Nodule Synthesis
Main steps of the study "Patient-Specific 3d Cellular Automata Nodule Growth Synthesis In Lung Cancer Without The Need Of External Data (2021)".

#### Data preprocessing:
```bash
pip install -r requirements_inpainting.txt
python process_lungs_and_segmentations.py
```

#### Nodule inpainting with deep image prior
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