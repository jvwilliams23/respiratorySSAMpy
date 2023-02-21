# respiratorySSAMpy

## Overview

This project aimed to use a statistical shape and appearance model (SSAM) to automatically generate 3D airway and lung shapes from a single 2D chest X-ray image. 

We use a point cloud of landmarks with digitally reconstructed radiographs (DRR) to create a SSAM that describes the shape and appearance correlation across a population. The SSAM parameters are then iteratively adapted to create a new shape which matches a new (unseen) X-ray image. The fit of the generated shape is evaluated with regards to the outline of the lung edge-map, the fit of the modelled appearance to the X-ray's appearance and other metrics.

## Usage

### Installation
First, install the required dependencies (best practice is to use a virtual environment)
```bash
conda create --name ssam_env python=3.10
conda activate ssam_env
pip install hjson matplotlib networkx nevergrad numpy pyssam scikit-image scikit-learn scipy vedo
```
Download the data (**ADD LINK**)

### Running the 2D-to-3D reconstruction script

```bash
python reconstructResp_nofissures.py -c config_nofissures_gaussblur_2proj.json
```

## Get Help
Please submit an issue to the issues panel on this repository.

## Citing this repository
If you use the code or models in this repository, please cite our paper
```
@article{TODO}
```
