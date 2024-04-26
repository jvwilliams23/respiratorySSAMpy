# respiratorySSAMpy

[![DOI](https://zenodo.org/badge/362392583.svg)](https://zenodo.org/badge/latestdoi/362392583)

## Overview

This project aimed to use a statistical shape and appearance model (SSAM) to automatically generate 3D airway and lung shapes from a single 2D chest X-ray image. 

We use a point cloud of landmarks with digitally reconstructed radiographs (DRR) to create a SSAM that describes the shape and appearance correlation across a population. The SSAM parameters are then iteratively adapted to create a new shape which matches a new (unseen) X-ray image. The fit of the generated shape is evaluated with regards to the outline of the lung edge-map, the fit of the modelled appearance to the X-ray's appearance and other metrics.

Data to reproduce the results is now available in the `dataset/` directory.

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
@article{williams2024validated,
  title={Validated respiratory drug deposition predictions from 2D and 3D medical images with statistical shape models and convolutional neural networks},
  author={Williams, Josh and Ahlqvist, Haavard and Cunningham, Alexander and Kirby, Andrew and Katz, Ira and Fleming, John and Conway, Joy and Cunningham, Steve and Ozel, Ali and Wolfram, Uwe},
  journal={Plos one},
  volume={19},
  number={1},
  pages={e0297437},
  year={2024},
  publisher={Public Library of Science San Francisco, CA USA}
}
```
