# respiratorySSAMpy
Statistical shape and appearance model for airways and lung lobes

## Usage

### Installation
First, install the required dependencies (best practice is to use a virtual environment)
```bash
conda create --name ssam_env python=3.10
conda activate ssam_env
pip install hjson matplotlib networkx nevergrad numpy pyssam scikit-image scikit-learn scipy vedo
```
Download the data (**ADD LINK**)

### Running the reconstruction script

```bash
python reconstructResp_nofissures.py -c config_nofissures_gaussblur_2proj.json
```
