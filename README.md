# Yawn Detector

# Setup

## Installation

1. Install make
2. Install CUDA 11.5. Please follow this guide: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html
3. Install CUDNN. Please follow this guide. https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html
4. Install poetry. Please see guide here: https://python-poetry.org/docs/
5. Install cmake: `apt-get install cmake build-essential`
5. Run `make dev`


# ML Model

## Experimentation setup

* Run the command `dvc pull` to download all data needed for running experiments as well as saved models

## Data

Data can be found in the following directories:

* `data/` - Contains raw data and manually separated video frames for yawn and non-yawn
* `pipeline_outs/` - Contains data from pipeline artifacts like results, ml model weights, preprocessed data, etc


## DAG / Pipeline Setup
+----------------+ 
| data/split.dvc | 
+----------------+ 
         *         
         *         
         *         
  +------------+   
  | preprocess |   
  +------------+   
         *         
         *         
         *         
    +-------+      
    | train |      
    +-------+      
+--------------+ 
| data/raw.dvc | 
+--------------+ 

* preprocess - Runs preprocessing step on data
* train - trains data

## Running Experiments

1. To run a new experiment, edit `params.yaml` and modify parameters.
2. To run training pipeline, run `dvc repro`
3. To display results, run `dvc metrics show`
4. To save experiment, run `git add`, `git commit` and `git push` for all modified data.
5. To view all experiment results, run `dvc exp show -T`