# ET-Tox

Adapted from https://github.com/torchmd/torchmd-net

## Installation

1. Clone the repository:
    ```
    git clone https://github.com/torchmd/ET-Tox.git
    cd ET-Tox
    ```

2. Install Mambaforge (https://github.com/conda-forge/miniforge/#mambaforge). It is recommended to use `mamba` rather than `conda`. `conda` is known to produce broken enviroments with PyTorch.

3. Create an environment and activate it:
    ```
    mamba env create -f environment.yml
    mamba activate et_tox
    ```

4. Install TorchMD-NET into the environment:
    ```
    pip install -e .
    ```

## Cite
```
Cremer J, Sandonas LM, Tkatchenko A, Clevert D-A, de Fabritiis G. Equivariant Graph Neural Networks for Toxicity Prediction. ChemRxiv. Cambridge: Cambridge Open Engage; 2023;  This content is a preprint and has not been peer-reviewed.
```

## Data
The data must be downloaded here: https://zenodo.org/record/7942946, and unpacked into the respective data folder (files containing "tdc" into data/TDCTox, files containing "mnet" into data/MoleculeNet)

## Usage
Specifying training arguments can either be done via a configuration yaml file or through command line arguments directly. Example yaml files can be found in "train_yaml" folder. GPUs can be selected by their index by listing the device IDs (coming from `nvidia-smi`) in the `CUDA_VISIBLE_DEVICES` environment variable. Otherwise, the argument `--ngpus` can be used to select the number of GPUs to train on (-1 uses all available GPUs or the ones specified in `CUDA_VISIBLE_DEVICES`).
For training on random splits:
```
mkdir output_dir
CUDA_VISIBLE_DEVICES=0 python train.py --conf train_yaml/tox21.yaml --log-dir output_dir/
```
For training on scaffold splits:
```
mkdir output_dir
python train.py --conf ./train_yaml/"${name}".yaml --log-dir output_dir/ --seed "${seed}" --splits ./data/TDCTox/splits/"${name}"_split_1_scaffold.npz --use-energy-feature false --dataset-split scaffold
```

```
mkdir output_dir
python train.py --conf ./train_yaml/"${name}".yaml --log-dir output_dir/ --seed "${seed}" --splits ./data/MoleculeNet/splits/"${name}"_seed"${seed}"_confs1_scaffold.npz
```

For training with multiple conformers per molecule (specify num_confs) on a scaffold split on TDC:
```
mkdir output_dir
python train.py --conf ./train_yaml/"${name}".yaml --log-dir output_dir/ --seed "${seed}" --splits ./data/TDCTox/splits/"${name}"_seed"${seed}"_confs"${num_confs}"_scaffold.npz --dataset-split scaffold
```

```
mkdir output_dir
python train.py --conf ./train_yaml/"${name}".yaml --log-dir output_dir"${name}" --seed "${seed}" --splits ./data/MoleculeNet/splits/"${name}"_seed"${seed}"_confs$"{num_confs}"_scaffold.npz
```

## Pretrained models
Pretrained models are available at https://zenodo.org/record/7942946

## Usage
Unpack models into ./models and either test the models in a jupyter notebook environment by using test_models.ipynb or give the --test-run flag and specify --test-checkpoint model_path


## Acknowledgement
This study was partially funded by the European Union’s Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie Actions grant agreement “Advanced machine learning for Innovative Drug Discovery (AIDD)” No. 956832.