# PocketAnchor
[![DOI](https://zenodo.org/badge/424266672.svg)](https://zenodo.org/badge/latestdoi/424266672)

Learning Structure-based Pocket Representations for Protein-Ligand Interaction Prediction.

<div><img width=200 src=https://github.com/tiantz17/PocketAnchor/blob/main/figure/pocketanchor.png></div>

The code for data processing can be found in [https://github.com/lishuya17/PocketAnchorData](https://github.com/lishuya17/PocketAnchorData).

The processed data can be found in docker image: [https://hub.docker.com/r/tiantz17/pocketanchor-models](https://hub.docker.com/r/tiantz17/pocketanchor-models). (Not recommended)

(Update) You can pull another docker image containing code, data, environment, trained models, and prediction results for reproduction: [https://hub.docker.com/r/tiantz17/pocketanchor](https://hub.docker.com/r/tiantz17/pocketanchor).

# 1. Requirements

```
cuda                11.2
python              3.7.4
torch               1.7.1
torch-geometric     1.6.3
numpy               1.19.0
pandas              1.2.4
rdkit               2020.03.3.0
scikit-learn        0.21.3 
scipy               1.6.3 
tensorboard         2.4.1
```

# 2. Reproducing results

1. Prepare a environment that satisfying the above requirements;
2. Download the trained model and the input data files in [docker](https://hub.docker.com/r/tiantz17/pocketanchor-models) image;
3. Extract the following files and unzip into this folder;
    - PocketAnchor-models.zip
    - PocketAnchor-data-Affinity.zip
    - PocketAnchor-data-PocketDetection.zip
4. Run the inference scripts below (run time ranges from a few minutes to a couple of hours depending on the size of dataset);
5. The results can be found in ```[TASK]/results/[FOLDER]/```.


## 1. PocketAnchor-site
Protein ligand binding site prediction

```
python runPrediction.py --task PocketDetection --dataset COACH420
python runPrediction.py --task PocketDetection --dataset HOLO4k
```

## 2. PocketAnchor-affinity
Protein-ligand binding affinity prediction

```
python runPrediction.py --task Affinity --dataset CASF --setting original --info original
python runPrediction.py --task Affinity --dataset CASF --setting newprotein --info newprotein
python runPrediction.py --task Affinity --dataset CASF --setting expanded --info expanded
```

# 3. Train PocketAnchor

1. Prepare a environment that satisfying the above requirements;
2. Generate anchor positions and the corresponding features of customized dataset following [PocketAnchorData](https://github.com/lishuya17/PocketAnchorData).
3. Run the training scripts below;
4. The trained models can be found in ```[TASK]/models/[FOLDER]```.


## 1. PocketAnchor-site
Protein ligand binding site prediction

```
python runTrain.py --task PocketDetection --dataset scPDB
```

## 2. PocketAnchor-affinity
Protein-ligand binding affinity prediction

```
python runTrain.py --task Affinity --dataset CASF --setting original --info original
python runTrain.py --task Affinity --dataset CASF --setting newprotein --info newprotein
```
