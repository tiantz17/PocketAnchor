# PocketAnchor

Learning Structure-based Subpocket Representations for Protein-Ligand Interaction Prediction

<div><img width=200 src=https://github.com/tiantz17/PocketAnchor/blob/main/figure/pocketanchor.png></div>

## Requirements

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

## Reproducing results

1. Prepare the environment that satisfying the above requirements;
2. Download the trained model files:

    - [PocketAnchor-models.zip (105 MB)](https://drive.google.com/file/d/11xe9xqG4oSScXpJhYYOBH_KAnG4mlCKf/view?usp=sharing)

3. Download the input data files:

    - PocketAnchor-data-AffinityInteractionDistance.zip (5.68GB) (TBA)

    - PocketAnchor-data-PocketDetection.zip (2.82GB) (TBA)

4. Run the following inference scripts;
5. The results can be find in ```[TASK]/results/[FOLDER]/```.


### 1. PocketAnchor-site
Protein ligand binding site prediction

```
python runPrediction.py --task PocketDetection --dataset COACH420
python runPrediction.py --task PocketDetection --dataset HOLO4k
```

### 2. PocketAnchor-affinity
Protein-ligand binding affinity prediction

```
python runPrediction.py --task Affinity --setting original
python runPrediction.py --task Affinity --setting newprotein
python runPrediction.py --task Affinity --setting expanded
```

### 3. PocketAnchor-interaction
Non-covalent interaction prediction

```
python runPrediction.py --task Interaction
```

### 4. PocketAnchor-distance
Protein ligand binding pose classfication

```
python runPrediction.py --task Distance
```
