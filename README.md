# PocketAnchor

Learning Structure-based Pocket Representations for Protein-Ligand Interaction Prediction

<div><img width=200 src=https://github.com/tiantz17/PocketAnchor/blob/main/figure/pocketanchor.png></div>

# Requirements

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

# Reproducing results

1. Prepare a environment that satisfying the above requirements;
2. Download the trained model files:

    - [PocketAnchor-models.zip (59.1MB)](https://drive.google.com/file/d/1FGhtZgH18F6JHz-IB_Wqm6pbHDdIfk3-/view?usp=sharing)

3. Download the input data files:

    - [PocketAnchor-data-Affinity.zip (5.69GB)](https://drive.google.com/file/d/1yLzUmqkJDtEH8b22VMkUjoK0c60TyHGR/view?usp=sharing)

    - [PocketAnchor-data-PocketDetection.zip (3.66GB)](https://drive.google.com/file/d/1tHkmsKXVrr4w08S2ZY7uwlhPj7qimuah/view?usp=sharing)

4. Run the inference scripts below (run time about 10 mins);
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
python runPrediction.py --task Affinity --setting original
python runPrediction.py --task Affinity --setting newprotein
python runPrediction.py --task Affinity --setting expanded
```
