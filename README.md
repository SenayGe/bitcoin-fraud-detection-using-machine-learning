# Bitcoin money laundering detection using ML

---
## Available modules and scripts
- models.py - defines implemented models
- data_preprocessing - elliptic dataset preprocessing
- classical_ml.py - Train and run the classical machine learning algorithms
- gcn_mlp.py - to trian and test GCN + MLP
- utils.py - helper functions

## Installation and setup
Clone repository  <br />
Download the Elliptic Bitcoin dataset from https://www.kaggle.com/ellipticco/elliptic-data-set and save the three .csv files under the same directory inside dataset/.
### Running the ml models
To `run` the classical ml algos:
```
python classical_ml.py
```
To `run` the GCN+MLP model:
```
python gcn_mlp.py
```
