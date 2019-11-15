# Embedding and learning with signatures

This is the code accompanying the article "Embedding and learning with 
signature". It is based on three recent datasets, called Quick, Draw!, Motion
Sense and Urban Sound. They can be found at the following addresses:

* Quick, Draw!: https://www.kaggle.com/c/quickdraw-doodle-recognition/data
* Motion Sense: https://www.kaggle.com/malekzadeh/motionsense-dataset
* Urban Sound: http://urbansounddataset.weebly.com

The path to the directory where data is saved must be filled in the variable
data_dir in read_format.py.

The code enables to test the combination signature + machine learning algorithm
with various embedding, truncation order of signatures and algorithms.

## Environment

All packages needed to run the scripts are listed in requirements.txt. You can
install them with

```python
pip install -r requirements.txt
```

## Experiments

### A single experiment

The file script_train_sig.py enables to launch one experiment, that is one
training of a dataset with one embedding, one truncation order and one
algorithm. For example, typing in a shell the following command

```bash
python script_train_sig.py quick_draw time random_forest 2
```

will train a random forest on the dataset Quick, Draw! embedded with the time
path and signature truncated at order 2. The possible arguments to the script
are the following

* dataset: quick_draw, motion_sense, urban_sound
* embedding: raw, time, lead_lag, rectilinear. If dataset=quick_draw, then
stroke_1, stroke_2 and stroke_3 are also allowed.
* algorithm: neural_network, random_forest, nearest_neighbors and xgboost.

If the embedding is lead_lag then the lag must be given after the truncation
order. For example

```bash
python script_train_sig.py quick_draw lead_lag random_forest 2 1
```

will use the lead-lag embedding with a lag of 1 and signatures truncated at
order 2.

### Comparison of embeddings

This script script_comparison_all_embeddings.py can be launched to get all 
results of embeddings comparison on Figures ?????. You just need to choose a
dataset launch

```bash
python script_comparison_all_embeddings quick_draw
```

to get a csv files with all results for the Quick, Draw! dataset. Its columns
are ['accuracy','embedding','algo','order','n_features]. Similarly it can be
launched with motion_sense or urban_sound as argument to get the results of the
other dataset.


### Comparison of dyadic partitions

The script script_dyadic_study.py launches the experiments necessary to compare
dyadic partitions (see Figure ????). It can be launched with as argument the
desired dataset (quick_draw, motion_sense or urban_sound). For example, to get
the results for Quick, Draw! :

```bash
python script_dyadic_study.py quick_draw
```

It outputs a csv file with all the results, with columns
['accuracy','embedding','algo','order','dyadic_level','n_features].

### Lag selection

Similarly, the script script_ll_selection.py launches the experiments necessary
to compare different lags (see Figure ???). It can be launched with as argument 
the desired dataset (quick_draw, motion_sense or urban_sound). For example, to
get the results for Quick, Draw! :

```bash
python script_ll_selection.py quick_draw
```

It outputs a csv file with all the results, with columns
['accuracy','embedding','algo','order','ll','n_features].







