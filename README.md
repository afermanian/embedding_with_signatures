# Embedding and learning with signatures

This is the code accompanying the article [1]. It is based on three recent
datasets, called Quick, Draw!, MotionSense and Urban Sound. They can be found at
the following links:

* Quick, Draw!: https://www.kaggle.com/c/quickdraw-doodle-recognition/data
* Motion Sense: https://www.kaggle.com/malekzadeh/motionsense-dataset
* Urban Sound: http://urbansounddataset.weebly.com

The path to the directory where these datasets are saved must be filled in the
variable data_dir in read_format.py, preprocessing.py and
script_train_quick_draw_generator.py.

The code enables to test the combination signature + machine learning algorithm
with various embedding, truncation order of signatures and algorithms. It also
reproduces all results presented in the article [1].

## First steps

### Packages

All packages needed to run the scripts are listed in requirements.txt. You can
install them with

```python
pip install -r requirements.txt
```

### Data preprocessing

Before doing some experiments execute preprocessing.py to shuffle the
datasets Motion Sense and Urban Sound:

```bash
python preprocessing.py
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

* dataset: quick_draw, motion_sense, urban_sound, arma
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
results of embeddings comparison represented in Figures 9, 10, 11 and 15 in [1].
Choose a dataset and launch the script. For example,

```bash
python script_comparison_all_embeddings quick_draw
```

outputs a csv files with all results for the Quick, Draw! dataset. Its columns
are ['accuracy','embedding','algo','order','n_features]. Similarly it can be
launched with motion_sense or urban_sound as argument to get the results of the
other dataset.

### Autoregressive simulations

The study of the influence of the lag for different AR(p) processes (Figure 16) may be obtained by running the following script:

```bash
python script_arma_ll_selection.py
```

The study of the influence of the truncation order and the sample size (Figures 17 and 18) may be obtained with

```bash
python script_arma_sample_size.py
```

### Comparison of dyadic partitions

The script script_dyadic_study.py launches the experiments necessary to compare
dyadic partitions (Figure 19). It can be launched with as argument the
desired dataset (quick_draw, motion_sense or urban_sound). For example, to get
the results for Quick, Draw! :

```bash
python script_dyadic_study.py quick_draw
```

It outputs a csv file with all the results, with columns
['accuracy','embedding','algo','order','dyadic_level','n_features].

### Lag selection

Similarly, the script script_ll_selection.py launches the experiments necessary
to compare different lags (see Figure 20). It can be launched with as argument 
the desired dataset (quick_draw, motion_sense or urban_sound). For example, to
get the results for Quick, Draw! :

```bash
python script_ll_selection.py quick_draw
```

It outputs a csv file with all the results, with columns
['accuracy','embedding','algo','order','ll','n_features].


### Performance of the signature

* For Motion Sense, a F1 score of 93.5 is obtained with an XGBoost classifier, a
lead lag embedding and a signature truncated at order 3. It is obtained by
running:

```bash
python script_train_sig.py motion_sense lead_lag xgboost 3 1
```

* For Urban Sound, the hyperparameters of a random forest have been tuned with a
randomized grid search. The best params are stored in the variable custom_params
in script_train. Uncomment the corresponding lines and train a random forest
with these parameters by launching:

```bash
python script_train_sig.py urban_sound lead_lag random_forest 5 5
```
This yields an accuracy of 70.2

* For Quick, Draw!, it is necessary to train a neural network with more data,
which is handled by a data generator defined in quic_draw_generator.py. Then,
script_train_sig_generator.py trains the model. Be aware that this script
demands more computational capacities. Before doing so, some preprocessing is
necessary to save a normalization vector. Run the script
preprocessing_generator_quic_draw.py to do so. It will save this vector as a
.npy array.

## References

[1]: Fermanian, A. (2019) [Embedding and learning with signatures](https://arxiv.org/abs/1911.13211)








