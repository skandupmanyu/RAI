# Proxy model

This model is designed to predict a protected group (race, gender,...) on a given dataset based on a classification model trained on 
IPUMS dataset as it is consider as a superior dataset for fairness assessment. 

## 1. Description

### 1.1 Business case

Users of this model are insurances that seek to assess fairness and equity from their models. The
algorithm allows to tailor a model depending on the use case(feature selection, algorithm parameters).
We have also created a set of metrics to assess performances of the model

- Use case:
Based on historical data, our ML model must be able to predict the race (e.g. hispanic) of their customers,
thus evaluating the impact of one change of their solution on one group compared to others. 

### 1.2 Data

The `data/` directory contains all data used and generated from the model.
 
- `raw` contains the IPUMS dataset on different formats
- `intermediate` contains different feature set from models variations
- `artefacts` contains model outputs (model,evaluation metrics, figures...)
- `input` contains the dataset to predict based on the trained model 

### 1.3 Config

This folder contains the configuration file **config.yml**. It is the entry point for the user. This 
file contains all the detailed parameters for different variations of the model. 


| Parameter   |     Description     |  Values |
|----------|:-------------:|------:|
| model_name |  Name of folder created from running `config` | proxy_model_hispanic_income |
| input_data |    Name of the training raw dataset located in `data/raw`   |   usa_00004.csv.gz |
| load_data/select_columns | True if only select a set of columns while loading the dataset |   True/False |
| load_data/use_features | list of columns to read when reading the raw files (Use same case as in the actual dataset) |    - AGE, -EDUC |
| load_data/load_params | parameters of the pandas read_csv method | compression: 'gzip', header: 0, sep: ',' |
| sample_size | sample size of the training set | 200000 |
| use_msa | Whether to use the msa dataset to build the features set | True |
| features |  |  |
| features |  |  |
| pg_target | protected group target variable | 'hispanic','black' |
| rai_target | toy model target variable | 'inctot' |
| threshold_rai_target | Percentile of the distribution that we consider as high income |  |
| bias_ranker/pos_rate |  |  |
| bias_ranker/bias_tolerance |  |  |
| latest_year_dataset | last year of training dataset | 2019 |
| model/name | name of the sklearn classification model used to train the pg model | 'LGBMClassifier' |
| model/params | pg classification model parameters | max_depth: 3 n_estimators: 1000 |
| model/grid_search | pg classification model grid search parameters  | n_estimators:  [100,200,400] |
| test_cutoff | train_test_split parameter | 0.8 |
| split_random_size | random number generated for repeatability of training split | 42 |
| proportion_pg_traget_over_population | expected proportion of pg_target on new prediction data, used for pred probabilities cutoff  | 0.1 |
| prediction_data | name of the dataset to predict on (JSON format and same structure as the features set) | 'ipums_default_met_sex_age_marst_educ.json' |
| 

## 2. How to run

The algorithm could be run both from the command line inside the folder RAI or directly in an editor inside functions. 
There are 4 main functions:
- run_dataset: create the features set through the command `python -m src dataset`
- run_train: train the pg model then compute fairness metrics on the actual dataset through the command `python -m src train`
- run_predict: predict pg variables on the given prediction data through the command `python -m src predict`
- run_all: run the 3 previous steps successively through the command `python -m src run`

## 3. Training sets

The default dataset is `usa_00004.csv.gz`. This data is a comprehensive dataset with a wide taxonomy for all the features. We intend to create a model that will match with the majority of our prospective clients. 
Hence, we selected the most probable variables and standardized their values. This will lead to extra work on the client's side for the sake of consistency and repeatability.
We used the notebook `process ipums` to create a standardize version of the dataset. Please keep in mind that the prediction set should the exact shape and values of the feature set.
