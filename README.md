# Proxy model

This model is designed to predict a protected group (race, gender...) based on a classification model trained on 
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
| load_data/use_features | list of columns to read (Use same case as in the actual dataset |    - AGE, -EDUC |
| load_data/load_params | right-aligned |    $1 |
| load_data/select_columns | right-aligned |    $1 |
| load_data/select_columns | right-aligned |    $1 |
| load_data/select_columns | right-aligned |    $1 |
            
`NOTE`

> This data is totally fake. Consequently, the evaluation metrics values are meaningless. The goal is not to improve them!
