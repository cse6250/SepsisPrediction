# SepsisPrediction
> This is the final project of the CSE6250 Big Data in health. We use the MIMIC-III data to explore sepsis prediction.

### Architecture
```
.
│───data     // train, validation, test data and processed data
│
|───mimic     // scala codes for generating SOFA timeline information
|
|───out     // best models and some result images
|
|───src
│   └───data_preprocess     // calculate onset time and subsample data
|   |
|   └───etl_data    // transform data into sequences
|   |
|   └───sepsis_prediction_lstm    // codes for LSTM model
|   |
|   └───sepsis_prediction_ml    // codes for 4 machine learning models
│   
└───environment.yml     // environment and dependencies
|
└───README.md
```

### Prerequisite
- Set up python environment by 
```
conda env create -f environment.yml python=3.6
```
- Scala 2.11

### Data Preprocess
1. Download MIMIC-III database in PostgreSQL on local 
2. Run the commands in the README.md in [sepsis_mimic](https://github.com/cse6250/mimic-code) repository to get pivoted vital data and infection_time
3. Run the codes in the concept folder of [mimic_code](https://github.com/cse6250/mimic-code) repository to get pivoted SOFA score
4. Run the codes in './mimic' and generate SOFA timeline information
5. Run ```get_sepsis_onset_time.py``` in './src/data_preprocess' to retrieve ICU stays with sepsis and corresponding onset time
6. Run ```python data_preprocess.py``` in './src/data_preprocess' to get labeled pivoted vital data ready for model training

After data preprocess, the processed data are in the './data/sepsis/train', './data/sepsis/validation' and './data/sepsis/test'.

\* We have provided the processed data in './data/sepsis', so you don't have to do the above complicated process.

### Prediction models
1. Run ```etl_sepsis_data.py``` in './src/etl_data' to construct the features sequence data for prediction models
2. Run ```train_sepsis.py``` in './src/sepsis\_prediction\_ml' to run the 4 machine learning models
3. Run ```train_sepsis.py``` in './src/sepsis\_prediction\_lstm' to run the deep learning models
