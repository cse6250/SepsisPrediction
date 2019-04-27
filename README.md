# SepsisPrediction
This is the final project of the CSE6250. We mainly use the MIMIC-III data to predict the sepsis.

### Data Preprocess
Set up environment by ```conda env create -f environment.yml python=3.6```
1. Run ```sbt compile run``` in [mimic](/mimic) and generate SOFA timeline information
2. Run ```python get_sepsis_onset_time.py``` in [data_preprocess](/src/data_preprocess) to retrieve ICU stays with sepsis and corresponding onset time
3. Run ```python data_preprocess.py``` in [data_preprocess](/src/data_preprocess) to get labeled pivoted vital data ready for model training
