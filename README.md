# SepsisPrediction
This is the final project of the CSE6250. We mainly use the MIMIC-III data to predict the sepsis.

## ML model:
Please put the code named mimic_ml.py at the same level with data/

Currently, the arguments are fixed as:
traindata = pd.read_csv('data/sepsis/train_cleaned_pivoted_vital.csv')
testdata = pd.read_csv('data/sepsis/test_cleaned_pivoted_vital.csv')
prediction_window = 6

I will modify it later to make these three as the input arguments.
