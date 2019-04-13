import os
import pandas as pd
import csv
from datetime import datetime

import random

PATH_TRAINING = 'training/'

def process():
    """
    NaN value is kept in the list
    """

    file_list = os.listdir(PATH_TRAINING)
    seqs = []
    labels = []
    for file in sorted(file_list):
        records = []
        df = pd.read_csv(os.path.join(PATH_TRAINING, file), sep='|')
        y_df = df.iloc[:, -1]
        labels.append(y_df.tolist())
        df.drop(columns=['SepsisLabel'], inplace=True)
        for _, row in df.sort_values('ICULOS').iterrows():
            statistics = []
            for val in row:
                statistics.append(val)
            records.append(statistics)
        seqs.append(records)
    return seqs, labels

def getSepsisLabel(row, onset_time):
    sepsis_label = 0
    current_time = datetime.strptime(row['record_time'], '%Y-%m-%d %H:%M:%S')
    target_id = str(int(row['icustay_id']))
    if target_id in onset_time:
        onset = datetime.strptime(onset_time[target_id], '%Y-%m-%d %H:%M:%S')
        # print(row['record_time'], onset_time[int(row['icustay_id'])])
        if current_time > onset:
            sepsis_label = 1
    return sepsis_label

def cleanVitalData():
    raw_df = pd.read_csv('pivoted_vital.csv')
    nanid_df = raw_df.loc[raw_df['icustay_id'].isna()]
    vital_df = raw_df.dropna(subset=['icustay_id']).sort_values(['icustay_id', 'record_time'])
    filled_vital = vital_df.groupby(by='icustay_id').ffill().groupby(by='icustay_id').bfill()
    filled_vital.dropna(inplace=True)

    # onset_time = getOnsetTime()
    # filled_vital['sepsis_label'] = filled_vital.apply(getSepsisLabel, args=(onset_time,), axis=1)
    # filled_vital.to_csv('cleaned_pivoted_vital.csv', index=False)
    return filled_vital

def concatPersonalInfo(cleaned_pivoted_vital):
    person_raw_df = pd.read_csv('sepsis_cohort.csv')
    # personal_info_df = person_raw_df[['icustay_id', 'age', 'gender']]
    personal_info_df = person_raw_df.loc[:, ['icustay_id', 'age', 'gender']]
    gender_dict = {'M': 0, 'F': 1}
    personal_info_df.replace({'gender': gender_dict}, inplace=True)

    cleaned_pivoted_vital = cleaned_pivoted_vital.astype({'icustay_id': int})
    final_df = cleaned_pivoted_vital.merge(personal_info_df, how='left', on='icustay_id')

    onset_time = getOnsetTime()
    final_df['sepsis_label'] = final_df.apply(getSepsisLabel, args=(onset_time,), axis=1)
    # print(final_df)

    ####### sample_process
    train_id, valid_id, test_id = getSample()
    
    train_df = final_df.loc[final_df['icustay_id'].isin(train_id)]
    train_df.to_csv('train_sample_cleaned_pivoted_vital.csv', index=False)

    valid_df = final_df.loc[final_df['icustay_id'].isin(valid_id)]
    valid_df.to_csv('valid_sample_cleaned_pivoted_vital.csv', index=False)

    test_df = final_df.loc[final_df['icustay_id'].isin(test_id)]
    test_df.to_csv('test_sample_cleaned_pivoted_vital.csv', index=False)

def getOnsetTime():
    onset_time = {}
    with open('sepsis_onset_time.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            onset_time[row['icustay_id']] = row['onset_time']
    return onset_time

def getSample():
    vital_df = pd.read_csv('pivoted_vital.csv')
    icustay_id_collection = vital_df.icustay_id.dropna().astype(int).unique()
    icustay_id_set = set(icustay_id_collection)
    # print(icustay_id_set)
    sepsis_onset = pd.read_csv('sepsis_onset_time.csv')
    icustay_id_sepsis_set = set(sepsis_onset.icustay_id.astype(int).unique())
    # print(icustay_id_sepsis_set)
    icustay_id_non_sepsis_set = icustay_id_set - icustay_id_sepsis_set
    sepsis_id = list(random.sample(icustay_id_sepsis_set, 500))
    non_sepsis_id = list(random.sample(icustay_id_non_sepsis_set, 5000))
    return (sepsis_id[:400]+non_sepsis_id[:4000], sepsis_id[400:450]+non_sepsis_id[4000:4500], sepsis_id[450:]+non_sepsis_id[4500:])

if __name__ == '__main__':
    cleaned_pivoted_vital = cleanVitalData()
    concatPersonalInfo(cleaned_pivoted_vital)
    # train_id, valid_id, test_id = getSample()