import os
import pandas as pd
import csv
from datetime import datetime
import numpy as np
import random

def getSepsisLabel(row, onset_time):
    sepsis_label = 0
    current_time = datetime.strptime(row['record_time'], '%Y-%m-%d %H:%M:%S')
    target_id = str(int(row['icustay_id']))
    if target_id in onset_time:
        onset = datetime.strptime(onset_time[target_id], '%Y-%m-%d %H:%M:%S')
        if current_time == onset:
            sepsis_label = 1
        elif current_time > onset:
            sepsis_label = np.nan
    return sepsis_label

def cleanVitalData():
    raw_df = pd.read_csv('../../pivoted_vital.csv')
    nanid_df = raw_df.loc[raw_df['icustay_id'].isna()]
    vital_df = raw_df.dropna(subset=['icustay_id']).sort_values(['icustay_id', 'record_time'])
    filled_vital = vital_df.groupby(by='icustay_id').ffill().groupby(by='icustay_id').bfill()
    filled_vital.dropna(inplace=True)

    return filled_vital

def concatPersonalInfo(cleaned_pivoted_vital):
    person_raw_df = pd.read_csv('../../sepsis_cohort.csv')
    personal_info_df = person_raw_df.loc[:, ['icustay_id', 'age', 'gender']]
    gender_dict = {'M': 0, 'F': 1}
    personal_info_df.replace({'gender': gender_dict}, inplace=True)

    cleaned_pivoted_vital = cleaned_pivoted_vital.astype({'icustay_id': int})
    final_df = cleaned_pivoted_vital.merge(personal_info_df, how='left', on='icustay_id')

    onset_time = getOnsetTime()
    final_df['sepsis_label'] = final_df.apply(getSepsisLabel, args=(onset_time,), axis=1)
    final_df.dropna(inplace=True)
    final_df['sepsis_label'] = final_df['sepsis_label'].astype(int)

    ########## generate data distribution
    # final_df['count'] = final_df.groupby('icustay_id')['icustay_id'].transform('count')
    # statistics = final_df[['icustay_id', 'count']].drop_duplicates()
    # statistics['sepsis'] = statistics.apply(lambda row: str(int(row['icustay_id'])) in onset_time, axis=1)
    # print(statistics)
    # statistics.to_csv('data_distribution.csv', index=False)

    #### remove data occurrence less than 6
    final_df = final_df.groupby('icustay_id').filter(lambda x: len(x) > 6)
    selected_ids = set(final_df['icustay_id'].tolist())
    selected_onset_ids = set([int(x) for x in onset_time.keys()]).intersection(selected_ids)

    ####### sample_process
    icu_size = len(selected_ids)
    icu_with_sepsis_size = len(selected_onset_ids)
    print(icu_size - icu_with_sepsis_size, icu_with_sepsis_size)
    train_id, valid_id, test_id = getSample(selected_onset_ids, selected_ids-selected_onset_ids, icu_with_sepsis_size, icu_with_sepsis_size)
    ####### end sample

    train_df = final_df.loc[final_df['icustay_id'].isin(train_id)]
    train_df.to_csv('train_cleaned_pivoted_vital.csv', index=False)

    valid_df = final_df.loc[final_df['icustay_id'].isin(valid_id)]
    valid_df.to_csv('valid_cleaned_pivoted_vital.csv', index=False)

    test_df = final_df.loc[final_df['icustay_id'].isin(test_id)]
    test_df.to_csv('test_cleaned_pivoted_vital.csv', index=False)

def getOnsetTime():
    onset_time = {}
    with open('sepsis_onset_time.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            onset_time[row['icustay_id']] = row['onset_time']
    return onset_time

def getSample(icustay_id_sepsis_set, icustay_id_non_sepsis_set, sepsis_icuid_size, nonsepsis_icuid_size):

    sepsis_id = list(random.sample(icustay_id_sepsis_set, sepsis_icuid_size))
    non_sepsis_id = list(random.sample(icustay_id_non_sepsis_set, nonsepsis_icuid_size))

    sepsis_first_cut = int(sepsis_icuid_size*0.7)
    sepsis_second_cut = int(sepsis_icuid_size*0.8)
    nonsepsis_first_cut = int(nonsepsis_icuid_size*0.7)
    nonsepsis_second_cut = int(nonsepsis_icuid_size*0.8)
    print(sepsis_second_cut, nonsepsis_second_cut)

    return (sepsis_id[:sepsis_first_cut]+non_sepsis_id[:nonsepsis_first_cut], sepsis_id[sepsis_first_cut:sepsis_second_cut]+non_sepsis_id[nonsepsis_first_cut:nonsepsis_second_cut], sepsis_id[sepsis_second_cut:]+non_sepsis_id[nonsepsis_second_cut:])

if __name__ == '__main__':
    cleaned_pivoted_vital = cleanVitalData()
    concatPersonalInfo(cleaned_pivoted_vital)
