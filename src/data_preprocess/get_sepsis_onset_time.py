import os
import csv
import glob
import pandas as pd

def read_csv():
    PATH = 'mimic/sofa_timeline'

    df = pd.concat([pd.read_csv(f, header=None, names=['icustay_id', 'hr', 'starttime', 'sofa']) for f in glob.glob(PATH + '/part-0000[0-9]')])
    sorted_df = df.sort_values(['icustay_id', 'hr'], ascending=True)
    prev_id = sorted_df.iloc[0]['icustay_id']
    prev_sofa = sorted_df.iloc[0]['sofa']
    onset_time = {}
    for _, row in sorted_df.iterrows():
        if row['icustay_id'] != prev_id:
            prev_id = row['icustay_id']    
        elif row['icustay_id'] in onset_time:
            continue
        else:
            if row['sofa'] >= prev_sofa + 2:
                onset_time[row['icustay_id']] = row['starttime']
            
        prev_sofa = row['sofa']
    return onset_time

def write_csv(sepsis_onset_time):
    FILENAME = 'sepsis_onset_time.csv'
    
    with open(FILENAME, 'w') as csv_file:
        fieldnames = ['icustay_id', 'onset_time']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for icustay_id, onset_time in sepsis_onset_time.items():
            writer.writerow({'icustay_id': icustay_id, 'onset_time': onset_time})

if __name__ == '__main__':
    onset_time = read_csv()
    # print(onset_time)
    write_csv(onset_time)