import pandas as pd
from collections import Counter, defaultdict

df = pd.read_csv('data_distribution.csv')

sepsis = []
nonsepsis = []

for _, row in df.iterrows():
    if row['sepsis'] is True:
        sepsis.append(int(row['count']))
    else:
        nonsepsis.append(int(row['count']))

sepsis_bins = {
    '1': range(0, 2),
    '2': range(2, 3),
    '3': range(3, 4),
    '4': range(4, 5),
    '5': range(5, 6),
    '6': range(6, 7),
    '7': range(7, 8),
    '8': range(8, 9),
    '9': range(9, 10),
    '10-14': range(10, 15),
    '15-19': range(15, 20),
    '20-24': range(20, 25),
    '25-29': range(25, 30),
    '30+': range(30, 50)
}

sepsis_collect = defaultdict(int)
for count in sepsis:
    for key, val in sepsis_bins.items():
        if count in val:
            sepsis_collect[key] += 1
            break

bins = {
    '0-4': range(0, 5),
    '5-9': range(5, 10),
    '10-14': range(10, 15),
    '15-19': range(15, 20),
    '20-24': range(20, 40),
    '25-29': range(25, 30),
    '30-34': range(30, 35),
    '35-39': range(35, 40),
    '40-44': range(40, 45),
    '45-49': range(45, 50),
    '50-54': range(50, 55),
    '55-59': range(55, 60),
    '60-64': range(60, 65),
    '65-69': range(65, 70),
    '70-79': range(70, 80),
    '80-99': range(80, 100),
    '100-199': range(100, 200),
    '200-499': range(200, 500),
    '500+': range(500, 4000)
}

collect = defaultdict(int)

for count in nonsepsis:
    for key, val in bins.items():
        if count in val:
            collect[key] += 1
            break

print(collect)