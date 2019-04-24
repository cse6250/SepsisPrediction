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

print(Counter(sepsis))
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

print(sepsis_collect)

bins = {
    '0-5': range(0, 5),
    '5-10': range(5, 10),
    '10-20': range(10, 20),
    '20-40': range(20, 40),
    '40-80': range(40, 80),
    '80-150': range(80, 150),
    '150-500': range(150, 500),
    '500+': range(500, 4000)
}

collect = defaultdict(int)

for count in nonsepsis:
    for key, val in bins.items():
        if count in val:
            collect[key] += 1
            break

print(collect)