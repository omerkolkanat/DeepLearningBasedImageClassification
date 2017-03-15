import csv
from collections import defaultdict

columns = defaultdict(list) # each value in each column is appended to a list

with open('result.csv') as f:
    reader = csv.DictReader(f) # read rows into a dictionary format
    for row in reader: # read a row as {column1: value1, column2: value2,...}
        for (k,v) in row.items(): # go over each column name and value
            columns[k].append(v) # append the value into the appropriate list
                                 # based on column name k

# print(columns['label'])
inc = 0
notInc = 0
netinc = 0
netNotInc = 0



for x in range(0, len(columns['label'])):
    if((float(columns['label'][x]) >= 0) and (float(columns['label'][x])<= 0.2)):
            inc = inc + 1
    if((float(columns['label'][x]) >= 0.8) and (float(columns['label'][x])<= 1.0)):
            notInc = notInc + 1
print(inc)
print(notInc)