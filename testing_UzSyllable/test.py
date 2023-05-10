from UzSyllable import syllables
import csv

with open('test.csv', mode='r', encoding='utf8') as file:
    rows = csv.reader(file)
    count = 0
    count_true = 0
    for row in rows:
        try:
            count += 1
            if syllables(row[0])[0] == row[1]:
                count_true += 1
        except:
            continue
    print(count_true/count, count) # 0.9442929776729792 13571
