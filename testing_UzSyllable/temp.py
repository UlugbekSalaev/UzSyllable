import csv
with open('result.csv', mode='r') as file:
    reader = csv.reader(file)
    for row in reader:
        if row[3] == 'False':
            with open('errors.csv', mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(row)