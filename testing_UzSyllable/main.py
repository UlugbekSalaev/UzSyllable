from UzSyllable import UzSyllable
import csv
count = 0
count_true = 0
with open('dataset.csv', mode='r') as file:
    reader = csv.reader(file)
    for row in reader:
        word = row[0]
        syllable = row[1]
        count += 1
        answer = ' '.join(UzSyllable.syllables(word))
        check = False
        if syllable == answer:
            check = True
            count_true += 1
        with open('result.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([word, syllable, answer, check])
print(count_true/count) #0.9961678626441748