import UzSyllable
import csv
count = 0
count_syll_true = 0
count_line_true = 0
with open('dataset.csv', mode='r') as file:
    reader = csv.reader(file)
    for row in reader:
        word = row[0]
        syllable = row[1]
        line_break = row[2]
        count += 1
        answer_syll = ' '.join(UzSyllable.syllables(word))
        answer_line = ' '.join(UzSyllable.line_break(word))
        check_syll = False
        check_line = False
        if syllable == answer_syll:
            check_syll = True
            count_syll_true += 1
        if line_break == answer_line:
            check_line = True
            count_line_true += 1
        with open('result.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([word, syllable, answer_syll, check_syll, line_break, answer_line, check_line])
print(count_syll_true/count) #0.996180385968867
print(count_line_true/count) #0.996180385968867


