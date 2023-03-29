from UzSyllable import UzSyllable
import csv
count = 0
count_true = 0
with open('temp.csv', mode='r') as file:
    reader = csv.reader(file)
    for row in reader:
        for row in reader:
            word = row[0]
            line_break = row[1]
            count += 1
            # answer = ''
            # if not word.__contains__(' ') and not word.__contains__('-'):
            #     answer = ' '.join(UzSyllable.line_break(word))
            #     # 0.7530494677520351
            answer = ' '.join(UzSyllable.test_lb(word)) #0.9961803381340012
            check = False
            if line_break == answer:
                check = True
                count_true += 1
            with open('result_t.csv', mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([word, line_break, answer, check])
    print(count_true / count)



# import csv
# with open('dataset.csv', mode='r') as file:
#     reader = csv.reader(file)
#     for row in reader:
#         word = row[0]
#         syllable = row[1]
#         tokens = syllable.split('-')
#         begin = end = ''
#         if word.__contains__('-'):
#             for i in range(0, len(word)):
#                 if word[i] == '-':
#                     begin = word[i-1]
#                     end = word[i+1]
#         lines = list()
#         if not syllable.__contains__('-'):
#             lines.append(syllable)
#         for j in range(1, len(tokens)):
#             if len(tokens[j - 1]) > 1 and len(tokens[j]) > 1:
#                 w = ''
#                 for i in range(0, j):
#                     w += tokens[i]
#                 w += '-'
#                 for i in range(j, len(tokens)):
#                     w += tokens[i]
#                 if begin != end != '':
#                     for i in range(0, len(w)-1):
#                         if w[i] == begin and w[i+1] == end:
#                             w = w[0:i+1]+'-'+w[i+1:len(w)]
#                             break
#                 lines.append(w)
#             else:
#                 lines.append(tokens[j])
#
#         line_break = ' '.join(lines)
#         with open('temp.csv', mode='a', newline='') as file:
#             writer = csv.writer(file)
#             writer.writerow([word, line_break])