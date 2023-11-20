import pandas as pd

# Replace 'input_file.csv' with the path to your input CSV file
input_file_path = 'newsyllable_w.csv'

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(input_file_path, encoding="utf8")

# Define lists of vowels and consonants
vowels = ['a', 'u', 'i', 'o', 'e', 'ō']
consonants = ['b', 'v', 'g', 'd', 'j', 'y', 'k', 'l', 'm', 'n', 'p', 'r', 's', 't', 'f', 'x', 'w', 'q', 'ḡ', 'h', 'z']

# Function to replace vowels and consonants
def replace_letters(word):
    modified_word = ''
    for letter in word:
        if letter.lower() in vowels:
            modified_word += 'V'
        elif letter.lower() in consonants:
            modified_word += 'C'
        else:
            if letter!="-":
                print(letter, word)
                return ""
            modified_word += letter
    return modified_word

# Apply the replacement function to each element in the DataFrame
df_replaced = df.applymap(replace_letters)

# Replace 'output_file_modified.csv' with the desired path for the output CSV file
output_file_path = 'newsyllableCV.csv'

# Save the modified DataFrame to another CSV file
df_replaced.to_csv(output_file_path, index=False)

print(f"Modified DataFrame saved to {output_file_path}")

'''
1) a'lam dagi barcha yumshatish belgisini uchiramiz, chunki bu xar doim oldindagi harf bn qoladi
2) ḡ, ō larni bitta harfli qilib olamiz, chunki apostrof xarfni uzi bn yuradi
3) ng, ch, sh diagraf harflarni bitta belgi bilan almashtiramiz, chunki bular ajralmasligi kerak bugunda xam, lex.uzdagi imlo qoidasiga kura


'''
