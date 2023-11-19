import os
import string

# Set the root folder where your text documents are located
root_folder = 'C:/Users/E-MaxPCShop/Desktop/Uzbek_News_Dataset_small'

# Set the path for the output text file
output_file_path = 'getallwords.txt'

# Define a set to store unique tokens
unique_tokens = set()

# Define a set of punctuation to remove
punctuation_to_remove = set([',', '.', '!', '?', ']', '[', '}', '{', '“', '”','"'])

# Function to tokenize and process the text
def process_text(text):
    # Tokenize the text by white space
    tokens = text.split()

    # Remove specified punctuation and convert to lowercase
    tokens = [token.lower().strip(''.join(punctuation_to_remove)) for token in tokens]

    return tokens

# Traverse through the folders and files
for foldername, subfolders, filenames in os.walk(root_folder):
    print(foldername, subfolders, len(filenames))
    for filename in filenames:
        file_path = os.path.join(foldername, filename)

        # Read the text from the file
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()

        # Tokenize and process the text
        tokens = process_text(text)

        # Update the set of unique tokens
        unique_tokens.update(tokens)

# Save the current tokens to the output text file immediately after reading each document
with open(output_file_path, 'a', encoding='utf-8') as output_file:
    output_file.write('\n'.join(unique_tokens) + '\n')

print(f"Unique tokens saved to {output_file_path}")