from  UzSyllable import UzSyllable
input_file_path = 'output.txt'
output_file_path = 'newsyllable.txt'
# obj = UzSyllable.syllables("maktab")[0]
with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
    for line_number, line in enumerate(input_file, start=1):
        # Remove newline character from the end of the line
        words = line.strip().split()

        # Write each word to the output file
        for word in words:
            try:
                # Remove newline character from the end of the line
                words = line.strip().split()

                # Write each word to the output file
                for word in words:
                    output_file.write(word + ','+UzSyllable.syllables(word)[0]+'\n')

            except Exception as e:
                print(f"Skipping line {word} due to an error: {e}")



# Inform the user about the output file
print(f"All words have been saved to {output_file_path}")
