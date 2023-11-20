import torch
from transformers import BertTokenizer, BertForTokenClassification
from torch.nn import Softmax

# Sample dataset
word_sequences = ['CVCCVC', 'VCVC', 'CVC', 'CVCCVCCV']
syllable_sequences = ['CVC-CVC', 'V-CVC', 'CVC', 'CVC-CVC-CV']

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Encode input sequences
input_ids = tokenizer(word_sequences, return_tensors='pt', padding=True, truncation=True)['input_ids']

# Load pre-trained BERT model for token classification
model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=len(set(''.join(syllable_sequences))))

# Set model to evaluation mode
model.eval()

# Make predictions
with torch.no_grad():
    outputs = model(input_ids)
    logits = outputs.logits

# Softmax activation
softmax = Softmax(dim=-1)
predictions = softmax(logits)

# Extract predicted labels
predicted_labels = torch.argmax(predictions, dim=-1)

# Convert predicted labels back to syllable sequences
predicted_syllables = [tokenizer.decode(seq, skip_special_tokens=True) for seq in predicted_labels.numpy()]

# Print results
for word, true_syllable, predicted_syllable in zip(word_sequences, syllable_sequences, predicted_syllables):
    print(f"Word: {word}, True Syllables: {true_syllable}, Predicted Syllables: {predicted_syllable}")
