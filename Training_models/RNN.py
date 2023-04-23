import pandas as pd
import torch
import random
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForTokenClassification, AdamW
from transformers import get_linear_schedule_with_warmup

# Set the random seed for reproducibility
random.seed(42)
torch.manual_seed(42)

# Load the syllabification dataset
df = pd.read_csv('uzbek_syllabification_dataset.csv')

# Split the dataset into training and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Define the BERT tokenizer for Uzbek language
tokenizer = BertTokenizer.from_pretrained('coppercitylabs/uzbert-base-uncased', do_lower_case=False)

# Define the BERT model for token classification
model = BertForTokenClassification.from_pretrained('coppercitylabs/uzbert-base-uncased', num_labels=2)

# Define the optimizer and the learning rate scheduler
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
total_steps = len(train_df) * 5
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)


# Define the training function
def train_model(model, tokenizer, optimizer, scheduler, train_df, val_df, batch_size, num_epochs):
    # Create the dataloaders for the training and validation sets
    train_loader = create_data_loader(train_df, tokenizer, batch_size)
    val_loader = create_data_loader(val_df, tokenizer, batch_size)

    # Set the device to GPU if available, otherwise to CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Set the initial best validation loss to infinity
    best_val_loss = float('inf')

    # Train the model for the specified number of epochs
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # Set the model to training mode
        model.train()

        # Initialize the training loss accumulator
        train_loss = 0.0

        # Train the model on the training set
        for batch in train_loader:
            # Load the inputs and labels to the device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            # Backward pass
            loss.backward()

            # Update the optimizer and the learning rate scheduler
            optimizer.step()
            scheduler.step()

            # Accumulate the training loss
            train_loss += loss.item() * len(input_ids)

        # Compute the average training loss
        train_loss /= len(train_df)

        # Print the training loss
        print(f'Training loss: {train_loss:.4f}')

        # Set the model to evaluation mode
        model.eval()

        # Initialize the validation loss and the number of correct
        # predictions
        val_loss = 0.0
        num_correct = 0

        # Evaluate the model on the validation set
        with torch.no_grad():
            for batch in
