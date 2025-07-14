import collections
import numpy as np

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from nltk.tokenize import word_tokenize

PAD = 0
UNK = 1


class AGNewsDataset(Dataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        return x, y


# Load and preprocess data
def load_and_preprocess_data(train_file, val_file):
    # Read data from train and val file
    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)

    # Combine Title and Description columns
    train_df['Text'] = train_df['Title'] + ' ' + train_df['Description']
    val_df['Text'] = val_df['Title'] + ' ' + val_df['Description']

    train_df = train_df.drop(['Title', 'Description'], axis=1)
    val_df = val_df.drop(['Title', 'Description'], axis=1)

    # Remove \ and "" from Text
    train_df['Text'] = train_df['Text'].str.replace('\\', '').str.replace('"', '')
    val_df['Text'] = val_df['Text'].str.replace('\\', '').str.replace('"', '')

    # Labels currently in (1, 4) range -> Transform the labels to be in (0, 3) range
    train_df['Class Index'] = train_df['Class Index'] - 1
    val_df['Class Index'] = val_df['Class Index'] - 1

    return train_df, val_df


# Every sentence in the batch is confined to same sequence length
def pad_sequence(batch):
    """ Input
    x = torch.tensor[
            [7, 167, 854, 45, 12, 78, 67, 4567, 2345],
            [56, 678, 435, 9087, 123],
            ___
        ],
    y = torch.LongTensor [2, 3, ___]
    """
    """ Output padded_text
    torch.tensor[
        [7, 167, 854, 45, 12, 78, 67, 4567, 2345],
        [56, 678, 435, 9087, 123, 0, 0, 0, 0],
        ___
    ]
    """
    """ Output labels = torch.LongTensor [2, 3, ___]
    """
    texts = [text for text, label in batch]
    labels = torch.LongTensor([label for text, label in batch])
    padded_text = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=PAD)
    return padded_text, labels


# 1. Load and clean the training and validation data,
# 2. Tokenize sentence into list of words
# 3. Build the vocabulary, assign number to a word
# 4. Create training & testing input tensors
# 5. Create training & testing - datasets and dataloaders
def create_dataloaders(train_file, val_file, batch_size, seq_len, vocab_size):
    train_df, val_df = load_and_preprocess_data(train_file, val_file)

    y_train = torch.LongTensor(train_df['Class Index'].values)
    y_val = torch.LongTensor(val_df['Class Index'].values)

    # Tokenize the sentences using word tokenizer
    """ Input
    [
        ['Wall St. Bears Claw Back Into the Black'],
        ['Carlyle Looks Toward Commercial Aerospace'],
        ___
    ]
    """
    """ Output
    [
        ['wall', 'st', '.', 'bears', 'claw', 'back', 'into', 'the', 'black'],
        ['carlyle', 'looks','toward', 'commercial', 'aerospace'],
        ___
    ]
    """
    x_train_texts = [word_tokenize(row['Text'].lower())[0: seq_len] for index, row in train_df.iterrows()]
    x_val_texts = [word_tokenize(row['Text'].lower())[0: seq_len] for index, row in val_df.iterrows()]

    # Build a vocabulary of most common words in the training data (Max no. of words = vocab_size - 2)
    # Returns counter from all the sentences ({'toward': 6, 'claw': 2, 'aerospace': 1})
    counter = collections.Counter()
    for text in x_train_texts:
        counter.update(text)

    # Returns Counter array (Max no. of words = vocab_size - 2)
    """
    [
        ['toward', 6],
        ['claw', 2],
        ['aerospace', 1],
        ___
    ]
    """
    most_common_words = np.array(counter.most_common(vocab_size - 2))
    # Selects only words from the array
    vocab = most_common_words[:, 0]

    # Create an dictionary {word --> id}
    """
        {'toward': 2, 'claw': 3, 'aerospace': 4, ___}
    """
    word_to_id = {vocab[i]: i + 2 for i in range(len(vocab))}

    # Create training/testing input tensors using above word_to_id dictionary
    """
    torch.tensor[
        [7, 167, 854, 45, 12, 78, 67, 4567, 2345],
        [56, 678, 435, 9087, 123],
        ___      
    ]
    """
    x_train = [torch.tensor([word_to_id.get(word, UNK) for word in text])
               for text in x_train_texts]
    x_val = [torch.tensor([word_to_id.get(word, UNK) for word in text])
             for text in x_val_texts]

    # Create datasets using input and output tensors
    train_dataset = AGNewsDataset(x_train, y_train)
    val_dataset = AGNewsDataset(x_val, y_val)

    """ Dataloader outputs batch like this
    tensor([[1556, 5, 744, ..., 0, 0, 0],
            [1, 7498, 67, ..., 0, 0, 0],
            [3859, 13, 14, ..., 0, 0, 0],
            ...,
            [43, 433, 7, ..., 0, 0, 0],
            [3299, 5, 27, ..., 0, 0, 0],
            [10825, 4076, 2298, ..., 0, 0, 0]], device='mps:0')
    tensor([2, 1, 0, 1, 0, 0, 3, 0, 3, 0, 0, 2, 0, 1, 0, 1, 0, 2, 1, 0, 3, 1, 0, 1,
            3, 1, 0, 1, 2, 2, 3, 0, 2, 2, 0, 0, 3, 2, 0, 0, 0, 1, 3, 3, 2, 0, 2, 0,
            0, 0, 3, 3, 3, 3, 0, 3, 3, 3, 1, 1, 1, 3, 3, 3, 1, 0, 2, 2, 0, 0, 0, 3,
            1, 2, 2, 1, 2, 3, 0, 3, 3, 3, 0, 0, 0, 1, 0, 2, 1, 3, 0, 3, 3, 1, 1, 1,
            2, 3, 0, 2, 1, 2, 1, 0, 1, 2, 3, 2, 3, 0, 1, 1, 1, 3, 0, 3, 0, 1, 2, 2,
            1, 0, 1, 1, 2, 2, 0, 2], device='mps:0')
    """
    # Create dataloaders using above datasets
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_sequence)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_sequence)

    return train_dataloader, val_dataloader, word_to_id
