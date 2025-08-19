import collections

import numpy as np

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from nltk.tokenize import word_tokenize

from src.utils.constants import PADDING_ID, UNKNOWN_ID, START_OF_SENTENCE_ID, END_OF_SENTENCE_ID
from src.utils.constants import PADDING_VALUE, UNKNOWN_VALUE, START_OF_SENTENCE_VALUE, END_OF_SENTENCE_VALUE


class TranslationDataset(Dataset):

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
def load_and_preprocess_data(translation_file):
    df = pd.read_csv(translation_file)
    val_df = df.sample(frac=0.05, random_state=41)

    return df, val_df


# Every sentence in the batch is confined to same sequence length
def pad_sequence(batch):
    """ Input
    x = [
            torch.tensor[7, 167, 854, 45, 12, 78, 67, 4567, 2345],
            torch.tensor[56, 678, 435, 9087, 123],
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
    src_texts = [src_text for src_text, tgt_text in batch]
    tgt_texts = [tgt_text for src_text, tgt_text in batch]
    padded_src_texts = torch.nn.utils.rnn.pad_sequence(src_texts, batch_first=True, padding_value=PADDING_VALUE)
    padded_tgt_texts = torch.nn.utils.rnn.pad_sequence(tgt_texts, batch_first=True, padding_value=PADDING_VALUE)
    return padded_src_texts, padded_tgt_texts


def create_vocab_dictionary(df_list_of_tokenized_words, vocab_size):
    # Build a vocabulary of most common words in the training data (Max no. of words = vocab_size - 2)
    # Returns counter from all the sentences ({'toward': 6, 'claw': 2, 'aerospace': 1})
    counter = collections.Counter()
    for text in df_list_of_tokenized_words:
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
        {'PAD': 0, 'UNK': 1, 'SOS': 2, 'EOS: 3', 'toward': 4, 'claw': 5, 'aerospace': 5, ___}
    """
    word_to_id = {vocab[i]: i + 4 for i in range(len(vocab))}
    word_to_id[PADDING_ID] = PADDING_VALUE
    word_to_id[UNKNOWN_ID] = UNKNOWN_VALUE
    word_to_id[START_OF_SENTENCE_ID] = START_OF_SENTENCE_VALUE
    word_to_id[END_OF_SENTENCE_ID] = END_OF_SENTENCE_VALUE

    return word_to_id


# 1. Load and clean the data
# 2. Tokenize sentence into list of words
# 3. Build the vocabulary, assign number to a word
# 4. Create training & testing input tensors
# 5. Create training & testing - datasets and dataloaders
def create_dataloaders(translation_file, batch_size, src_seq_len, tgt_seq_len,
                       src_vocab_size, tgt_vocab_size):
    train_df, val_df = load_and_preprocess_data(translation_file)

    # Tokenize the sentences using word tokenizer, and create word_to_id dictionary (source and target)
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
    x_train_texts = [word_tokenize(row['en'].lower())[0: src_seq_len-2] for index, row in train_df.iterrows()]
    x_val_texts = [word_tokenize(row['en'].lower())[0: src_seq_len-2] for index, row in val_df.iterrows()]

    src_word_to_id = create_vocab_dictionary(x_train_texts + x_val_texts, src_vocab_size)

    y_train_texts = [word_tokenize(row['fr'].lower())[0: tgt_seq_len-2] for index, row in train_df.iterrows()]
    y_val_texts = [word_tokenize(row['fr'].lower())[0: tgt_seq_len-2] for index, row in val_df.iterrows()]

    tgt_word_to_id = create_vocab_dictionary(y_train_texts + y_val_texts, tgt_vocab_size)

    # Create training/validation input tensors using above word_to_id dictionaries
    """
    [
        torch.tensor[2, 7, 167, 854, 45, 12, 78, 67, 4567, 2345, 3],
        torch.tensor[2, 56, 678, 435, 9087, 123, 3],
        ___      
    ]
    """
    x_train = [
        torch.tensor(
            [START_OF_SENTENCE_VALUE] +
            [src_word_to_id.get(word, src_word_to_id.get(UNKNOWN_ID)) for word in text] +
            [END_OF_SENTENCE_VALUE]
        )
        for text in x_train_texts
    ]
    x_val = [
        torch.tensor(
            [START_OF_SENTENCE_VALUE] +
            [src_word_to_id.get(word, src_word_to_id.get(UNKNOWN_ID)) for word in text] +
            [END_OF_SENTENCE_VALUE]
        )
        for text in x_val_texts
    ]

    y_train = [
        torch.tensor(
            [START_OF_SENTENCE_VALUE] +
            [tgt_word_to_id.get(word, tgt_word_to_id.get(UNKNOWN_ID)) for word in text] +
            [END_OF_SENTENCE_VALUE]
        )
        for text in y_train_texts
    ]
    y_val = [
        torch.tensor(
            [START_OF_SENTENCE_VALUE] +
            [tgt_word_to_id.get(word, tgt_word_to_id.get(UNKNOWN_ID)) for word in text] +
            [END_OF_SENTENCE_VALUE]
        )
        for text in y_val_texts
    ]

    # Create datasets using input and output tensors
    train_dataset = TranslationDataset(x_train, y_train)
    val_dataset = TranslationDataset(x_val, y_val)

    """ Dataloader outputs batch like this
    tensor([[2, 1556, 5, 744, ..., 0, 0, 0],
            [2, 1, 7498, 67, ..., 0, 0, 0],
            [2, 3859, 13, 14, ..., 0, 0, 0],
            ...,
            [2, 43, 433, 7, ..., 0, 0, 0],
            [2, 3299, 5, 27, ..., 0, 0, 0],
            [2, 10825, 4076, 2298, ..., 0, 0, 0]], device='mps:0')
    tensor([[2, 67, 5, 4342, ..., 0, 0, 0],
            [2, 1, 213, 23434, ..., 0, 0, 0],
            [2, 432, 13, 23, ..., 0, 0, 0],
            ...,
            [2, 25, 762, 98, ..., 0, 0, 0],
            [2, 3299, 323, 2323, ..., 0, 0, 0],
            [2, 12, 45, 7576, ..., 0, 0, 0]], device='mps:0')
    """
    # Create dataloaders using above datasets
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_sequence)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_sequence)

    return train_dataloader, val_dataloader, src_word_to_id, tgt_word_to_id
