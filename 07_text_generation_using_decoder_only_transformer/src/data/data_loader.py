import torch
from torch.utils.data import Dataset, DataLoader

from nltk.tokenize import word_tokenize

from src.utils.constants import PADDING, UNKNOWN, START_OF_SENTENCE, END_OF_SENTENCE


# Create a vocabulary from the book text
def create_vocab(text):
    tokens = word_tokenize(text)
    unique_tokens = set(tokens)

    vocab = {token: i+4 for i, token in enumerate(unique_tokens)}
    vocab[PADDING] = 0
    vocab[UNKNOWN] = 1
    vocab[START_OF_SENTENCE] = 2
    vocab[END_OF_SENTENCE] = 3

    return vocab


class TextDataset(Dataset):

    def __init__(self, seq_len, book_text, vocab):
        self.seq_len = seq_len

        # Create an array of tokens for all the words in the book, size = number of tokenized words
        # [89, 7856, 5467, 4245, 89, ___ ]
        self.data = [vocab.get(token, UNKNOWN) for token in word_tokenize(book_text)]

        print('Number of tokenized words: ', len(self.data))

        # [89, 7856, 5467, 4245, 89, ___ ] + ([3] * 20)
        # [89, 7856, 5467, 4245, 89, ___, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
        self.data = self.data + ([vocab[END_OF_SENTENCE]] * self.seq_len)
        print('Number of tokenized words after adding <eos>: ', len(self.data))

    def __len__(self):
        return len(self.data) - self.seq_len + 1

    def __getitem__(self, idx):
        sequence = self.data[idx:idx + self.seq_len]

        # Except last word/token
        input_sequence = torch.tensor(sequence[:-1], dtype=torch.long)
        # Except first word/token
        target_sequence = torch.tensor(sequence[1:], dtype=torch.long)

        return input_sequence, target_sequence


def create_dataloaders(batch_size, seq_len, book_filepath):

    # Create vocab
    with open(book_filepath, 'r', encoding='utf-8') as file:
        book_text = file.read()
    vocab = create_vocab(book_text)

    # Create dataset using the book's text
    train_dataset = TextDataset(seq_len, book_text, vocab)

    # Create dataloaders using above datasets
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, vocab
