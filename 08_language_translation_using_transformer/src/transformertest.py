import torch

from src.model.transformer import build_transformer

if __name__ == '__main__':

    d_model = 512
    src_vocab_size = 1000
    tgt_vocab_size = 1000
    src_seq_len = 100
    tgt_seq_len = 100
    dropout = 0.1
    num_layers = 6
    num_heads = 8
    d_ff = 2048

    transformer = build_transformer(d_model, src_vocab_size, tgt_vocab_size, src_seq_len, tgt_seq_len, dropout,
                                    num_layers, num_heads, d_ff)

    batch_size = 2
    seq_length = 100

    src_mask = None
    tgt_mask = None

    sample_input = torch.rand(batch_size, seq_length, d_model)

    print(sample_input)

    encoder_output = transformer.encode(sample_input, src_mask)
    print("Encoder output shape: ", encoder_output.shape)


