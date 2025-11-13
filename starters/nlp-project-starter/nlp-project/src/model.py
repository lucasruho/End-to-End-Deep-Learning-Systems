import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size:int, emb_dim:int, hidden_dim:int, num_layers:int, bidirectional:bool, dropout:float, num_classes:int, pad_idx:int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        mult = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_dim*mult, num_classes)

    def forward(self, toks, lengths):
        emb = self.embedding(toks)                                # B x T x E
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, (h_n, c_n) = self.lstm(packed)
        # Take last hidden of the top layer
        if self.lstm.bidirectional:
            h = torch.cat((h_n[-2], h_n[-1]), dim=1)             # B x 2H
        else:
            h = h_n[-1]                                          # B x H
        h = self.dropout(h)
        logits = self.fc(h)                                      # B x C
        return logits
