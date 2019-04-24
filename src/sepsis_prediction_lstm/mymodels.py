import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class MyLSTM(nn.Module):
    def __init__(self, dim_input):
        super(MyLSTM, self).__init__()
        self.embedding = nn.Linear(in_features=dim_input, out_features=8)
        nn.init.xavier_normal_(self.embedding.weight)
        self.rnn = nn.LSTM(input_size=8, hidden_size=4, num_layers=2, batch_first=True, dropout=0.5)
        self.output = nn.Linear(in_features=4, out_features=2)
        nn.init.xavier_normal_(self.output.weight)
    
    def forward(self, input_tuple):
        seqs, lengths = input_tuple

        embedded = torch.tanh((self.embedding(seqs)))

        seqs_packed = pack_padded_sequence(embedded, lengths, batch_first=True)

        seqs, _ = self.rnn(seqs_packed)

        unpacked_output, _ = pad_packed_sequence(seqs, batch_first=True)

        idx = (torch.LongTensor(lengths) - 1).view(-1, 1).expand(len(lengths), unpacked_output.size(2))
        idx = idx.unsqueeze(1)

        last_output = unpacked_output.gather(1, idx).squeeze(1)

        output = self.output(last_output)

        return output
