import torch.nn as nn
from torch.autograd import Variable

from models.simple import SimpleNet
import torch

HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.25

class RNNModel(SimpleNet):
    """Container module with an encoder, a recurrent module, and a decoder."""
    import torch.nn as nn

    def __init__(self, bert):

        super().__init__()

        self.bert = bert

        embedding_dim = bert.config.to_dict()['hidden_size']

        self.rnn = nn.GRU(embedding_dim,
                          HIDDEN_DIM,
                          num_layers=N_LAYERS,
                          bidirectional=BIDIRECTIONAL,
                          batch_first=True,
                          dropout=0 if N_LAYERS < 2 else DROPOUT)

        self.out = nn.Linear(HIDDEN_DIM * 2 if BIDIRECTIONAL else HIDDEN_DIM, OUTPUT_DIM)

        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, text):

        # text = [batch size, sent len]

        with torch.no_grad():
            embedded = self.bert(text)[0]

        # embedded = [batch size, sent len, emb dim]

        _, hidden = self.rnn(embedded)

        # hidden = [n layers * n directions, batch size, emb dim]

        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        else:
            hidden = self.dropout(hidden[-1, :, :])

        # hidden = [batch size, hid dim]

        output = self.out(hidden)

        # output = [batch size, out dim]

        return output.squeeze(1), output