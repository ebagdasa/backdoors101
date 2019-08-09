import torch.nn as nn
from torch.autograd import Variable
import torch
from models.simple import SimpleNet


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken+1, ninp, padding_idx=ntoken)
        # self.encoder.requires_grad = False
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout, bidirectional=True)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.fc = nn.Linear(2*nhid, 1)

        self.init_weights()
        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers


    def init_weights(self):
        initrange = 0.1
        # self.encoder.weight.data.uniform_(-initrange, initrange)
        # embeddings = torch.load('data/aag/embeddings_10k.pt')
        # self.encoder.from_pretrained(embeddings)
        # print(torch.mean(embeddings).item())
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-initrange, initrange)


    def forward(self, input):
        emb = self.drop(self.encoder(input))
        output, (hidden, _) = self.rnn(emb)
        output = self.drop(output)
        hidden = self.drop(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        output = self.fc(hidden).squeeze(0)

        return output

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())

#
# class ShallowModel(nn.Module):
#     """Container module with an encoder, a recurrent module, and a decoder."""
#
#     def __init__(self, ntoken, nout, dropout=0.5, tie_weights=False):
#         super(ShallowModel, self).__init__()
#         self.drop = nn.Dropout(dropout)
#         self.encoder = nn.Embedding(ntoken, 200)
#         self.fc = nn.Linear(200, nout)
#
#         self.init_weights()
#
#     def init_weights(self):
#         initrange = 0.1
#         self.encoder.weight.data.uniform_(-initrange, initrange)
#
#     def forward(self, input):
#         emb = self.drop(self.encoder(input))
#         output = self.fc(emb)
#         return output
#     #
#     # def init_hidden(self, bsz):
#     #     weight = next(self.parameters()).data
#     #     if self.rnn_type == 'LSTM':
#     #         return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
#     #                 Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
#     #     else:
    #         return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())