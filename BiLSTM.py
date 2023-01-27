import torch.nn as nn
import torch

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
class BiLSTM(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_size, embedding_matrix, dropout, outdim = 1):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        # Create the embedding layer as usual
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding = nn.Embedding(vocab_size, vocab_size , padding_idx=0)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(embedding_dim, self.hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(self.hidden_size * 4, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(64, outdim)

    def forward(self, x):
        x = x.to(device)
        x = x.long()
        h_embedding = self.embedding(x)
        h_embedding = torch.squeeze(torch.unsqueeze(h_embedding, 0))

        h_lstm, _ = self.lstm(h_embedding)
        avg_pool = torch.mean(h_lstm, 1)
        max_pool, _ = torch.max(h_lstm, 1)
        # print("avg_pool", avg_pool.size())
        # print("max_pool", max_pool.size())
        conc = torch.cat((avg_pool, max_pool), 1)
        conc = self.relu(self.linear(conc))
        conc = self.dropout(conc)
        out = self.out(conc)
        return out
