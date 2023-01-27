import torch.nn as nn
import torch.nn.functional as F
import torch

from BiLSTM import BiLSTM

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, embedding_matrix, dropout):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        # Create the embedding layer as usual
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding = nn.Embedding(vocab_size, vocab_size , padding_idx=0)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(embedding_dim, self.hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(self.hidden_size * 4, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.long()
        x = x.to(device)
        h_embedding = self.embedding(x)
        h_embedding = torch.squeeze(torch.unsqueeze(h_embedding, 0))

        h_lstm, _ = self.lstm(h_embedding)
        avg_pool = torch.mean(h_lstm, 1)
        max_pool, _ = torch.max(h_lstm, 1)
        # print("avg_pool", avg_pool.size())
        # print("max_pool", max_pool.size())
        conc = torch.cat((avg_pool, max_pool), 1)
        conc = self.relu(self.linear(conc))
        out = self.dropout(conc)
        return out
class BiLSTM_NLI(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, dropout, embedding_matrix, output_dim):
        super(BiLSTM_NLI, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.bilstm_net1 = BiLSTM(vocab_size, embedding_dim, hidden_size, embedding_matrix, dropout)
        self.bilstm_net2 = BiLSTM(vocab_size, embedding_dim, hidden_size, embedding_matrix, dropout)
	    #self.bilstm_net1.to(device)
        #self.bilstm_net2.to(device)
        self.fc1 = nn.Linear(4*hidden_size, hidden_size)

        # activation
        self.relu1 = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        # output layer
        self.fc2 = nn.Linear(hidden_size, output_dim)


    def forward(self, x1, x2):
        # x -> (batch size, max_sent_length)
        x1 = x1.long()
        x1.to(device)
        out1 = self.bilstm_net1.forward(x1)

        x2 = x2.long()
        x2.to(device)
        out2 = self.bilstm_net2.forward(x2)
        x = torch.cat([out1, out2, out1*out2, (out1-out2)], 1)
        # x = torch.Tensor([out1.detach().numpy() , out2.detach().numpy() ,
        #                                 out1*out2.detach().numpy() , (out1-out2).detach().numpy()])

        out = self.fc1(x)
        out = self.relu1(out)
        # out = self.sigmoid(out)
        out = self.fc2(out)
        return out
