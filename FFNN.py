from torch import nn

class FFNN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, embedding_matrix, vocab_size, num_classes):
        super(FFNN, self).__init__()
        # embedding (lookup layer) layer
        # padding_idx argument makes sure that the 0-th token in the vocabulary
        # is used for padding purposes i.e. its embedding will be a 0-vector
        # self.embedding = nn.Embedding.from_pretrained(embedding_matrix, padding_idx=0)
        # self.embedding = nn.Embedding(240, 768)
        # hidden layer
        self.fc1 = nn.Linear(4*embedding_dim, hidden_dim)

        # activation
        self.relu1 = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        # output layer
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x has shape (batch_size, max_sent_len)
        # embedded = self.embedding(x)
        # `embedding` has shape (batch size, max_sent_len, embedding dim)
        ########################################################################
        # Q: Compute the average embeddings of shape (batch_size, embedding_dim)
        ########################################################################
        # Implement averaging that ignores padding (average using actual sentence lengths).
        # How this effect the result?
        # averaged = "<TODO>"
        # averaged = embedded.mean(-1)
        # sent_lens = x.ne(0).sum(1, keepdims=True)
        # averaged = embedded.sum(1) / sent_lens
        out = self.fc1(x)
        out = self.relu1(out)
        # out = self.sigmoid(out)
        out = self.fc2(out)
        return out