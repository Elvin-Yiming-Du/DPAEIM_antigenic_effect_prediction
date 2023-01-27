import torch.nn as nn
import torch.nn.functional as F
import torch

from CNN_Model import CNN
from FFNN import FFNN
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, out_channels, window_size, output_dim, dropout):
        super(CNN, self).__init__()

        # Create the embedding layer as usual
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # in_channels -- 1 text channel
        # out_channels -- the number of output channels
        # kernel_size is (window size x embedding dim)
        self.conv = nn.Conv2d(
            in_channels=1, out_channels=out_channels,
            kernel_size=(window_size, embedding_dim))

        # the dropout layer
        self.dropout = nn.Dropout(dropout)

        # the output layer
        self.fc = nn.Linear(out_channels, output_dim)

    def forward(self, x):
        # x -> (batch size, max_sent_length)
        x = x.long()
        embedded = self.embedding(x.to(device))
        # embedded -> (batch size, max_sent_length, embedding_dim)

        # images have 3 RGB channels
        # for the text we add 1 channel
        embedded = embedded.unsqueeze(1)
        # embedded -> (batch size, 1, max_sent_length, embedding dim)

        # Compute the feature maps
        feature_maps = self.conv(embedded)

        ##########################################
        # Q: What is the shape of `feature_maps` ?
        ##########################################
        # A: (batch size, n filters, max_sent_length - window size + 1, 1)

        feature_maps = feature_maps.squeeze(3)

        # Q: why do we remove 1 dimension here?
        # A: we do need the 1 channel anymore
        # Apply ReLU
        feature_maps = F.relu(feature_maps)

        # Apply the max pooling layer
        pooled = F.max_pool1d(feature_maps, feature_maps.shape[2])
        pooled = pooled.squeeze(2)

        ####################################
        # Q: What is the shape of `pooled` ?
        ####################################
        # A: (batch size, n_filters)

        dropped = self.dropout(pooled)
        preds = self.fc(dropped)
        return preds

class CNN_NLI(nn.Module):
    def __init__(self, vocab_size, embedding_dim, out_channels, window_size, output_dim, dropout, embedding_matrix, HIDDEN_DIM):
        super(CNN_NLI, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.cnn_net1 = CNN(vocab_size, embedding_dim, out_channels, window_size, 1, dropout)
        self.cnn_net2 = CNN(vocab_size, embedding_dim, out_channels, window_size, 1, dropout)
        self.cnn_net1.embedding.weight.data.copy_(embedding_matrix)
        self.cnn_net2.embedding.weight.data.copy_(embedding_matrix)
        self.fc1 = nn.Linear(4, HIDDEN_DIM)

        # activation
        self.relu1 = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        # output layer
        self.fc2 = nn.Linear(HIDDEN_DIM, output_dim)


    def forward(self, x1, x2):
        # x -> (batch size, max_sent_length)
        x1 = x1.long()
        out1 = self.cnn_net1.forward(x1)

        x2 = x2.long()
        out2 = self.cnn_net2.forward(x2)
        x = torch.cat([out1, out2, out1*out2, (out1-out2)], 1)
        # x = torch.Tensor([out1.detach().numpy() , out2.detach().numpy() ,
        #                                 out1*out2.detach().numpy() , (out1-out2).detach().numpy()])

        out = self.fc1(x)
        out = self.relu1(out)
        # out = self.sigmoid(out)
        out = self.fc2(out)
        return out
