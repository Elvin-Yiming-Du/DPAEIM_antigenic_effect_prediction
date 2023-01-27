import torch.nn as nn
import torch.nn.functional as F
import torch
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
