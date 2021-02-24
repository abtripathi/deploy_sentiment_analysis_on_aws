import torch.nn as nn

class LSTMClassifier(nn.Module):
    """
    This is the simple RNN model we will be using to perform Sentiment Analysis.
    """

    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        """
        Initialize the model by settingg up the various layers.
        """
        super(LSTMClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.dense = nn.Linear(in_features=hidden_dim, out_features=1)
        self.sig = nn.Sigmoid()
        
        self.word_dict = None

    def forward(self, x):
        """
        Perform a forward pass of our model on some input.
        """
        #print("start of x",x.shape)
        x = x.t()
        #print("after transpose x",x.shape)
        lengths = x[0,:]
        #print("lengths.shape",lengths.shape)
        reviews = x[1:,:]
        #print("reviews.shape",reviews.shape)
        embeds = self.embedding(reviews)
        #print("embeds.shape",embeds.shape)
        lstm_out, _ = self.lstm(embeds)
        #print("lstm_out.shape",lstm_out.shape)
        out = self.dense(lstm_out)
        #print("out after linear.shape",out.shape)
        #print("out",out) 
        #print(lengths -1)
        #print(range(len(lengths)))
        out = out[lengths - 1, range(len(lengths))]
        #print("out after manipulation",out.shape)
        
        return self.sig(out.squeeze())