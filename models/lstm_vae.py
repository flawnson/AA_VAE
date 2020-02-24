import torch.nn as nn

from models.vae_template import VaeTemplate


class LSTMVae(VaeTemplate):
    def __init__(self, h_dim, z_dim, vocab, embedding_size,  device, layers, bidirectional: bool):
        super(LSTMVae, self).__init__(None, None, device, z_dim * 2, z_dim)
        embedding_size = int(embedding_size/2)*2
        self.embeds = nn.Embedding(vocab, embedding_size)
        self.rectifier = nn.ReLU()
        self.encoder_rnn = nn.LSTM(embedding_size, z_dim, num_layers=layers,
                                   bidirectional=bidirectional,
                                   batch_first=True)
        self.decoder_rnn = nn.LSTM(z_dim * (2 if bidirectional else 1), int(embedding_size / (2 if bidirectional else 1)),
                                   num_layers=layers, bidirectional=bidirectional,
                                   batch_first=True)
        self.deembed = nn.Linear(embedding_size, vocab)

        self.smax = nn.Softmax(dim=2)

    def forward(self, x):
        self.encoder_rnn.flatten_parameters()
        x = self.embeds(x)
        h, _ = self.encoder_rnn(x)
        z, mu, log_var = self.bottleneck(h)
        z = self.fc3(z)
        self.decoder_rnn.flatten_parameters()
        val = self.smax(self.deembed(self.decoder_rnn(z)[0]))
        if self.postprocess is not None:
            val = self.postprocess(val)
        return val.transpose(1, 2), mu, log_var
