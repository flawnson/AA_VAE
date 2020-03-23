from utils.model_common import *


class TransformerModel(nn.Module):
    def __init__(self, model_config, h_dim, z_dim, input_size, device, embeddings_static, requires_grad=True):
        torch.manual_seed(0)
        self.device = device

        self.name = "transformer_vae"
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None
        nheads = model_config["heads"]
        layers = model_config["layers"]
        self.moving_dimension = model_config["internal_dimension"]
        feed_forward_dim = model_config["feed_forward"]
        self.embedder = nn.Conv1d(kernel_size=3, in_channels=embeddings_static.shape[1],
                                  out_channels=self.moving_dimension, stride=1, padding=1, bias=False)
        self.deembed = nn.ConvTranspose1d(kernel_size=3, in_channels=self.moving_dimension,
                                          out_channels=embeddings_static.shape[0], padding=1, bias=False)
        self.encoder = nn.Embedding(embeddings_static.shape[0], embeddings_static.shape[1])
        self.encoder.weight.data.copy_(embeddings_static)
        self.encoder.weight.requires_grad = False

        encoder_layers = TransformerEncoderLayer(d_model=self.moving_dimension, nhead=nheads,
                                                 dim_feedforward=feed_forward_dim)
        self.transformer_encoder = TransformerEncoder(encoder_layers, layers)

        decoder_layer = TransformerEncoderLayer(self.moving_dimension, nhead=nheads, dim_feedforward=feed_forward_dim)
        self.transformer_decoder = TransformerEncoder(decoder_layer, layers)
        h_dim = input_size * self.moving_dimension
        self.fc1: nn.Module = nn.Linear(h_dim, z_dim)
        self.fc2: nn.Module = nn.Linear(h_dim, z_dim)
        self.fc3: nn.Module = nn.Linear(z_dim, h_dim)
        # self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedder.weight.data.uniform_(-initrange, initrange)
        self.deembed.weight.data.uniform_(-initrange, initrange)

    def bottleneck(self, h):
        mu = self.fc1(h)
        log_var = self.fc2(h)
        z = reparameterization(mu, log_var, self.device)
        return z, mu, log_var

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x):
        input_len = x.shape[1]
        src = self.encoder(x).transpose(1, 2)
        src = self.embedder(src)
        src = src.transpose(1, 2)
        output = self.transformer_encoder(src)
        output = output.view(x.shape[0], -1)
        z, mu, log_var = self.bottleneck(output)
        output = self.fc3(z)
        output = output.view(z.shape[0], input_len, -1)
        output = self.transformer_decoder(output)
        output = self.deembed(output.transpose(1, 2))
        return output, mu, log_var
