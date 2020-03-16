import torch
import torch.nn as nn

from models.vae_template import VaeTemplate


def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.xavier_uniform_(m.weight)


class GatedCNN(VaeTemplate, nn.Module):

    def __init__(self, model_config, hidden_size, embedding_size, data_length, device, embeddings_static,
                 requires_grad=False):
        self.name = "gated_cnn"
        seq_len = data_length
        vocab_size = embeddings_static.shape[0]
        embd_size = embeddings_static.shape[1]
        n_layers = model_config["layers"]
        kernel = [model_config["kernel_size_0"], embeddings_static.shape[1]]
        out_chs = model_config["channels"]
        res_block_count = model_config["residual"]
        ans_size = hidden_size
        self.out_chs = out_chs
        self.res_block_count = res_block_count
        # hidden_size = out_chs * seq_len
        super(GatedCNN, self).__init__(None, None, device, hidden_size, embedding_size, embedding=None)
        self.embedding = nn.Embedding(vocab_size, embd_size)
        self.embedding.weight.data.copy_(embeddings_static)
        self.embedding.weight.requires_grad = requires_grad
        padding = int((kernel[0] - 1) / 2)
        # nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, ...
        self.conv_0 = nn.Conv2d(1, out_chs, kernel, padding=(padding, 0))
        self.b_0 = nn.Parameter(torch.randn(1, out_chs, 1, 1), requires_grad=True)
        self.conv_gate_0 = nn.Conv2d(1, out_chs, kernel, padding=(padding, 0))
        self.c_0 = nn.Parameter(torch.randn(1, out_chs, 1, 1), requires_grad=True)

        self.conv = nn.ModuleList(
            [nn.Conv2d(out_chs, out_chs, (kernel[0], 1), padding=(padding, 0)) for _ in range(n_layers)])
        self.conv_gate = nn.ModuleList(
            [nn.Conv2d(out_chs, out_chs, (kernel[0], 1), padding=(padding, 0)) for _ in range(n_layers)])
        self.b = nn.ParameterList(
            [nn.Parameter(torch.randn(1, out_chs, 1, 1), requires_grad=True) for _ in range(n_layers)])
        self.c = nn.ParameterList(
            [nn.Parameter(torch.randn(1, out_chs, 1, 1), requires_grad=True) for _ in range(n_layers)])

        self.fc = nn.Linear(out_chs * seq_len, ans_size)
        self.fc_d = nn.Linear(ans_size, out_chs * seq_len)

        self.conv_d = nn.ModuleList(
            [nn.ConvTranspose2d(out_chs, out_chs, (kernel[0], 1), padding=(padding, 0)) for _ in range(n_layers)])
        self.conv_gate_d = nn.ModuleList(
            [nn.ConvTranspose2d(out_chs, out_chs, (kernel[0], 1), padding=(padding, 0)) for _ in range(n_layers)])
        self.b_d = nn.ParameterList(
            [nn.Parameter(torch.randn(1, out_chs, 1, 1), requires_grad=True) for _ in range(n_layers)])
        self.c_d = nn.ParameterList(
            [nn.Parameter(torch.randn(1, out_chs, 1, 1), requires_grad=True) for _ in range(n_layers)])

        self.conv_l = nn.ConvTranspose2d(out_chs, vocab_size, (kernel[0], 1), padding=(padding, 0))
        self.b_l = nn.Parameter(torch.randn(1, vocab_size, 1, 1), requires_grad=True)
        self.conv_gate_l = nn.ConvTranspose2d(out_chs, vocab_size, (kernel[0], 1), padding=(padding, 0))
        self.c_l = nn.Parameter(torch.randn(1, vocab_size, 1, 1), requires_grad=True)
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        # Embedding
        bs = x.size(0)  # batch size
        seq_len = x.size(1)
        x = self.embedding(x)  # (bs, seq_len, embd_size)
        # CNN
        x = x.unsqueeze(1)  # (bs, Cin, seq_len, embd_size), insert Channnel-In dim
        A = self.conv_0(x)  # (bs, Cout, seq_len, 1)
        res_input = A
        h = A

        for i, (conv, conv_gate) in enumerate(zip(self.conv, self.conv_gate)):
            A = conv(h)
            B = conv_gate(h)
            h = A * self.sigmoid(B)  # (bs, Cout, seq_len, 1)
            if i % self.res_block_count == 0:  # size of each residual block
                h += res_input
                res_input = h

        h = h.view(bs, -1)  # (bs, Cout*seq_len)
        h = self.fc(h)
        return self.bottleneck(h)

    def forward(self, x):
        bs = x.size(0)  # batch size
        seq_len = x.size(1)
        z, mu, var = self.encode(x)
        z = self.fc3(z)
        res_input = self.fc_d(z)
        res_input = res_input.view(bs, self.out_chs, seq_len, -1)
        h = res_input
        for i, (conv, conv_gate) in enumerate(zip(self.conv_d, self.conv_gate_d)):
            A = conv(h)
            B = conv_gate(h)
            h = A * self.sigmoid(B)  # (bs, Cout, seq_len, 1)

            if i % self.res_block_count == 0:  # size of each residual block
                h += res_input
                res_input = h
        A = self.conv_l(h)

        h = A.squeeze(3)
        return h, mu, var
