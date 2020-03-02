import torch
import torch.nn as nn

from models.vae_template import VaeTemplate


def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.xavier_uniform_(m.weight)


class GatedCNNShrinking(VaeTemplate, nn.Module):

    def __init__(self, model_config, hidden_size, embedding_size, data_length, device, embeddings_static):
        self.name = "gated_cnn"
        seq_len = data_length
        vocab_size = embeddings_static.shape[0]
        embd_size = embeddings_static.shape[1]
        n_layers = model_config["layers"]
        kernel = [model_config["kernel_size_0"], model_config["kernel_size_1"]]
        out_chs = model_config["channels"]
        res_block_count = model_config["residual"]
        self.out_chs = out_chs
        self.res_block_count = res_block_count

        super(GatedCNNShrinking, self).__init__(None, None, device, hidden_size, embedding_size, embedding=None)
        self.embedding = nn.Embedding(vocab_size, embd_size)
        padding = int((kernel[0] - 1) / 2)

        self.conv_0 = nn.Conv2d(1, out_chs, kernel, padding=(padding, 0))
        self.b_0 = nn.Parameter(torch.randn(1, out_chs, 1, 1))
        self.conv_gate_0 = nn.Conv2d(1, out_chs, kernel, padding=(padding, 0))
        self.c_0 = nn.Parameter(torch.randn(1, out_chs, 1, 1))

        self.conv = nn.ModuleList(
            [nn.Conv2d(out_chs, out_chs, (kernel[0], 1), padding=(padding, 0)) for _ in range(n_layers)])
        self.conv_gate = nn.ModuleList(
            [nn.Conv2d(out_chs, out_chs, (kernel[0], 1), padding=(padding, 0)) for _ in range(n_layers)])
        self.b = nn.ParameterList([nn.Parameter(torch.randn(1, out_chs, 1, 1)) for _ in range(n_layers)])
        self.c = nn.ParameterList([nn.Parameter(torch.randn(1, out_chs, 1, 1)) for _ in range(n_layers)])

        self.conv.apply(init_weights)
        self.conv_gate.apply(init_weights)
        self.fc_e = nn.Linear(out_chs * seq_len, hidden_size *2)
        self.fc_h1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_h2 = nn.Linear(hidden_size, hidden_size * 2)
        self.fc_d = nn.Linear(hidden_size, out_chs * seq_len)

        self.conv_d = nn.ModuleList(
            [nn.ConvTranspose2d(out_chs, out_chs, (kernel[0], 1), padding=(padding, 0)) for _ in range(n_layers)])
        self.conv_gate_d = nn.ModuleList(
            [nn.ConvTranspose2d(out_chs, out_chs, (kernel[0], 1), padding=(padding, 0)) for _ in range(n_layers)])

        self.b_d = nn.ParameterList([nn.Parameter(torch.randn(1, out_chs, 1, 1)) for _ in range(n_layers)])
        self.c_d = nn.ParameterList([nn.Parameter(torch.randn(1, out_chs, 1, 1)) for _ in range(n_layers)])

        self.conv_l = nn.ConvTranspose2d(out_chs, vocab_size, (kernel[0], 1), padding=(padding, 0))
        self.b_l = nn.Parameter(torch.randn(1, vocab_size, 1, 1))
        self.conv_gate_l = nn.ConvTranspose2d(out_chs, vocab_size, (kernel[0], 1), padding=(padding, 0))
        self.c_l = nn.Parameter(torch.randn(1, vocab_size, 1, 1))
        self.sigmoid = nn.Sigmoid()

        self.conv_l.apply(init_weights)
        self.conv_gate_l.apply(init_weights)
        self.conv_0.apply(init_weights)
        self.conv_gate_0.apply(init_weights)
        self.conv_d.apply(init_weights)
        self.conv_gate_d.apply(init_weights)

    def forward(self, x):
        bs = x.size(0)
        seq_len = x.size(1)
        x = self.embedding(x)
        x = x.unsqueeze(1)
        A = self.conv_0(x)
        A += self.b_0.repeat(1, 1, seq_len, 1)
        B = self.conv_gate_0(x)
        B += self.c_0.repeat(1, 1, seq_len, 1)
        h = A * self.sigmoid(B)
        res_input = h

        for i, (conv, conv_gate) in enumerate(zip(self.conv, self.conv_gate)):
            A = conv(h) + self.b[i].repeat(1, 1, seq_len, 1)
            B = conv_gate(h) + self.c[i].repeat(1, 1, seq_len, 1)
            h = A * self.sigmoid(B)
            if i % self.res_block_count == 0:
                h += res_input
                res_input = h

        h = h.view(bs, -1)
        h = self.fc(h)
        z, _, _ = self.bottleneck(h)
        z = self.fc3(z)

        res_input = self.fc_d(z)
        res_input = res_input.view(bs, self.out_chs, seq_len, -1)
        h = res_input
        for i, (conv, conv_gate) in enumerate(zip(self.conv_d, self.conv_gate_d)):
            A = conv(h) + self.b[i].repeat(1, 1, seq_len, 1)
            B = conv_gate(h) + self.c[i].repeat(1, 1, seq_len, 1)
            h = A * self.sigmoid(B)
            if i % self.res_block_count == 0:
                h += res_input
                res_input = h

        A = self.conv_l(h)
        A += self.b_l.repeat(1, 1, seq_len, 1)
        B = self.conv_gate_l(h)
        B += self.c_l.repeat(1, 1, seq_len, 1)
        h = A * self.sigmoid(B)
        h = h.squeeze(3)

        return h
