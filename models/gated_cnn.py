import torch
import torch.nn as nn

from models.vae_template import VaeTemplate


class GatedCNN(VaeTemplate, nn.Module):

    def __init__(self, model_config, hidden_size, embedding_size, data_length, device, embeddings_static):
        self.name = "gated_cnn"
        seq_len = data_length
        vocab_size = embeddings_static.shape[0]
        embd_size = embeddings_static.shape[1]
        n_layers = model_config["layers"]
        kernel = model_config["kernel_size"]
        out_chs = model_config["channels"]
        res_block_count = model_config["residual"]
        ans_size = hidden_size
        self.out_chs = out_chs
        self.res_block_count = res_block_count

        embedding = torch.nn.Embedding(23, 30, max_norm=1)
        embedding.weight.data.copy_(embeddings_static)
        embedding.weight.requires_grad = False
        super(GatedCNN, self).__init__(None, None, device, hidden_size, embedding_size, embedding=embedding)
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

        self.fc = nn.Linear(out_chs * seq_len, ans_size)
        self.fc_d = nn.Linear(ans_size, out_chs * seq_len)

        self.conv_d = nn.ModuleList(
            [nn.ConvTranspose2d(out_chs, out_chs, (kernel[0], 1), padding=(padding, 0)) for _ in range(n_layers)])
        self.conv_gate_d = nn.ModuleList(
            [nn.ConvTranspose2d(out_chs, out_chs, (kernel[0], 1), padding=(padding, 0)) for _ in range(n_layers)])
        self.b_d = nn.ParameterList([nn.Parameter(torch.randn(1, out_chs, 1, 1)) for _ in range(n_layers)])
        self.c_d = nn.ParameterList([nn.Parameter(torch.randn(1, out_chs, 1, 1)) for _ in range(n_layers)])

        self.conv_l = nn.Conv2d(1, vocab_size, kernel, padding=(padding, 0))
        self.b_l = nn.Parameter(torch.randn(1, vocab_size, 1, 1))
        self.conv_gate_l = nn.Conv2d(1, vocab_size, kernel, padding=(padding, 0))
        self.c_l = nn.Parameter(torch.randn(1, vocab_size, 1, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Embedding
        bs = x.size(0)  # batch size
        seq_len = x.size(1)
        x = self.embedding(x)  # (bs, seq_len, embd_size)

        # CNN
        x = x.unsqueeze(1)  # (bs, Cin, seq_len, embd_size), insert Channnel-In dim
        # Conv2d
        #    Input : (bs, Cin,  Hin,  Win )
        #    Output: (bs, Cout, Hout, Wout)
        A = self.conv_0(x)  # (bs, Cout, seq_len, 1)
        A += self.b_0.repeat(1, 1, seq_len, 1)
        B = self.conv_gate_0(x)  # (bs, Cout, seq_len, 1)
        B += self.c_0.repeat(1, 1, seq_len, 1)
        h = A * self.sigmoid(B)  # (bs, Cout, seq_len, 1)
        res_input = h

        for i, (conv, conv_gate) in enumerate(zip(self.conv, self.conv_gate)):
            A = conv(h) + self.b[i].repeat(1, 1, seq_len, 1)
            B = conv_gate(h) + self.c[i].repeat(1, 1, seq_len, 1)
            h = A * self.sigmoid(B)  # (bs, Cout, seq_len, 1)
            if i % self.res_block_count == 0:  # size of each residual block
                h += res_input
                res_input = h

        h = h.view(bs, -1)  # (bs, Cout*seq_len)
        h = self.fc(h)  # (bs, ans_size)
        z, _, _ = self.bottleneck(h)
        z = self.fc3(z)

        res_input = self.fc_d(z)
        res_input = res_input.view(bs, self.out_chs, seq_len, -1)
        h = res_input
        for i, (conv, conv_gate) in enumerate(zip(self.conv_d, self.conv_gate_d)):
            A = conv(h) + self.b[i].repeat(1, 1, seq_len, 1)
            B = conv_gate(h) + self.c[i].repeat(1, 1, seq_len, 1)
            h = A * self.sigmoid(B)  # (bs, Cout, seq_len, 1)
            if i % self.res_block_count == 0:  # size of each residual block
                h += res_input
                res_input = h
        A = self.conv_l(x)  # (bs, Cout, seq_len, 1)
        A += self.b_l.repeat(1, 1, seq_len, 1)
        B = self.conv_gate_l(x)  # (bs, Cout, seq_len, 1)
        B += self.c_l.repeat(1, 1, seq_len, 1)
        h = A * self.sigmoid(B)
        h = h.squeeze(3)
        return h
