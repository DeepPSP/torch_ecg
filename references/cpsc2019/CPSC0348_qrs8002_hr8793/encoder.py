import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict


norm_func = nn.InstanceNorm1d
use_bias = False  # if norm_func is nn.BatchNorm1d else True


class ResBlock(nn.Module):
    def __init__(self, inc, outc=None, ks=71, stride=1, norm_func=norm_func):
        super(ResBlock, self).__init__()
        if outc is None:
            outc = inc
        self.conv = nn.Sequential(
            nn.Conv1d(inc, outc, ks, stride, padding=(ks-1)//2, bias=use_bias),
            norm_func(outc),
            nn.ReLU(inplace=True),
            nn.Conv1d(outc, outc, ks, 1, padding=(ks-1)//2, bias=use_bias),
            norm_func(outc)
        )
        self.shortcut = nn.Sequential(nn.Conv1d(inc, outc, ks, stride, padding=(ks-1)//2, bias=use_bias),
                                      norm_func(outc)) if inc!=outc or stride!=1 else None

    def forward(self, x):
        out = self.conv(x)
        residual = x if self.shortcut is None else self.shortcut(x)
        out += residual
        return F.relu(out)


class cnn_encoder(nn.Module):
    def __init__(self, device, encoder_hidden=256):
        super(cnn_encoder, self).__init__()

        self.device = device
        self.encoder_hidden = encoder_hidden

        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv1d(1, encoder_hidden//8, kernel_size=71, stride=1, padding=35, bias=use_bias)),
            ('norm0', norm_func(encoder_hidden//8)),
            ('relu0', nn.ReLU(inplace=True))
        ]))
        self.features.add_module('rb1', ResBlock(encoder_hidden//8, encoder_hidden//8, stride=1))
        self.features.add_module('rb2', ResBlock(encoder_hidden//8, encoder_hidden//8, stride=1))
        self.features.add_module('rb3', ResBlock(encoder_hidden//8, encoder_hidden//4, stride=2))
        self.features.add_module('rb4', ResBlock(encoder_hidden//4, encoder_hidden//4, stride=1))
        self.features.add_module('rb5', ResBlock(encoder_hidden//4, encoder_hidden//2, stride=5))
        self.features.add_module('rb7', ResBlock(encoder_hidden//2, encoder_hidden//2, stride=1))
        self.features.add_module('rb8', ResBlock(encoder_hidden//2, encoder_hidden, stride=5))
        self.features.add_module('rb9', ResBlock(encoder_hidden, encoder_hidden, stride=1))
        self.features.add_module('rb10', ResBlock(encoder_hidden, encoder_hidden, stride=5))


        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        context = self.features(x)  # B x (C=hidden) X (L=t)
        state = None
        return context, state  # return: B x hidden x t


class rnn_encoder(nn.Module):
    def __init__(self, device, feature_size=250, encoder_hidden=256, num_layers=2, ks=71, dropout=0.):
        super(rnn_encoder, self).__init__()

        self.device = device
        self.feature_size = feature_size
        self.encoder_hidden = encoder_hidden
        self.num_layers = num_layers
        depth = 32

        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv1d(1, depth, kernel_size=ks, stride=1, padding=(ks-1)//2, bias=use_bias)),
            ('norm0', norm_func(depth)),
            ('relu0', nn.ReLU(inplace=True))
        ]))

        self.features.add_module('rb1', ResBlock(depth, depth, ks=ks, stride=1))
        self.features.add_module('rb2', ResBlock(depth, depth, ks=ks, stride=1))
        self.features.add_module('rb3', ResBlock(depth, depth, ks=ks, stride=1))
        self.features.add_module('rb4', ResBlock(depth, depth, ks=ks, stride=1))

        self.features.add_module('post', nn.Sequential(OrderedDict([
            ('conv1', nn.Conv1d(depth, 1, kernel_size=ks, stride=1, padding=(ks-1)//2, bias=use_bias)),
            ('norm1', norm_func(1)),
            ('relu1', nn.ReLU(inplace=True))
        ])))

        self.rnn = nn.LSTM(input_size=self.feature_size, hidden_size=self.encoder_hidden, num_layers=num_layers,
                           bidirectional=True, dropout=dropout)


        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)


    def forward(self, x):
        batch_size = x.size(0)
        cnn_features = self.features(x) # B x (C=1) X L

        rnn_input = torch.split(cnn_features.squeeze(1), self.feature_size, dim=1)
        rnn_input = torch.stack(rnn_input, 0) # t x B x L/t

        init_h = torch.empty(self.num_layers*2, batch_size, self.encoder_hidden).to(self.device)
        init_c = torch.empty(self.num_layers*2, batch_size, self.encoder_hidden).to(self.device)
        nn.init.zeros_(init_h)
        nn.init.zeros_(init_c)
        state = (init_h, init_c)

        context, state = self.rnn(rnn_input, state) # context: t x B x hidden
        context = torch.split(context, self.encoder_hidden, dim=2)
        context = context[0] + context[1]

        hn, cn = state
        hn = torch.split(hn, 2, dim=0)
        hn = hn[0] + hn[1]
        cn = torch.split(cn, 2, dim=0)
        cn = cn[0] + cn[1]
        state = (hn, cn)

        return context.permute(1, 2, 0), state # return: B x hidden x t