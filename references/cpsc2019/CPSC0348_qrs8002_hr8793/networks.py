import numpy as np
import torch
import torch.nn as nn

from encoder import ResBlock, rnn_encoder
from decoder import rnn_decoder

from processor import preprocess, postprocess


class seq2seq(nn.Module):

    def __init__(self, device, teacher_forcing_ratio=0., dropout=0., random_init=False):
        super(seq2seq, self).__init__()
        self.random_init = random_init
        self.teacher_forcing_ratio = teacher_forcing_ratio

        self.encoder = rnn_encoder(device=device, feature_size=250, encoder_hidden=256, dropout=dropout)
        self.decoder = rnn_decoder(device=device, decoder_hidden=256,teacher_forcing_ratio=self.teacher_forcing_ratio,
                                   dropout=dropout, random_init=random_init)

    def forward(self, inputs, targets):
        contexts, encoder_state = self.encoder(inputs)
        outputs, final_state = self.decoder(targets, contexts.transpose(1, 2), encoder_state)
        return outputs

    def sample(self, inputs, return_attns=False):
        contexts, encoder_state = self.encoder(inputs)
        outputs, attns = self.decoder.sample(contexts.transpose(1, 2), encoder_state)
        return outputs if not return_attns else (outputs, attns)


class CNN_en1D(nn.Module):
    def __init__(self, L, ks=71):
        super(CNN_en1D, self).__init__()
        self.L = L
        self.C = 64
        self.cnn_g1 = nn.Sequential(nn.Conv1d(1, self.C, ks, 1, padding=(ks-1)//2),
                                    nn.BatchNorm1d(self.C),
                                    nn.ReLU(self.C),
                                    ResBlock(self.C, ks=ks, norm_func=nn.BatchNorm1d),
                                    ResBlock(self.C, ks=ks, norm_func=nn.BatchNorm1d))

        self.cnn_g2 = nn.Sequential(ResBlock(self.C, ks=ks, norm_func=nn.BatchNorm1d),
                                    ResBlock(self.C, ks=ks, norm_func=nn.BatchNorm1d),
                                    nn.Conv1d(self.C, 1, ks, 1, padding=(ks-1)//2))

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)


    def forward(self, input):
        cnn_out = self.cnn_g1(input)
        cnn_out = self.cnn_g2(cnn_out)
        cnn_out = cnn_out.view(-1, self.L)

        return torch.sigmoid(cnn_out)


class qrs_detector(object):
    def __init__(self, model_path, model_type='fcn', fs=500):
        assert model_type in ['fcn', 'seq2seq']
        self.model_type=model_type
        self.fs = fs
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        if self.model_type=='fcn':
            self.model = CNN_en1D(L=5000).to(self.device)
        else:
            self.model = seq2seq(device=self.device).to(self.device)

        #print('Loading model from {}.'.format(model_path))

        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()

        # print('Epoch: {}，Se: {}， pre: {}, F1: {}, acc:{}'.
        #       format(checkpoint['epoch'], checkpoint['se'], checkpoint['pre'], checkpoint['f1'],
        #              checkpoint['cpsc_acc']))

    def detection(self, signal):
        signal = preprocess(signal, fs=self.fs)
        signal_tensor = torch.from_numpy(signal).view(1, 1, -1).float().to(self.device)

        if self.model_type=='fcn':
            outputs = self.model(signal_tensor)  # outputs: 1 x 1 x beats
            beat_locs2 = list(outputs.squeeze().detach().cpu().numpy())
            beat_locs = postprocess(beat_locs2, margin=12)
        else:
            outputs = self.model.sample(signal_tensor)  # outputs: 1 x 1 x beats
            beat_locs = np.rint(outputs.view(-1).detach().cpu().numpy()).astype(np.int) if outputs \
                                                                                           is not None else np.array([])
        return beat_locs




