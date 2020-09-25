import torch
import torch.nn as nn

import random

from attention import general_attention

sos_token = 0.
eos_token = 5500.


class rnn_decoder(nn.Module):
    def __init__(self, device, decoder_hidden=256, num_layers=2,
                 teacher_forcing_ratio=0., dropout=0., random_init=False):

        super(rnn_decoder, self).__init__()
        self.device = device
        self.decoder_hidden = decoder_hidden
        self.num_layers = num_layers
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.random_init = random_init

        # self.fc_in = nn.Linear(decoder_hidden+1, decoder_hidden)
        self.rnn = nn.LSTM(input_size=decoder_hidden + 1, hidden_size=decoder_hidden, num_layers=num_layers,
                           dropout=dropout)

        self.attention = general_attention(decoder_hidden)
        self.fc = nn.Linear(decoder_hidden, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.LSTM):
                for layer in m._all_weights:
                    nn.init.xavier_uniform_(m._parameters[layer[0]])
                    nn.init.xavier_uniform_(m._parameters[layer[1]])

        nn.init.constant_(self.fc.bias, 2500)

    def forward_step(self, input, state, contexts):  # input: B x hidden, context: B x t x hidden
        output, state = self.rnn(input.unsqueeze(0), state)  # output: 1 x B x hidden
        hidden, attn_weigths = self.attention(output.squeeze(0), contexts)  # B x hidden
        # output = self.linear(hidden) # B x 1
        output = hidden
        return output, state, attn_weigths

    def forward(self, targets, contexts, init_state=None):  # teacher forcing
        batch_size = contexts.size(0)
        if init_state == None:
            init_h = torch.empty(self.num_layers, batch_size, self.decoder_hidden).to(self.device)
            init_c = torch.empty(self.num_layers, batch_size, self.decoder_hidden).to(self.device)
            if self.random_init:
                nn.init.orthogonal_(init_h)
                nn.init.orthogonal_(init_c)
            else:
                nn.init.zeros_(init_h)
                nn.init.zeros_(init_c)
            init_state = (init_h, init_c)

        outputs, state, attns = [], init_state, []

        targets = torch.split(targets, split_size_or_sections=1, dim=2)  # time x [B x 1 x 1]
        max_time_step = len(targets)

        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False
        decoder_input = torch.cat([contexts[:, -1, :],
                                   torch.tensor(batch_size * [[sos_token]]).to(self.device)], 1)  # Bx(hidden+1)

        for t in range(0, max_time_step):
            at_hidden, state, attn_weights = self.forward_step(decoder_input, state, contexts)
            output = self.fc(at_hidden)
            outputs += [output]
            attns += [attn_weights]

            label = targets[t].squeeze(1) if use_teacher_forcing else output.detach()
            decoder_input = torch.cat([at_hidden.detach(), label], 1)

        outputs = torch.stack(outputs, 2)
        attns = torch.stack(attns, 2)
        return outputs, attns

    def sample(self, contexts, init_state=None, max_time_step=50):
        batch_size = contexts.size(0)
        if init_state == None:
            init_h = torch.empty(self.num_layers, batch_size, self.decoder_hidden).to(self.device)
            init_c = torch.empty(self.num_layers, batch_size, self.decoder_hidden).to(self.device)
            if self.random_init:
                nn.init.orthogonal_(init_h)
                nn.init.orthogonal_(init_c)
            else:
                nn.init.zeros_(init_h)
                nn.init.zeros_(init_c)
            init_state = (init_h, init_c)

        outputs, state, attns = [], init_state, []

        decoder_input = torch.cat([contexts[:, -1, :], torch.tensor([[sos_token]]).to(self.device)], 1)  # Bx(hidden+1)

        for t in range(0, max_time_step):
            at_hidden, state, attn_weights = self.forward_step(decoder_input, state, contexts)

            output = self.fc(at_hidden)
            if output.item() >= eos_token / 1.1:
                break
            outputs += [output]
            attns += [attn_weights]

            label = output.detach()
            decoder_input = torch.cat([at_hidden.detach(), label], 1)

        outputs = torch.stack(outputs, 2) if len(outputs) > 0 else None
        attns = torch.stack(attns, 2) if len(attns) > 0 else None
        return outputs, attns