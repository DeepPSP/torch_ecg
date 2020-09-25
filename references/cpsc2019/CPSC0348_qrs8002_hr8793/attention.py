import torch
import torch.nn as nn


class dot_attention(nn.Module):
    def __init__(self, hidden_size):
        super(dot_attention, self).__init__()
        self.linear_out = nn.Linear(2*hidden_size, hidden_size)
        self.softmax = nn.Softmax(dim=1)
        self.act = nn.ReLU()

    def forward(self, x, context): # x: b x hidden, context: b x time x hidden
        gamma_h = x.unsqueeze(2)    # batch x hidden x 1
        weights = torch.bmm(context, gamma_h).squeeze(2)   # batch x time
        weights = self.softmax(weights)   # batch x time
        c_t = torch.bmm(weights.unsqueeze(1), context).squeeze(1) # batch x hidden

        output = self.act(self.linear_out(torch.cat([c_t, x], 1)))
        return output, weights

class general_attention(nn.Module):
    def __init__(self, hidden_size, acti=False):
        super(general_attention, self).__init__()
        self.linear_in = nn.Linear(hidden_size, hidden_size)
        self.linear_out = nn.Linear(2*hidden_size, hidden_size)
        self.softmax = nn.Softmax(dim=1)
        self.act = nn.ReLU()
        self.acti = acti

    def forward(self, x, context): # x: b x hidden, context: b x time x hidden
        gamma_h = self.linear_in(x).unsqueeze(2)    # batch x hidden x 1
        if self.acti:
            gamma_h = torch.tanh(gamma_h)
        weights = torch.bmm(context, gamma_h).squeeze(2)   # batch x time
        weights = self.softmax(weights)   # batch x time
        c_t = torch.bmm(weights.unsqueeze(1), context).squeeze(1) # batch x hidden

        output = self.act(self.linear_out(torch.cat([c_t, x], 1)))
        return output, weights

class concat_attention(nn.Module):
    def __init__(self, hidden_size, time_step=20):
        super(concat_attention, self).__init__()
        self.linear_v = nn.Linear(hidden_size, time_step)
        self.linear_g = nn.Linear(hidden_size, time_step)
        self.linear_h = nn.Linear(time_step, 1)

        self.linear_o = nn.Linear(2*hidden_size, hidden_size)
        self.softmax = nn.Softmax(dim=1)
        self.act = nn.ReLU()

        self.time_step = time_step

    def forward(self, x, context):# x: b x hidden, context: b x time x hidden
        gamma_c = self.linear_v(context) # b x time x time
        gamma_h = self.linear_g(x).unsqueeze(1).repeat(1, self.time_step, 1) # b x time -> b x 1 x time -> b x time x time
        weights = self.linear_h(torch.tanh(gamma_c + gamma_h)).squeeze(2)
        weights = self.softmax(weights)
        c_t = torch.bmm(weights.unsqueeze(1), context).squeeze(1)

        output = self.act(self.linear_o(torch.cat([c_t, x], 1)))
        return output, weights