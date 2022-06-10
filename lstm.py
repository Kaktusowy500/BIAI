import torch
import torch.nn as nn


class LSTM(nn.Module):
    """
    input_size - will be 1 in this example since we have only 1 predictor (a sequence of previous values)
    hidden_size - Can be chosen to dictate how much hidden "long term memory" the network will have
    output_size - This will be equal to the prediciton_periods input to get_x_y_pairs
    """

    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()

        self.is_cuda = False
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):
        if hidden == None:
            self.hidden = [torch.zeros(1, 1, self.hidden_size),
                           torch.zeros(1, 1, self.hidden_size)]
        else:
            self.hidden = hidden

        if self.is_cuda:
            for i, hidden_layer in enumerate(self.hidden):
                self.hidden[i] = hidden_layer.cuda()

        lstm_out, self.hidden = self.lstm(x.view(len(x), 1, -1),
                                          self.hidden)

        predictions = self.linear(lstm_out.view(len(x), -1))

        return predictions[-1], self.hidden

    def cuda(self):
        self.is_cuda = True
        self.to(device='cuda:0')
