import torch
import torch.nn as nn

class RNN_base(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, activation):
        super().__init__()
        sigma = 0.01
        self.W_xh = nn.Parameter(torch.randn(input_size, hidden_size)*sigma)
        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size)*sigma) 
        self.b_h = nn.Parameter(torch.zeros(hidden_size))
        self.activation = activation
        self.W_hq = nn.Parameter(torch.randn(hidden_size, output_size)*sigma)
        self.b_q = nn.Parameter(torch.zeros(output_size))
    
    def forward(self, X, state=None):
        if state is None:
            state = torch.zeros(X.shape[1], self.W_hh.shape[0], device=X.device)
        outputs = []
        states = []
        for Xt in X:  #X is the input with shape (seq_len, batch_size, input_size)
            # For performance it's better to concatenate Xt with state and W_xh and W_hh to do one multiplication then split it
            state = self.activation(Xt @ self.W_xh + state @ self.W_hh + self.b_h)
            states.append(state)
            output = state @ self.W_hq + self.b_q
            outputs.append(output)
        return outputs, states    

    @property
    def device(self):
        return next(self.parameters()).device

class RNN_lstm(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        sigma = 0.01
        self.W_xi = nn.Parameter(torch.randn(input_size, hidden_size)*sigma)
        self.W_xf = nn.Parameter(torch.randn(input_size, hidden_size)*sigma)
        self.W_xo = nn.Parameter(torch.randn(input_size, hidden_size)*sigma)
        self.W_xg = nn.Parameter(torch.randn(input_size, hidden_size)*sigma)

        self.W_hi = nn.Parameter(torch.randn(hidden_size, hidden_size)*sigma)
        self.W_hf = nn.Parameter(torch.randn(hidden_size, hidden_size)*sigma)
        self.W_ho = nn.Parameter(torch.randn(hidden_size, hidden_size)*sigma)
        self.W_hg = nn.Parameter(torch.randn(hidden_size, hidden_size)*sigma)

        self.b_i = nn.Parameter(torch.zeros(hidden_size))
        self.b_f = nn.Parameter(torch.zeros(hidden_size))
        self.b_o = nn.Parameter(torch.zeros(hidden_size))
        self.b_g = nn.Parameter(torch.zeros(hidden_size))

        self.W_hq = nn.Parameter(torch.randn(hidden_size, output_size)*sigma)
        self.b_q = nn.Parameter(torch.zeros(output_size))

    def forward(self, X, state=None):
        if state is None:
            # Hidden state and cell state
            state = (torch.zeros(X.shape[1], self.W_xi.shape[1], device=X.device), torch.zeros(X.shape[1], self.W_xi.shape[1], device=X.device))
        
        outputs = []
        states = []
        for Xt in X:
            H, C = state
            # For performance it's better to concatenate the 4 gate weights and do a single matrix multiplication then split result
            # i kept it this way for readability
            I = torch.sigmoid(Xt @ self.W_xi + H @ self.W_hi + self.b_i)
            F = torch.sigmoid(Xt @ self.W_xf + H @ self.W_hf + self.b_f)
            O = torch.sigmoid(Xt @ self.W_xo + H @ self.W_ho + self.b_o)
            G = torch.tanh(Xt @ self.W_xg + H @ self.W_hg + self.b_g)
            C = F * C + I * G
            H = O * torch.tanh(C)
            state = (H, C)
            states.append(state)
            output = H @ self.W_hq + self.b_q
            outputs.append(output)
        return outputs, states

    @property
    def device(self):
        return next(self.parameters()).device

class RNN_gru(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        sigma = 0.01
        self.W_xr = nn.Parameter(torch.randn(input_size, hidden_size)*sigma)
        self.W_xu = nn.Parameter(torch.randn(input_size, hidden_size)*sigma)
        self.W_xh = nn.Parameter(torch.randn(input_size, hidden_size)*sigma)

        self.W_hr = nn.Parameter(torch.randn(hidden_size, hidden_size)*sigma)
        self.W_hu = nn.Parameter(torch.randn(hidden_size, hidden_size)*sigma)
        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size)*sigma)

        self.b_r = nn.Parameter(torch.zeros(hidden_size))
        self.b_u = nn.Parameter(torch.zeros(hidden_size))
        self.b_h = nn.Parameter(torch.zeros(hidden_size))

        self.W_hq = nn.Parameter(torch.randn(hidden_size, output_size)*sigma)
        self.b_q = nn.Parameter(torch.zeros(output_size))

    def forward(self, X, state=None):
        if state is None:
            state = torch.zeros(X.shape[1], self.W_xh.shape[1], device=X.device)

        outputs = []
        states = []
        for Xt in X:
            R = torch.sigmoid(Xt @ self.W_xr + state @ self.W_hr + self.b_r)
            U = torch.sigmoid(Xt @ self.W_xu + state @ self.W_hu + self.b_u)
            H = torch.tanh(Xt @ self.W_xh + (state * R) @ self.W_hh + self.b_h)
            state = U * state + (1 - U) * H
            states.append(state)
            outputs.append(state @ self.W_hq + self.b_q)
        return outputs, states

    @property
    def device(self):
        return next(self.parameters()).device