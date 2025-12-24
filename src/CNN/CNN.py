import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        # Kaiming Init for CNN
        # fan_in = in_channels * k_h * k_w
        fan_in = in_channels * kernel_size[0] * kernel_size[1]
        std = (2.0 / fan_in)**0.5
        
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1]) * std)
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        if self.padding > 0:
            x = self.pad(x, self.padding)
        
        res = torch.stack([self.corr2d_multi_in_out(img, self.weight) for img in x])
        return res + self.bias.view(1, -1, 1, 1)

    def pad(self, X, p):
        #(B, C, H, W) we pad H and W
        B, C, H, W = X.shape
        padded = torch.zeros((B, C, H + 2*p, W + 2*p), device=X.device)
        padded[:,:, p:p+H, p:p+W] = X
        return padded

    def corr2d(self, X, K):
        h, w = K.shape
        out_h = (X.shape[0] - h) // self.stride + 1
        out_w = (X.shape[1] - w) // self.stride + 1
        Y = torch.zeros(out_h, out_w, device=X.device)
        for i in range(out_h):
            for j in range(out_w):
                i_step = i * self.stride
                j_step = j * self.stride
                Y[i, j] = (X[i_step:i_step+h, j_step:j_step+w] * K).sum()
        return Y

    def corr2d_multi_in(self, X, K):
        # X: (C_in, H, w), K: (C_in, k_h, k_w)
        # Sum all channels into one 2D slice
        return sum(self.corr2d(x, k) for x, k in zip(X, K))

    def corr2d_multi_in_out(self, X, K):
        # X: (C_in, H, W), K: (C_out, C_in, k_h, k_w)
        # Apply filters one by one to get C_out feature maps
        return torch.stack([self.corr2d_multi_in(X, k) for k in K])


class Pool2D(nn.Module):
    def __init__(self, pool_size=(2,2), stride=2, mode='max'):
        super().__init__()
        self.pool_size = pool_size
        self.stride = stride
        self.mode = mode

    def forward(self, x):
        B, C, H, W = x.shape
        p_h, p_w = self.pool_size
        
        out_h = (H - p_h) // self.stride + 1
        out_w = (W - p_w) // self.stride + 1
        
        output = torch.zeros(B, C, out_h, out_w, device=x.device)
        for b in range(B):
            for c in range(C):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start = i * self.stride
                        w_start = j * self.stride
                        window = x[b, c, h_start:h_start + p_h, w_start:w_start + p_w]
                        if self.mode == 'max':
                            output[b, c, i, j] = window.max()
                        elif self.mode == 'avg':
                            output[b, c, i, j] = window.mean()
        return output

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ReLU(nn.Module):
    def forward(self, x):
        return torch.maximum(x, torch.tensor(0.0, device=x.device))
