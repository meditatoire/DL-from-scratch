import torch

class Module:
    def __call__(self, *args):
        return self.forward(*args)

class Dense(Module):
    def __init__(self, in_dim, out_dim, device=None):
        super().__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        std = (2/in_dim)**0.5 #Kaiming initialization
        
        self.weight = torch.randn(in_dim, out_dim, device=device) * std
        self.bias = torch.zeros(1, out_dim, device=device)

        self.grad_weight = None
        self.grad_bias = None

        self.input = None #forward pass cache
    
    def forward(self, x):
        self.input = x
        return x.matmul(self.weight) + self.bias

    def backward(self, grad_output):
        self.grad_weight = self.input.T.matmul(grad_output)
        self.grad_bias = grad_output.sum(axis=0, keepdims=True)
        grad_input = grad_output.matmul(self.weight.T)
        return grad_input

class MLP(Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

class MSE(Module):
    def forward(self, pred, target):
        self.pred = pred
        self.target = target
        return ((pred - target) ** 2).mean()

    def backward(self):
        n = self.pred.shape[0]
        return 2.0 * (self.pred - self.target) / n

class CrossEntropyLoss(Module):
    def __init__(self):
        super().__init__()
        self.probs = None
        self.target = None
    #Using log-sum-exp trick for stability
    def forward(self, logits, target):
        #shifting logits to 0 for numerical stability
        max_logits = logits.max(dim=1, keepdim=True).values
        shifted_logits = logits - max_logits

        log_sum_exp = torch.log(torch.exp(shifted_logits).sum(dim=1, keepdim=True))

        if target.dim() == 1:
            target = target.unsqueeze(1)
        #We select the logit of the target class
        target_logits = shifted_logits.gather(1, target)

        loss = log_sum_exp - target_logits

        exp_logits = torch.exp(shifted_logits)
        self.probs = exp_logits / exp_logits.sum(dim=1, keepdim=True)
        self.target = target

        return loss.mean()

    def backward(self):
        #dL/dz = p - y where p is the probabilty and y is the correct target
        batch_size = self.probs.shape[0]
        grad = self.probs.clone() # p-0 for incorrect classes
        grad.scatter_add_(1, self.target, torch.full_like(self.target, -1.0, dtype=torch.float)) #substruct one from the correct class
        return grad / batch_size

class ReLU(Module):
    def forward(self, x):
        self.input = x
        return torch.maximum(x, torch.zeros_like(x))

    def backward(self, grad_output):
        relu_grad = (self.input > 0).float()
        return grad_output * relu_grad