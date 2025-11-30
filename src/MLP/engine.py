import torch
import matplotlib.pyplot as plt
from MLP import Dense

class Optimizer:
    def __init__(self, model, learning_rate):
        self.model = model
        self.learning_rate = learning_rate
    
    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        for layer in self.model.layers:
            if isinstance(layer, Dense):
                layer.grad_weight = torch.zeros_like(layer.weight)
                layer.grad_bias = torch.zeros_like(layer.bias)


class GradientDescent(Optimizer):
    def step(self):
        for layer in self.model.layers:
            if isinstance(layer, Dense):
                layer.weight -= layer.grad_weight * self.learning_rate
                layer.bias -= layer.grad_bias * self.learning_rate

class DataLoader:
    def __init__(self, X, y, batch_size=32, shuffle = True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = X.shape[0]
        
    def __iter__(self):
        if self.shuffle:
            idx = torch.randperm(self.num_samples, device=self.X.device)
        else:
            idx = torch.arange(self.num_samples, device=self.X.device)
        
        for i in range(0, self.num_samples, self.batch_size):
            batch_idx = idx[i:i + self.batch_size]
            yield self.X[batch_idx], self.y[batch_idx]

class Engine:
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.loss_history = []

    def fit(self, dataloader, epochs, print_every=10):
        self.loss_history = []
        for epoch in range(epochs):
            epoch_loss = 0
            num_batches = 0

            for X_batch, y_batch in dataloader:
                #forward
                pred = self.model(X_batch)
                loss = self.loss_fn(pred, y_batch)

                #backward
                grad = self.loss_fn.backward()
                self.model.backward(grad)
                #update
                self.optimizer.step()
                self.optimizer.zero_grad()

                epoch_loss += loss
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            self.loss_history.append(avg_loss)
            
            if (epoch + 1) % print_every == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss}")
                
        return self.loss_history

    def plot(self, log_scale=False):
        plt.plot([loss.to("cpu") for loss in self.loss_history])
        if log_scale:
            plt.yscale("log")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()