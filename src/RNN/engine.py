import torch
import torch.nn as nn
import random

def generate_text(model, prefix, num_preds, vocab, char_to_idx, idx_to_char):
    model.eval()
    with torch.no_grad():
        state = None
        outputs = [char_to_idx[prefix[0]]]
        def get_input():
            return torch.tensor([outputs[-1]], device=model.device).reshape(1, 1)
        
        # Warm up with prefix
        for char in prefix[1:]:
            X = nn.functional.one_hot(get_input(), num_classes=len(vocab)).float()
            _, states = model(X, state)
            state = states[-1]
            outputs.append(char_to_idx[char])
            
        # Predict num_preds characters
        for _ in range(num_preds):
            X = nn.functional.one_hot(get_input(), num_classes=len(vocab)).float()
            y, states = model(X, state)
            state = states[-1]
            outputs.append(int(y[0].argmax(dim=1).item()))
            
    model.train()
    return ''.join([idx_to_char[i] for i in outputs])

class DataLoader():
    def __init__(self, corpus_indices, encoded_text, seq_length, batch_size):
        self.corpus_indices = corpus_indices
        self.encoded_text = encoded_text
        self.seq_length = seq_length
        self.batch_size = batch_size
    
    def __iter__(self):
        num_examples = (self.encoded_text.shape[0] - 1) // self.seq_length
        example_indices = list(range(0, num_examples * self.seq_length, self.seq_length))
        random.shuffle(example_indices)
        
        for i in range(0, len(example_indices), self.batch_size):
            batch_indices = example_indices[i : i + self.batch_size]
            if len(batch_indices) < self.batch_size:
                continue
            
            X = torch.stack([self.encoded_text[j : j + self.seq_length] for j in batch_indices])
            Y = torch.stack([self.corpus_indices[j + 1 : j + self.seq_length + 1] for j in batch_indices])
            
            yield X.transpose(0, 1), Y.transpose(0, 1)

    def __len__(self):
        num_examples = (self.encoded_text.shape[0] - 1) // self.seq_length
        return num_examples // self.batch_size

class Training():
    def __init__(self, model, data_loader, optimizer, epochs, vocab, char_to_idx, idx_to_char, grad_clip=1.0):
        self.model = model
        self.data_loader = data_loader
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optimizer
        self.epochs = epochs
        self.vocab = vocab
        self.char_to_idx = char_to_idx
        self.idx_to_char = idx_to_char
        self.grad_clip = grad_clip

    def train(self):
        self.model.to(self.model.device)
        losses = []
        for epoch in range(1, self.epochs + 1):
            total_loss = 0
            for X, Y in self.data_loader:
                X, Y = X.to(self.model.device), Y.to(self.model.device)
                
                y_hat, _ = self.model(X)
                y_hat = torch.stack(y_hat).reshape(-1, y_hat[0].shape[-1])
                Y = Y.reshape(-1)
                
                loss = self.criterion(y_hat, Y.long())
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(self.data_loader)
            losses.append(avg_loss)
            print(f"Epoch {epoch} loss: {avg_loss}")
            sample = generate_text(self.model, 'the ', 50, self.vocab, self.char_to_idx, self.idx_to_char)
            print(f"Sample: {sample}")
        return losses

class Inference():
    def __init__(self, model, vocab, char_to_idx, idx_to_char):
        self.model = model
        self.vocab = vocab
        self.char_to_idx = char_to_idx
        self.idx_to_char = idx_to_char
    
    def predict(self, prefix, num_preds):
        return generate_text(self.model, prefix, num_preds, self.vocab, self.char_to_idx, self.idx_to_char)
