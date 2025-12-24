import torch
import matplotlib.pyplot as plt

class DataLoader:
    def __init__(self, X, y, batch_size=32, shuffle=True):
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
            
    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size

class Engine:
    def __init__(self, model, loss_fn, optimizer, device=None):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.train_loss_history = []
        self.test_loss_history = []
        self.test_acc_history = []

    def fit(self, train_loader, test_loader=None, epochs=10):
        print(f"Starting training on {self.device}...")
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            num_batches = len(train_loader)
            
            # Training loop
            for i, (X, y) in enumerate(train_loader):
                X, y = X.to(self.device), y.to(self.device)
                
                # Forward pass
                outputs = self.model(X)
                loss = self.loss_fn(outputs, y)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                
                if (i + 1) % max(1, num_batches // 5) == 0 or (i + 1) == num_batches:
                    print(f"Epoch {epoch+1}/{epochs} - Batch {i+1}/{num_batches} - Loss: {loss.item():.4f}")
            
            avg_train_loss = total_loss / num_batches
            self.train_loss_history.append(avg_train_loss)
            
            # Validation loop
            if test_loader:
                avg_test_loss, acc = self.evaluate(test_loader)
                self.test_loss_history.append(avg_test_loss)
                self.test_acc_history.append(acc)
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}, Test Acc: {acc:.2f}%")
            else:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}")

    def evaluate(self, loader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)
                outputs = self.model(X)
                loss = self.loss_fn(outputs, y)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        
        acc = 100 * correct / (total if total > 0 else 1)
        avg_loss = total_loss / len(loader)
        return avg_loss, acc

    def plot(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.train_loss_history, label="Train Loss")
        if self.test_loss_history:
            ax1.plot(self.test_loss_history, label="Test Loss")
        ax1.set_title("Loss History")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        
        # Accuracy plot
        if self.test_acc_history:
            ax2.plot(self.test_acc_history, label="Test Accuracy", color='green')
            ax2.set_title("Accuracy History")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Accuracy (%)")
            ax2.legend()
        
        plt.show()

    def plot_predictions(self, loader, num_images=10):
        self.model.eval()
        images, labels, preds = [], [], []
        
        with torch.no_grad():
            for X, y in loader:
                X_dev = X.to(self.device)
                outputs = self.model(X_dev)
                _, predicted = torch.max(outputs.data, 1)
                
                images.extend(X.cpu())
                labels.extend(y.cpu())
                preds.extend(predicted.cpu())
                
                if len(images) >= num_images:
                    break
        
        fig = plt.figure(figsize=(15, 7))
        for i in range(num_images):
            ax = fig.add_subplot(2, 5, i + 1, xticks=[], yticks=[])
            img = images[i].squeeze().numpy()
            ax.imshow(img, cmap='gray')
            color = "green" if preds[i] == labels[i] else "red"
            ax.set_title(f"True: {labels[i]}\nPred: {preds[i]}", color=color)
        plt.tight_layout()
        plt.show()


