import torch
import matplotlib.pyplot as plt
from MLP import MLP, Dense, ReLU, MSE
from engine import Engine, GradientDescent, DataLoader

# 1. Setup Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on {device}")

# 2. Generate Synthetic Regression Data (y = x^2)
x_start, x_end = -3, 3
X = torch.linspace(x_start, x_end, 200, device=device).reshape(-1, 1)
# Adding noise
y = X**2 + 0.2 * torch.randn_like(X)

# 3. Define the Model Architecture
layers = [
    Dense(1, 32, device=device),
    ReLU(),
    Dense(32, 32, device=device),
    ReLU(),
    Dense(32, 1, device=device)
]
model = MLP(layers)

# 4. Setup Training Components
loss_fn = MSE()
optimizer = GradientDescent(model, learning_rate=0.01)
dataloader = DataLoader(X, y, batch_size=32, shuffle=True)
engine = Engine(model, loss_fn, optimizer)

# 5. Train
print("Starting Regression Training...")
history = engine.fit(dataloader, epochs=200, print_every=20)

# 6. Plot Results
# Plot Loss
engine.plot()

# Plot Predictions vs True Data
model_preds = model(X).cpu()
X_cpu = X.cpu()
y_cpu = y.cpu()

plt.scatter(X_cpu, y_cpu, s=10, label='True Data', alpha=0.5)
plt.plot(X_cpu, model_preds, color='red', linewidth=2, label='MLP Prediction')
plt.title("Regression Fit")
plt.legend()

plt.show()