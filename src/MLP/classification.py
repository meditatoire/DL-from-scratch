import torch
import matplotlib.pyplot as plt
from MLP import MLP, Dense, ReLU, CrossEntropyLoss
from engine import Engine, GradientDescent, DataLoader

#Setup Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on {device}")

#Generate Synthetic Classification Data (4 Clusters)
num_samples = 100  # Samples per cluster

# Create 4 distinct centers
centers = [
    [2.0, 2.0],   # Class 0: Top Right
    [-2.0, -2.0], # Class 1: Bottom Left
    [-2.0, 2.0],  # Class 2: Top Left
    [2.0, -2.0]   # Class 3: Bottom Right
]

X_list = []
y_list = []

for class_idx, center in enumerate(centers):
    # Generate random points around the center
    cluster_data = torch.randn(num_samples, 2, device=device) * 0.8 + torch.tensor(center, device=device)
    cluster_labels = torch.full((num_samples,), class_idx, dtype=torch.long, device=device)
    
    X_list.append(cluster_data)
    y_list.append(cluster_labels)

X = torch.cat(X_list)
y = torch.cat(y_list)

#Define the Model Architecture
# Input: 2 features -> Hidden: 32 -> Hidden: 32 -> Output: 4 classes
layers = [
    Dense(2, 32, device=device),
    ReLU(),
    Dense(32, 32, device=device),
    ReLU(),
    Dense(32, 4, device=device) # Output dim must be 4 for 4 clusters
]
model = MLP(layers)

#Setup Training Components
loss = CrossEntropyLoss()
optimizer = GradientDescent(model, learning_rate=0.05)
dataloader = DataLoader(X, y, batch_size=32, shuffle=True)
engine = Engine(model, loss, optimizer)

#Train
print("Starting 4-Cluster Classification Training...")
history = engine.fit(dataloader, epochs=50, print_every=10)

#Plotting the loss curve
engine.plot()

#Evaluation & Prediction
model_preds_logits = model(X)
predicted_labels = torch.argmax(model_preds_logits, dim=1)

accuracy = (predicted_labels == y).float().mean()
print(f"Final Accuracy: {accuracy.item() * 100:.2f}%")

#Visualization: True Labels vs Predictions
plt.figure(figsize=(12, 5))

# Plot True Labels
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0].cpu(), X[:, 1].cpu(), c=y.cpu(), cmap='viridis', edgecolor='k', s=50)
plt.title("Ground Truth Labels")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

# Plot Model Predictions
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0].cpu(), X[:, 1].cpu(), c=predicted_labels.cpu(), cmap='viridis', edgecolor='k', s=50)
plt.title(f"Model Predictions (Acc: {accuracy.item()*100:.1f}%)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.tight_layout()
plt.show()