import torch
import torch.nn as nn

# Define single layer model
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc = nn.Linear(1, 1)

    def forward(self, x):
        return self.fc(x)

# Set the dtype for the input&label tensors
dtype = torch.float32

# Create single data point with label of 1
x = torch.tensor([[1]], dtype=dtype)
y = torch.tensor([[1]], dtype=dtype)

# Define loss function and optimizer
criterion = nn.MSELoss()

# Create model
model = Model()

# Define optimizer using model params
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Set min confidence
min_confidence = 0.75

# Train the model in a loop until the confidence > 75%
while True:
    # Forward pass
    output = model(x)
    loss = criterion(output, y)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Calculate confidence
    confidence = torch.sigmoid(output).item()

    if confidence >= min_confidence:
        break

# Print final confidence
print(f"I am {confidence * 100:.2f}% confident that 1 = 1!")
