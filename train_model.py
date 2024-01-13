import os
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

# Define the GCN Model
class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return x

# Load the Graph Data
def load_graph_data():
    graph_files = os.listdir('saved_graphs')
    datasets = []

    for file in graph_files:
        data = torch.load(f'saved_graphs/{file}')
        datasets.append((data.x[:, :-1], data.x[:, -1], data.edge_index))  # Assuming the last column is ground truth

    return datasets

datasets = load_graph_data()

# Train-Test Split
train_size = int(0.8 * len(datasets))
train_dataset = datasets[:train_size]
test_dataset = datasets[train_size:]

# Initialize the Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(num_features=datasets[0][0].size(1), hidden_channels=16).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()

# Train the Model
model.train()
for epoch in range(200):
    total_loss = 0
    for x, y, edge_index in train_dataset:
        x, y, edge_index = x.to(device), y.to(device), edge_index.to(device)
        optimizer.zero_grad()
        out = model(Data(x=x, edge_index=edge_index)).squeeze()
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch}, Loss: {total_loss / len(train_dataset)}')

# Evaluate the Model
model.eval()
test_loss = 0
with torch.no_grad():
    for x, y, edge_index in test_dataset:
        x, y, edge_index = x.to(device), y.to(device), edge_index.to(device)
        pred = model(Data(x=x, edge_index=edge_index)).squeeze()
        test_loss += criterion(pred, y).item()
print(f'Test MSE: {test_loss / len(test_dataset)}')
