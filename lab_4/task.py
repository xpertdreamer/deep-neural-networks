import torch 
import torch.nn as nn 
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('dataset_simple.csv')
# Here I tried to basically normalize input values to make loss a little lower
x_max = df.iloc[:, 0].max()
y_max = df.iloc[:, 1].max()
X = torch.tensor(df.iloc[0:, 0].values / x_max, 
                 dtype = torch.float32).reshape(-1, 1) # age as feature
y = torch.tensor(df.iloc[0:, 1].values / y_max, 
                 dtype = torch.float32).reshape(-1, 1) # income as target

input_size = X.shape[1]
hidden_size = 20
output_size = 1

class IncomePredNN(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super(IncomePredNN, self).__init__()
        self.fc1 = nn.Linear(in_size, hidden_size)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        return x
    
net = IncomePredNN(input_size, hidden_size, output_size)
loss = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1.0e-3)

epochs = 1500
for epoch in range(epochs):
    optimizer.zero_grad()
    pred = net.forward(X)
    loss_value = loss(pred, y)
    loss_value.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch} - loss = {loss_value.item():.4f}')

with torch.no_grad():
    predicted = net.forward(X).numpy()

for i in range(10):
    print(
    f'Age = {(X[i].item() * x_max):.0f}, '
    f'income = {(y[i].item() * y_max):.0f}, '
    f'pred = {(predicted[i][0] * y_max):.0f}'
    )

plt.figure()
plt.scatter(X.numpy().flatten() * x_max, 
            y.numpy().flatten() * y_max, 
            label="Actual", 
            color="blue",
            marker="o")
plt.scatter(X.numpy().flatten() * x_max, 
           predicted.flatten() * y_max, 
           label="Predicted", 
           color="red", 
           marker='x')
plt.xlabel('Age')
plt.ylabel('Income')
plt.legend()