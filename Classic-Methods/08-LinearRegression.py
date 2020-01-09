import torch
import matplotlib.pyplot as plt
from torch import nn

x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[2.0], [4.0], [6.0]])


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(1, 1) 

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

model = Model()

criterion = torch.nn.MSELoss(reduction='sum')
    # reduction (string, optional) – Specifies the reduction to apply to the output:  the output will be summed
    
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

losses=[]

for epoch in range(10):

    y_pred = model(x_data)

    loss = criterion(y_pred, y_data)
    losses.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

hour_var = torch.Tensor([[4.0]])
y_pred = model(hour_var)


plt.plot(range(len(losses)),losses, c='red', lw=3)
plt.title('Loss', fontsize=25)
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.text(3,max(losses)/2,f'Final loss: {(losses[-1]):.3f}',fontsize=15, bbox=dict(facecolor='red', alpha=0.3))
plt.show()