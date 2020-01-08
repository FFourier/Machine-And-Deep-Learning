import torch

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
w = torch.tensor([1.0], requires_grad=True)

# our model forward pass
def forward(x):
    return x * w

# Loss function
def loss(y_pred, y_val):
    return (y_pred - y_val) ** 2

#test value
test=4

# Before training
print(f'\nPrediction (before training) of {test} = {forward(test).item()}\n')

# Training loop
for epoch in range(10):
    print(f'\t\tx_v \ty_v  \tgrad')
    for x_val, y_val in zip(x_data, y_data):
        
        y_pred = forward(x_val) # 1) Forward pass
        l = loss(y_pred, y_val) # 2) Compute loss
        l.backward()            # 3) Back propagation to update weights
        
        print(f'\tgrad:\t{x_val:.2f} \t{y_val:.2f} \t{w.grad.item():.2f}')
        
        w.data = w.data - 0.01 * w.grad.item()
        #0.01 as learning rate
        
        # Manually zero the gradients after updating weights
        w.grad.data.zero_()
        
    print(f'\nProgress: {epoch}\tw = {w.data.item():.2f}\tloss = {l.item():.2f}\n')
    print(f'----------------------------------------------------\n')


# After training
print(f'Predicted score (after training) of {test} = {forward(test).item():.4f}\n')