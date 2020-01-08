import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]


# our model for the forward pass
def forward(x):
    return x * w


# Loss function
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2

# List of weights/Mean square Error (Mse) for each input
w_list = []
mse_list = []

for w in np.arange(0.0, 4.1, 0.1):
    
    # Print the weights and initialize the lost
    print(f'\nw = {w:.1f}\n')
    l_sum = 0
    
    print(f'\tx_v \ty_v \ty_p \t lo')
    for x_val, y_val in zip(x_data, y_data):
        # For each input and output, calculate y_hat
        # Compute the total loss and add to the total error
        
        y_pred_val = forward(x_val)
        l = loss(x_val, y_val)
        l_sum += l
        print(f'\t{x_val:.2f} \t{y_val:.2f} \t{y_pred_val:.2f} \t{l:.2f}')
        
    # Now compute the Mean squared error (mse) of each
    # Aggregate the weight/mse from this run
    print(f'\nMSE= {(l_sum / 3):.4f}\n')
    print(f'----------------------------------------')
    w_list.append(w)
    mse_list.append(l_sum / 3)

# Plot it all
plt.plot(w_list, mse_list)
plt.title('Loss vs weight')
plt.ylabel('Loss')
plt.xlabel('w')
plt.show()