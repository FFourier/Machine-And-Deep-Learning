import numpy as np
import matplotlib.pyplot as plt

# Training Data
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

#Test value
test=4

w = 1.0  # a random guess: random value


# our model forward pass
def forward(x):
    return x * w


# Loss function
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2


# compute gradient
def gradient(x, y):  # d_loss/d_w
    return 2 * x * (x * w - y)



# Before training
print(f'\nPrediction (before training) of {test} = {forward(test)}\n')

#Lists for graph
weights=[]
losses=[]

# Training loop
for epoch in range(10):
    print(f'\t\tx_v \ty_v  \tgrad')
    for x_val, y_val in zip(x_data, y_data):
        # Compute derivative w.r.t to the learned weights
        # Update the weights
        # Compute the loss and print progress
        grad = gradient(x_val, y_val)
        w = w - 0.01 * grad
        print(f'\tgrad:\t{x_val:.2f} \t{y_val:.2f} \t{grad:.2f}')
        l = loss(x_val, y_val)
    print(f'\nProgress: {epoch}\tw = {w:.2f}\tloss = {l:.2f}\n')
    weights.append(w)
    losses.append(l)
    print(f'----------------------------------------------------\n')


    
# After training
print(f'Predicted score (after training) of {test} = {forward(test):.4f}\n')

#Pltting

fig, axs = plt.subplots(1,3, constrained_layout=True)
fig.suptitle('Gradient Descent', fontsize=30)
fig.set_size_inches(16, 4)

axs[0].set_title('Input data')
axs[0].text(1,5,f'Where is going to be the number {test}?')
axs[0].plot(x_data,y_data,c='black')
for i in range(len(x_data)):
    axs[0].scatter(x_data[i],y_data[i],c='red',s=80)

axs[1].set_title('Loss calculation')
axs[1].set_ylabel('Loss')
axs[1].set_xlabel('Weight')
axs[1].plot(weights, losses,c='black')
axs[1].text(1.55,3,f'Iterations: {epoch+1}')
axs[1].text(1.5,3.5,f'Min loss in: ({np.max(weights):.3f}, {np.min(losses):.3f})')
for i in range(len(weights)):
    axs[1].scatter(weights[i],losses[i],c='green')
    

x_data.append(test)
y_data.append(forward(test))

axs[2].set_title('Output data')
axs[2].plot(x_data,y_data,c='black')
for i in range(len(x_data)):
    axs[2].scatter(x_data[i],y_data[i],c='red',s=80)
axs[2].scatter(test,forward(test),c='green',s=80)
axs[2].text(2.5,3,f'Output: ({test}, {forward(test):.3f})')

plt.show()