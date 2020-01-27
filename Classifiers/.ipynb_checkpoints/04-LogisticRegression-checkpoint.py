#! Libraries

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import statistics as st

#! Data

#* Samples number
n = 100
#* Labels
h = n // 2
dimen = 2

#* Generating random data
data = np.random.randn(n, dimen)

#* Data plotting

fig, axs = plt.subplots(1, 4, sharey=True, sharex=True, constrained_layout=True)
fig.suptitle("Preparing the data", fontsize=20)
fig.set_size_inches(16, 4)

axs[0].scatter(data[:, 0], data[:, 1], c="yellow", s=55, alpha=0.5)
axs[0].set_title("Random data")

data = data * 3

axs[1].scatter(data[:, 0], data[:, 1], c="green", s=55, alpha=0.5)
axs[1].set_title("Spreading data")

data[:h, :] = data[:h, :] - 3 * np.ones((h, dimen))
data[h:, :] = data[h:, :] + 3 * np.ones((h, dimen))

axs[2].scatter(data[:, 0], data[:, 1], c="red", s=55, alpha=0.5)
axs[2].set_title("Separating Data")

colors = ["blue", "red"]
color = np.array([colors[0]] * h + [colors[1]] * h).reshape(n)

axs[3].scatter(data[:, 0], data[:, 1], c=color, s=55, alpha=0.5)
axs[3].set_title("Prepared data")

plt.show()

#* Splitting the data in inputs (x) and outputs (y)
target = np.array([0] * h + [1] * h).reshape(n, 1)
x = torch.from_numpy(data).float().requires_grad_(True)
y = torch.from_numpy(target).float()

#! Model

#* Building the model
model = nn.Sequential(nn.Linear(2, 1), nn.Sigmoid())

#* Loss function and optimizer method
loss_function = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.025)


#! Training loop

losses = []
iterations = 100

for i in tqdm(range(iterations)):

    result = model(x)
    loss = loss_function(result, y)

    losses.append(loss.data)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()


#* Passing data through the model
prediction = model(x)

#* List with the corresponding labels
prediction = ["blue" if prediction[i] < 0.5 else "red" for i in range(len(prediction))]

#* weights
w = list(model.parameters())

w0 = w[0].data.numpy()


#! Visualization

#* Parameters to plot the line
x_axis = np.linspace(torch.min(x).data, torch.max(x).data, len(x))
y_axis = -(w[1].data.numpy() + x_axis * w0[0][0]) / w0[0][1]


#* Output plotting

fig, axs = plt.subplots(1, 3, constrained_layout=True)
fig.suptitle("Train and test", fontsize=20)
fig.set_size_inches(16, 4)

axs[0].plot(range(iterations), losses)
axs[0].set_title("Loss")

axs[1].plot(x_axis, y_axis, "g--")
axs[1].scatter(data[:, 0], data[:, 1], c=color, s=55, alpha=0.5)
axs[1].set_title("Output with error")

axs[2].plot(x_axis, y_axis, "g--")
for i in range(len(x)):
    axs[2].scatter(x[i, 0].data, x[i, 1].data, s=55, alpha=0.5, c=prediction[i])
axs[2].set_title("Output")

plt.show()

pred = model(x)
pred = [0 if pred[i] < 0.5 else 1 for i in range(len(pred))]
# Comparing each value with its reference and calculating the mean
pred = st.mean([1 if pred[i] == int(y[i]) else 0 for i in range(len(pred))])
print(f"Loss: {float(loss)}")
print(f"Accuracy: {pred}")

