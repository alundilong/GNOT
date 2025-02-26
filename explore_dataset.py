import pickle
import numpy as np
import matplotlib.pyplot as plt

data_path = "data/ns2d_1100_train.pkl"
#data_path = "data/heat2d_1100_train.pkl"
data_path = "data/inductor2d_1100_train.pkl"

data_all = pickle.load(open(data_path, "rb")) #[:16]

batch = data_all[0]
X = batch[0]
Y = batch[1]
theta = batch[2]
inputF = batch[3]

print(len(X),len(Y),len(theta),len(inputF))

X = np.array(X)
Y = np.array(Y)
theta = np.array(theta)
#inputF = np.array(inputF)
print(X.shape)
print(Y.shape)
#print(theta)
print(inputF[0].shape)

# Create subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # 1 row, 3 columns

# Titles for each channel
channel_titles = ['C1', 'C2', 'C3']
# Loop through each channel and plot
for i in range(3):
    sc = axes[i].scatter(X[:, 0], X[:, 1], c=Y[:, i], cmap='viridis', edgecolor='k')
    axes[i].set_title(channel_titles[i])
    axes[i].set_xlabel("X Coordinate")
    axes[i].set_ylabel("Y Coordinate")
    fig.colorbar(sc, ax=axes[i], label=f'Y[:, {i}] Value')

# Show the figure
plt.tight_layout()
plt.show()

# Scatter Plot
plt.figure(figsize=(6, 6))
plt.scatter(inputF[0][:, 0], inputF[0][:, 1], s=10, color='blue', alpha=0.6, edgecolor='k')

# Labels and Title
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.title("Scatter Plot of inputF[0]")

# Show plot
plt.grid(True)
plt.show()
