import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data_path = "data/ns2d_1100_train.pkl"
#data_path = "data/heat2d_1100_train.pkl"
data_path = "data/inductor2d_1100_train.pkl"
data_path = "/home/maoy/data/PorousMedia/meltingFoam/DL_workspace/data/pickle_time10_res50/notime_train_1.pkl"
data_path = "/home/maoy/data/PorousMedia/meltingFoam/DL_workspace/data/debug_pickle/notime_test_1.pkl"

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

n_channels = X.shape[1]
n_sqrt = int(np.sqrt(n_channels))  # Calculate the square root of the number of channels
n_rows = n_sqrt if n_sqrt * n_sqrt >= n_channels else n_sqrt + 1
n_cols = n_sqrt if n_sqrt * n_sqrt == n_channels else n_channels // n_rows + (n_channels % n_rows > 0)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 15))  # Adjust the subplot size if necessary

for i in range(n_channels):
    ax = axes.flatten()[i]
    #sc = ax.scatter(X[:, 0], X[:, 1], c=Y[:, i], cmap='viridis', edgecolor='k', s=0.1)
    #sc = ax.hist2d(X[:, 0], X[:, 1], weights=Y[:, i], bins=100, cmap='viridis')
    sc = ax.hexbin(X[:, 0], X[:, 1], C=X[:,i], gridsize=50, cmap="viridis", reduce_C_function=np.mean)
    #sc = sns.kdeplot(x=X[:,0], y=X[:,1], weights=Y[:,i], fill=True, cmap="viridis")
    ax.set_title(f'Channel {i}')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    fig.colorbar(sc, ax=ax, label=f'Channel {i} Value')

plt.tight_layout()  # This will adjust spacing between subplots to minimize overlap
plt.show()

n_channels = Y.shape[1]
n_sqrt = int(np.sqrt(n_channels))  # Calculate the square root of the number of channels
n_rows = n_sqrt if n_sqrt * n_sqrt >= n_channels else n_sqrt + 1
n_cols = n_sqrt if n_sqrt * n_sqrt == n_channels else n_channels // n_rows + (n_channels % n_rows > 0)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 15))  # Adjust the subplot size if necessary

for i in range(n_channels):
    ax = axes.flatten()[i]
    #sc = ax.scatter(X[:, 0], X[:, 1], c=Y[:, i], cmap='viridis', edgecolor='k', s=0.1)
    #sc = ax.hist2d(X[:, 0], X[:, 1], weights=Y[:, i], bins=100, cmap='viridis')
    sc = ax.hexbin(X[:, 0], X[:, 1], C=Y[:,i], gridsize=50, cmap="viridis", reduce_C_function=np.mean)
    #sc = sns.kdeplot(x=X[:,0], y=X[:,1], weights=Y[:,i], fill=True, cmap="viridis")
    ax.set_title(f'Channel {i}')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    fig.colorbar(sc, ax=ax, label=f'Channel {i} Value')

plt.tight_layout()  # This will adjust spacing between subplots to minimize overlap
plt.show()

n_channels = len(inputF)
n_sqrt = int(np.sqrt(n_channels))  # Calculate the square root of the number of channels
n_rows = n_sqrt if n_sqrt * n_sqrt >= n_channels else n_sqrt + 1
n_cols = n_sqrt if n_sqrt * n_sqrt == n_channels else n_channels // n_rows + (n_channels % n_rows > 0)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 15))  # Adjust the subplot size if necessary

for i in range(n_channels):
    ax = axes.flatten()[i]
    if inputF[i].shape[1] == 2:
        sc = ax.scatter(inputF[i][:, 0], inputF[i][:, 1], cmap='viridis', edgecolor='k', s=1)
    else:
        sc = ax.hexbin(inputF[i][:, 0], inputF[i][:, 1], C=inputF[i][:, 2], gridsize=50, cmap="viridis", reduce_C_function=np.mean)
    ax.set_title(f'Channel {i}')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    fig.colorbar(sc, ax=ax, label=f'Channel {i} Value')

plt.tight_layout()  # This will adjust spacing between subplots to minimize overlap
plt.show()

