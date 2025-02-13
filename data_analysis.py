import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pickle
import matplotlib.animation as animation

import os
from mpl_toolkits.mplot3d import Axes3D

resol = 50
data_path = f"/home/maoy/data/PorousMedia/meltingFoam/DL_workspace/data/pickle_time10_res{resol}/porousmelting_train.pkl"
directories_path = f"/home/maoy/data/PorousMedia/meltingFoam/DL_workspace/data/pickle_time10_res{resol}/directories.pkl"

data_all = pickle.load(open(data_path, "rb"))[:]
directories_all = pickle.load(open(directories_path, "rb"))[:]

n_samples = len(data_all)
input_features = ['x', 'y', 'time', 'Ux', 'Uy', 'pressure', 'temperature', 'liquid_fraction', 'solid_label']
output_features = ['Ux', 'Uy', 'pressure', 'temperature', 'liquid_fraction']
print(n_samples)

n_times = 11
n_x = resol
n_points = n_x * n_x + 4*n_x

X_data = []
Y_data = []

for ib in range(n_samples):
    inp = np.array(data_all[ib][0])
    X_data.append(inp)
    #inp = inp.reshape(ntimes,-1)
    out = np.array(data_all[ib][1])
    Y_data.append(out)
    #out = out.reshape(ntimes,-1)
    #print(inp.shape)
X_data = np.array(X_data)
Y_data = np.array(Y_data)

# quick view of the data trend
time = np.linspace(0, 2000, n_times)
mean_T_cases = []
std_T_cases = []
mean_alpha_cases = []
std_alpha_cases = []

for ib in range(n_samples):
    mean = np.mean(Y_data[ib,:,3].reshape(n_times,n_points),axis=1)
    std = np.std(Y_data[ib,:,3].reshape(n_times,n_points),axis=1)
    mean_T_cases.append(mean)
    std_T_cases.append(std)

    mean = np.mean(Y_data[ib,:,4].reshape(n_times,n_points),axis=1)
    std = np.std(Y_data[ib,:,4].reshape(n_times,n_points),axis=1)
    mean_alpha_cases.append(mean)
    std_alpha_cases.append(std)

# Create 3D plot
fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(221, projection='3d')
ax2 = fig.add_subplot(222, projection='3d')
ax3 = fig.add_subplot(223, projection='3d')
ax4 = fig.add_subplot(224, projection='3d')

# Plot each case in the depth (z) direction
line1s = []
line2s = []
line3s = []
line4s = []
for i in range(n_samples):
    case_name = os.path.basename(directories_all[i])
    line1, = ax1.plot(time, mean_T_cases[i], zs=i, zdir='y', label=f"{case_name}", alpha=0.6)
    line2, = ax2.plot(time, std_T_cases[i], zs=i, zdir='y', label=f"{case_name}", alpha=0.6)
    line3, = ax3.plot(time, mean_alpha_cases[i], zs=i, zdir='y', label=f"{case_name}", alpha=0.6)
    line4, = ax4.plot(time, std_alpha_cases[i], zs=i, zdir='y', label=f"{case_name}", alpha=0.6)
    line1s.append(line1)
    line2s.append(line2)
    line3s.append(line3)
    line4s.append(line4)

# Set labels
ax1.set_xlabel("Time")
ax1.set_ylabel("Cases")
ax1.set_zlabel("Mean")
ax1.set_title("3D Visualization of Mean Trends")

# Function to handle picking events
def on_pick(event):
    ind = lines.index(event.artist)
    print(f"Selected Case: {ind + 1}")

# Add picking
fig.canvas.mpl_connect('pick_event', on_pick)

# Enable rotation
plt.show()

# average X field values 
mean_values = np.mean(X_data, axis=(0, 1))
std_values = np.std(X_data, axis=(0, 1))

for i, feature in enumerate(input_features):
    print(f'input {feature}: Mean = {mean_values[i]:.4f}, Std = {std_values[i]:.4f}')

plt.figure(figsize=(15, 12))
for i, feature in enumerate(input_features):
    plt.subplot(3, 3, i + 1)
    sns.histplot(X_data[:, :, i].flatten(), bins=50, kde=True)
    plt.title(f'Distribution of {feature}')
plt.tight_layout()
path = "Xfeature.png"
plt.savefig(path)
plt.show()

# average Y field values 
mean_values = np.mean(Y_data, axis=(0, 1))
std_values = np.std(Y_data, axis=(0, 1))

for i, feature in enumerate(output_features):
    print(f'output {feature}: Mean = {mean_values[i]:.4f}, Std = {std_values[i]:.4f}')

plt.figure(figsize=(15, 12))
for i, feature in enumerate(output_features):
    plt.subplot(3, 3, i + 1)
    sns.histplot(Y_data[:, :, i].flatten(), bins=50, kde=True)
    plt.title(f'Distribution of {feature}')
plt.tight_layout()
path = "Yfeature.png"
plt.savefig(path)
plt.show()


# average field values over time
avg_ux_over_time = np.mean(Y_data[:, :, 0].reshape(n_samples, n_times, n_points), axis=(0, 2))
avg_uy_over_time = np.mean(Y_data[:, :, 1].reshape(n_samples, n_times, n_points), axis=(0, 2))
avg_press_over_time = np.mean(Y_data[:, :, 2].reshape(n_samples, n_times, n_points), axis=(0, 2))
avg_temp_over_time = np.mean(Y_data[:, :, 3].reshape(n_samples, n_times, n_points), axis=(0, 2))
avg_liquid_over_time = np.mean(Y_data[:, :, 4].reshape(n_samples, n_times, n_points), axis=(0, 2))

fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(6, 12))

# Plot each function in its respective subplot
axes[0].plot(range(n_times), avg_ux_over_time, label="mean Ux", color="b")
axes[0].set_ylabel('mean Ux Value')
axes[0].legend()

axes[1].plot(range(n_times), avg_uy_over_time, label="mean Uy", color="b")
axes[1].set_ylabel('mean Uy Value')
axes[1].legend()

axes[2].plot(range(n_times), avg_press_over_time, label="mean press", color="b")
axes[2].set_ylabel('mean press Value')
axes[2].legend()

axes[3].plot(range(n_times), avg_temp_over_time, label="mean T", color="b")
axes[3].set_ylabel('mean T Value')
axes[3].legend()

axes[4].plot(range(n_times), avg_liquid_over_time, label="mean fraction", color="b")
axes[4].set_ylabel('mean fraction Value')
axes[4].legend()

plt.tight_layout()
path = "mean_over_trend.png"
plt.savefig(path)
plt.show()

max_ux_over_time = np.max(Y_data[:, :, 0].reshape(n_samples, n_times, n_points), axis=(0, 2))
max_uy_over_time = np.max(Y_data[:, :, 1].reshape(n_samples, n_times, n_points), axis=(0, 2))
max_press_over_time = np.max(Y_data[:, :, 2].reshape(n_samples, n_times, n_points), axis=(0, 2))
max_temp_over_time = np.max(Y_data[:, :, 3].reshape(n_samples, n_times, n_points), axis=(0, 2))
max_liquid_over_time = np.max(Y_data[:, :, 4].reshape(n_samples, n_times, n_points), axis=(0, 2))

fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(6, 12))

# Plot each function in its respective subplot
axes[0].plot(range(n_times), max_ux_over_time, label="max Ux", color="b")
axes[0].set_ylabel('max Ux Value')
axes[0].legend()

axes[1].plot(range(n_times), max_uy_over_time, label="max Uy", color="b")
axes[1].set_ylabel('max Uy Value')
axes[1].legend()

axes[2].plot(range(n_times), max_press_over_time, label="max press", color="b")
axes[2].set_ylabel('max press Value')
axes[2].legend()

axes[3].plot(range(n_times), max_temp_over_time, label="max T", color="b")
axes[3].set_ylabel('max T Value')
axes[3].legend()

axes[4].plot(range(n_times), max_liquid_over_time, label="max fraction", color="b")
axes[4].set_ylabel('max fraction Value')
axes[4].legend()

plt.tight_layout()
path = "max_over_trend.png"
plt.savefig(path)
plt.show()

min_ux_over_time = np.min(Y_data[:, :, 0].reshape(n_samples, n_times, n_points), axis=(0, 2))
min_uy_over_time = np.min(Y_data[:, :, 1].reshape(n_samples, n_times, n_points), axis=(0, 2))
min_press_over_time = np.min(Y_data[:, :, 2].reshape(n_samples, n_times, n_points), axis=(0, 2))
min_temp_over_time = np.min(Y_data[:, :, 3].reshape(n_samples, n_times, n_points), axis=(0, 2))
min_liquid_over_time = np.min(Y_data[:, :, 4].reshape(n_samples, n_times, n_points), axis=(0, 2))

fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(6, 12))

# Plot each function in its respective subplot
axes[0].plot(range(n_times), min_ux_over_time, label="min Ux", color="b")
axes[0].set_ylabel('min Ux Value')
axes[0].legend()

axes[1].plot(range(n_times), min_uy_over_time, label="min Uy", color="b")
axes[1].set_ylabel('min Uy Value')
axes[1].legend()

axes[2].plot(range(n_times), min_press_over_time, label="min press", color="b")
axes[2].set_ylabel('min press Value')
axes[2].legend()

axes[3].plot(range(n_times), min_temp_over_time, label="min T", color="b")
axes[3].set_ylabel('min T Value')
axes[3].legend()

axes[4].plot(range(n_times), min_liquid_over_time, label="min fraction", color="b")
axes[4].set_ylabel('min fraction Value')
axes[4].legend()

plt.tight_layout()
path = "min_over_trend.png"
plt.savefig(path)
plt.show()


# animation

# Create figure
fig, ax = plt.subplots(figsize=(6, 5))
sc = ax.scatter([], [], c=[], cmap="coolwarm", s=10)  # Empty scatter plot
cbar = plt.colorbar(sc, ax=ax, label="Temperature")

t = 0
case_id = 400
x = X_data[case_id, :, 0].reshape(n_times, n_points)[t, :]
y = X_data[case_id, :, 1].reshape(n_times, n_points)[t, :]
ax.set_xlim(x.min(), x.max())  # Adjust based on data range
ax.set_ylim(y.min(), y.max())
ax.set_xlabel("X")
ax.set_ylabel("Y")
title = ax.set_title("Temperature Distribution at Time 0")

# Animation update function
def update(t):
    x = X_data[case_id, :, 0].reshape(n_times, n_points)[t, :]
    y = X_data[case_id, :, 1].reshape(n_times, n_points)[t, :]
    temperature = Y_data[case_id, :, 3].reshape(n_times, n_points)[t, :]

    sc.set_clim(vmin=np.min(temperature), vmax=np.max(temperature))
    cbar.update_normal(sc)

    sc.set_offsets(np.c_[x, y])  # Update positions
    sc.set_array(temperature)  # Update color data
    title.set_text(f"Temperature Distribution at Time {t}")

    return sc,

# Create animation
ani = animation.FuncAnimation(fig, update, frames=n_times, interval=200)

# Save animation as GIF
ani.save("temperature_evolution.gif", writer="pillow", fps=10)

# Show animation (optional)
plt.show()
