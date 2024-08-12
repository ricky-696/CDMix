import numpy as np
import matplotlib.pyplot as plt

# Data from the updated image
data_updated = {
    'Top0': 70.60,
    'Top1': 69.02,
    'Top2': 71.21,
    'Top3': 71.81,
    'Top4': 70.42,
}

# Plotting the updated data
fig, ax = plt.subplots()
ax.grid(which='both', linestyle='--', linewidth=0.5, color='grey')

ax.set_xticks(range(len(data_updated)))
ax.set_xlim(0, len(data_updated) - 1)

ax.plot(range(len(data_updated)), list(data_updated.values()))

ax.set_title('Evaluation of Top K Class')
ax.set_xlabel('Top K')
ax.set_ylabel('Model Accuracy (%)')

plt.show()
plt.savefig('topk.jpg')
plt.close(fig)