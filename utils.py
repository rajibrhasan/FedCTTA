
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import torch

file_path = 'logging/cifar10_test/tta_adapt/niid/ours_random_output_adapt_fedpl_client_lp_1_seed3_1738014263/collaboration.pkl'
with open(file_path, 'rb') as f:
    collaboration_graph = pickle.load(f)

# Stack the collaboration graphs into a single tensor
collaboration_graph = torch.stack(collaboration_graph).cpu()


plt.figure(figsize=(8, 6))
sns.heatmap(collaboration_graph[-1], annot=False, cmap="viridis", cbar=True) # Set cbar to True to show the colorbar
plt.title("Collaborative Matrix Heatmap")
plt.savefig('collab_mat_original.png')
plt.show()